"""Quantum-chemistry primitives as fused tensor-contraction kernels.

These are domain-specific entry points where the whole computation —
contraction, elementwise shaping, reduction — lives in a single NKI
program. PSUM holds the contraction, the Vector Engine folds the
elementwise, and an SBUF scalar accumulator collects the reduction.
One HBM round trip instead of several.

This is the differentiator vs ``torch.einsum`` (which composes ops)
and ``trnblas.gemm`` (which is just the contraction). The architecture
is the product.
"""

from __future__ import annotations

import torch

from .nki.dispatch import _ao_to_mo_route, _mp2_energy_route


def mp2_energy(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Second-order Møller–Plesset correlation energy, density-fitted.

    Computes

        E_MP2 = Σ_{ijab} T_{ijab} (2 T_{ijab} - T_{ijba}) / Δ_{ijab}

    where

        T_{ijab}  = Σ_P B[i,a,P] B[j,b,P]
        Δ_{ijab}  = ε_i + ε_j - ε_a - ε_b

    On Trainium the contraction, the spin-adapted numerator, the
    denominator and the reduction run inside NKI kernels: ``T`` accumulates
    in PSUM, is folded on the Vector Engine and never reaches device memory.

    - Native TorchNeuron path (Trn2/Trn3): the row kernel
      (``mp2_energy_row_kernel``, one launch per occupied row, pair symmetry,
      both physical cores at LNC=2). ``"auto"`` uses it for device-resident
      ``B`` once each row launch carries at least 3 GFLOP (padded
      ``2·(nocc//2+1)·nvir²·naux``; e.g. ``nocc=32, nvir=320, naux=1024``).
      There it measured 1.25-1.9x faster than aten on Trn2, with the gap
      growing with size (#51). Smaller problems,
      and host ``B``, use the aten formulation. ``set_backend("nki")`` uses
      the row kernel whenever it fits in SBUF. ``TRNTENSOR_MP2_ENERGY=
      auto|rows|aten|legacy`` overrides the choice.
    - PyTorch/XLA path (Trn1): the original single-launch kernel.

    The aten formulation contracts ``T`` in occupied-index chunks, so its
    peak memory is bounded.

    Parameters
    ----------
    B : (nocc, nvir, naux) tensor
        Density-fitted electron-repulsion-integral coefficients.
    eps_occ : (nocc,) tensor
        Occupied molecular orbital energies.
    eps_vir : (nvir,) tensor
        Virtual molecular orbital energies.

    Returns
    -------
    0-D tensor
        The correlation energy E_MP2.
    """
    if B.dim() != 3:
        raise ValueError(f"B must be 3D (nocc, nvir, naux); got shape {tuple(B.shape)}")
    nocc, nvir, _naux = B.shape
    if eps_occ.shape != (nocc,):
        raise ValueError(f"eps_occ must have shape ({nocc},); got {tuple(eps_occ.shape)}")
    if eps_vir.shape != (nvir,):
        raise ValueError(f"eps_vir must have shape ({nvir},); got {tuple(eps_vir.shape)}")

    route = _mp2_energy_route(B)
    if route != "aten":
        from .nki import dispatch as _d

        try:
            if route == "rows":
                return _d._nki_mp2_energy_rows(B, eps_occ, eps_vir)
            if _d._on_native_path():
                _d._warn_native_kernel_deprecated(
                    "original mp2_energy",
                    "the Trn2 row kernel replaces it (TRNTENSOR_MP2_ENERGY=rows)",
                )
            return _d._nki_mp2_energy(B, eps_occ, eps_vir)
        except Exception:
            import os

            if os.environ.get("TRNTENSOR_REQUIRE_NKI", "").lower() in ("1", "true", "yes"):
                raise
    return _cpu_mp2_energy(B, eps_occ, eps_vir)


# Elements of the per-chunk T block (chunk·nocc·nvir²) the aten MP2 path
# materializes at once: 2²⁴ fp32 elements = 64 MiB.
_MP2_CHUNK_ELEMS = 1 << 24


def _cpu_mp2_energy(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation in aten ops, on ``B``'s device.

    Vectorized over ``(j, a, b)`` and chunked over ``i``:
    ``T[i,j,a,b] = Σ_P B[i,a,P] B[j,b,P]`` for a chunk of ``i``, then
    ``Σ T (2T - T_ijba) / Δ``. ``T_ijba`` is the last-two-dims transpose
    of the same block, so each chunk is self-contained. Replaces the
    former nocc² Python loop (``examples/df_mp2_einsum.py``), which on a
    Trainium device ran 10-40x slower than this form (#41).
    """
    nocc, nvir, _naux = B.shape
    chunk = max(1, _MP2_CHUNK_ELEMS // max(1, nocc * nvir * nvir))
    eps_occ = eps_occ.to(B.device)
    eps_vir = eps_vir.to(B.device)
    ev = eps_vir[:, None] + eps_vir[None, :]  # (a, b)
    e = B.new_zeros(())
    for i0 in range(0, nocc, chunk):
        Bi = B[i0 : i0 + chunk]
        T = torch.einsum("iap,jbp->ijab", Bi, B)
        denom = (eps_occ[i0 : i0 + chunk, None] + eps_occ[None, :])[:, :, None, None] - ev
        e = e + (T * (2 * T - T.transpose(2, 3)) / denom).sum()
    return e


def ao_to_mo_transform(
    eri: torch.Tensor,
    C_occ: torch.Tensor,
    C_vir: torch.Tensor,
) -> torch.Tensor:
    """Fused 4-index AO→MO integral transform.

    Computes

        B[i, a, P] = Σ_{μ, ν} C_occ[μ, i] · C_vir[ν, a] · eri[μ, ν, P]

    as the canonical chemistry primitive, one NKI program on Trainium.

    - Native TorchNeuron path (Trn2/Trn3): ``ao_to_mo_trn2_kernel``. Both
      stages are plain GEMMs with the contraction index on the partition
      axis (no DMA transposes), and the auxiliary index is split across both
      physical cores at LNC=2. ``"auto"`` uses it for device-resident ``eri``
      with ``nocc ≤ 128`` when its SBUF working set fits; it measured
      1.1-1.6x faster than the two aten einsums on Trn2 (#51). Otherwise the
      two-step aten contraction below runs. ``TRNTENSOR_AO_TO_MO=
      auto|trn2|aten|legacy`` overrides the choice.
    - PyTorch/XLA path (Trn1): the original kernel (``nocc ≤ 128``).

    Without NKI the fallback is the same two-step contraction.

    Composes with :func:`mp2_energy`: ``B = ao_to_mo_transform(eri, C_occ,
    C_vir)`` then ``E = mp2_energy(B, eps_occ, eps_vir)`` is the full
    DF-MP2 pipeline from AO integrals to correlation energy.

    Parameters
    ----------
    eri : (nbasis, nbasis, naux) tensor
        Density-fitted two-electron integrals in the AO basis.
    C_occ : (nbasis, nocc) tensor
        Occupied molecular orbital coefficients.
    C_vir : (nbasis, nvir) tensor
        Virtual molecular orbital coefficients.

    Returns
    -------
    (nocc, nvir, naux) tensor
        The transformed DF coefficients ``B_ia^P``.
    """
    if eri.dim() != 3:
        raise ValueError(f"eri must be 3D (nbasis, nbasis, naux); got shape {tuple(eri.shape)}")
    nbasis_mu, nbasis_nu, _naux = eri.shape
    if nbasis_mu != nbasis_nu:
        raise ValueError(
            f"eri first two dims must match (nbasis, nbasis); got ({nbasis_mu}, {nbasis_nu})"
        )
    nbasis = nbasis_mu
    if C_occ.dim() != 2 or C_occ.shape[0] != nbasis:
        raise ValueError(f"C_occ must have shape ({nbasis}, nocc); got {tuple(C_occ.shape)}")
    if C_vir.dim() != 2 or C_vir.shape[0] != nbasis:
        raise ValueError(f"C_vir must have shape ({nbasis}, nvir); got {tuple(C_vir.shape)}")
    _, nocc = C_occ.shape
    _, nvir = C_vir.shape

    route = _ao_to_mo_route(eri, nocc, nvir)
    if route != "aten":
        from .nki import dispatch as _d

        # The original kernel: NOCC ≤ 128 is the stationary M dim of its
        # step-2 nc_matmul; M-tiling not implemented. (The Trn2 kernel's
        # route already requires nocc ≤ 128.)
        if route == "legacy" and nocc > 128:
            raise NotImplementedError(
                f"ao_to_mo_transform NKI path requires nocc ≤ 128 "
                f"(got nocc={nocc}); M-tiling not yet implemented."
            )
        try:
            if route == "trn2":
                return _d._nki_ao_to_mo_trn2(eri, C_occ, C_vir)
            if _d._on_native_path():
                _d._warn_native_kernel_deprecated(
                    "original ao_to_mo_transform",
                    "the Trn2 kernel replaces it (TRNTENSOR_AO_TO_MO=trn2)",
                )
            return _d._nki_ao_to_mo_transform(eri, C_occ, C_vir)
        except Exception:
            import os

            if os.environ.get("TRNTENSOR_REQUIRE_NKI", "").lower() in ("1", "true", "yes"):
                raise
    return _cpu_ao_to_mo_transform(eri, C_occ, C_vir)


def _cpu_ao_to_mo_transform(
    eri: torch.Tensor,
    C_occ: torch.Tensor,
    C_vir: torch.Tensor,
) -> torch.Tensor:
    """Reference: ``einsum("mi,na,mnP->iaP")`` as two explicit binary steps.

    Contracting ``μ`` first (``inP``), then ``ν``, is the O(nbasis²·nocc·naux)
    order; it never builds the 4-index ``C_occ ⊗ C_vir`` outer product.
    """
    half = torch.einsum("mi,mnP->inP", C_occ, eri)
    return torch.einsum("na,inP->iaP", C_vir, half)
