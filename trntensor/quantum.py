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

from .nki import dispatch as _dispatch
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
      growing with size (#51). Smaller problems use the aten formulation on
      the device. ``set_backend("nki")`` uses the row kernel whenever it fits
      in SBUF. ``TRNTENSOR_MP2_ENERGY=auto|rows|aten|legacy`` overrides the
      choice.
    - **Host ``B`` under native ``"auto"`` (#56)**: from
      ``nocc²·nvir²·naux`` of 1.5e8 multiply-adds (times host threads / 2),
      ``B`` and the orbital energies are uploaded and the device route above
      runs; the energy comes back as a host scalar. On Trn2 with 2 host threads that took 646 ms ->
      16 ms at nocc=32, nvir=224, naux=1024 and 15.8 s -> 135 ms at
      64/448/1536. Smaller problems stay on the host.
      ``TRNTENSOR_HOST_ROUTE=host|device`` overrides the size test.
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

    if _dispatch._offload_host(
        nocc * nocc * nvir * nvir * _naux, _dispatch._MP2_HOST_OFFLOAD_MACS, B, eps_occ, eps_vir
    ):
        e = _on_device(mp2_energy, B, eps_occ, eps_vir)
        if e is not None:
            return e.to(B.device)
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
            if route == "rows":
                _d._note_native_kernel_failure(("mp2", tuple(B.shape)))
    return _cpu_mp2_energy(B, eps_occ, eps_vir)


def _on_device(fn, *tensors: torch.Tensor):
    """``fn`` on the operands uploaded to the ``neuron`` device (#56), or None
    when that fails (e.g. the device is full), so the caller runs on the host.
    ``TRNTENSOR_REQUIRE_NKI=1`` re-raises instead."""
    import os

    try:
        return fn(*(t.to(_dispatch._NEURON_DEVICE) for t in tensors))
    except Exception:
        if os.environ.get("TRNTENSOR_REQUIRE_NKI", "").lower() in ("1", "true", "yes"):
            raise
        return None


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


def pack_tril(eri: torch.Tensor) -> torch.Tensor:
    """``(nbasis, nbasis, naux)`` μ↔ν-symmetric integrals → packed ``(npair, naux)``.

    Row ``μ(μ+1)/2 + ν`` holds ``eri[μ, ν]`` for ``ν ≤ μ``, with ``npair =
    nbasis(nbasis+1)/2``: PySCF's ``lib.pack_tril`` order (its ``_cderi``,
    ``(naux, npair)``, is the transpose) and ``trnblas.parallel.pack_tril``'s
    layout. Only the lower triangle is read; the input is assumed symmetric.
    """
    if eri.dim() != 3 or eri.shape[0] != eri.shape[1]:
        raise ValueError(f"eri must be (nbasis, nbasis, naux); got shape {tuple(eri.shape)}")
    n = eri.shape[0]
    r = torch.tril_indices(n, n)
    return eri.reshape(n * n, eri.shape[2]).index_select(0, (r[0] * n + r[1]).to(eri.device))


def unpack_tril(eri_packed: torch.Tensor, nbasis: int) -> torch.Tensor:
    """Inverse of :func:`pack_tril`: packed ``(npair, naux)`` → full ``(nbasis, nbasis, naux)``.

    One ``index_select`` on the packed tensor's device (exact).
    """
    npair, naux = eri_packed.shape
    if npair != nbasis * (nbasis + 1) // 2:
        raise ValueError(
            f"packed eri has {npair} rows; nbasis={nbasis} needs {nbasis * (nbasis + 1) // 2}"
        )
    from .nki.dispatch import _tril_rows

    idx = _tril_rows(nbasis).reshape(-1).to(eri_packed.device)
    return eri_packed.index_select(0, idx).reshape(nbasis, nbasis, naux)


def _as_packed_rows(eri: torch.Tensor, nbasis: int) -> torch.Tensor:
    """A 2-D ``eri`` as ``(npair, naux)``: accepts that, or PySCF's ``(naux, npair)``."""
    npair = nbasis * (nbasis + 1) // 2
    if eri.shape[0] == npair:
        return eri
    if eri.shape[1] == npair:
        return eri.T
    raise ValueError(
        f"eri must be 3D (nbasis, nbasis, naux), or packed 2D (npair, naux) / (naux, npair) "
        f"with npair = nbasis(nbasis+1)/2 = {npair} for nbasis={nbasis}; got {tuple(eri.shape)}"
    )


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
      with ``nocc ≤ 128`` when its SBUF working set fits and the stage-1
      work ``nocc·nbasis²·naux`` is at least 5e8 multiply-adds (about
      nbasis ≥ 256). There it measured 1.3-1.6x faster than the two aten
      einsums on Trn2 (#51). Otherwise the
      two-step aten contraction below runs. ``TRNTENSOR_AO_TO_MO=
      auto|trn2|aten|legacy`` overrides the choice.
    - PyTorch/XLA path (Trn1): the original kernel (``nocc ≤ 128``).

    Without NKI the fallback is the same two-step contraction.

    **Packed integrals (#54).** Real three-centre integrals are symmetric in
    μ↔ν, and drivers store only the lower triangle. A 2-D ``eri`` is taken
    as that packed form: ``(npair, naux)`` with row ``μ(μ+1)/2 + ν`` for
    ``ν ≤ μ`` and ``npair = nbasis(nbasis+1)/2`` (:func:`pack_tril`;
    ``trnblas.parallel.pack_tril``), or PySCF's ``_cderi`` ``(naux, npair)``
    as is (one transpose copy on the tensor's device; pass ``cderi.T`` when
    ``naux == npair``). On the native path the packed tensor feeds
    ``ao_to_mo_trn2_packed_kernel``, which gathers its rows in place: the
    full square never exists in device memory, and only half the bytes
    cross the host link. The aten and legacy routes unpack it with one
    ``index_select`` (exact).

    **Host inputs (#56).** Under native ``"auto"`` with ``eri``, ``C_occ``
    and ``C_vir`` all on the host, the inputs are uploaded (``eri`` in the
    layout given), the device route above runs and the result comes back to
    the host, once the stage-1 work ``nocc·nbasis²·naux`` reaches 2e8
    multiply-adds for a full ``eri`` or 3e7 for a packed one, times host
    threads / 2 (packed host input otherwise pays a host unpack). ``TRNTENSOR_HOST_ROUTE=host|device``
    overrides the size test.

    **Host integrals, device orbitals.** When ``eri`` is on the host and
    ``C_occ`` on a Neuron device, ``eri`` is uploaded to that device first
    (the result stays there). Send it packed: the upload runs at the host
    link's ~14 GB/s whatever the host buffer (pinned memory measured no
    faster), so halving the bytes is what moves it. Packing a full host
    tensor here would cost more host time than it saves, so a 3-D ``eri``
    is uploaded as is.

    Composes with :func:`mp2_energy`: ``B = ao_to_mo_transform(eri, C_occ,
    C_vir)`` then ``E = mp2_energy(B, eps_occ, eps_vir)`` is the full
    DF-MP2 pipeline from AO integrals to correlation energy.

    Parameters
    ----------
    eri : (nbasis, nbasis, naux), (npair, naux) or (naux, npair) tensor
        Density-fitted three-centre integrals in the AO basis, full or
        packed lower triangle.
    C_occ : (nbasis, nocc) tensor
        Occupied molecular orbital coefficients.
    C_vir : (nbasis, nvir) tensor
        Virtual molecular orbital coefficients.

    Returns
    -------
    (nocc, nvir, naux) tensor
        The transformed DF coefficients ``B_ia^P``.
    """
    if C_occ.dim() != 2 or C_vir.dim() != 2:
        raise ValueError(
            f"C_occ and C_vir must be 2D; got {tuple(C_occ.shape)} and {tuple(C_vir.shape)}"
        )
    nbasis = C_occ.shape[0]
    if eri.dim() == 2:
        eri = _as_packed_rows(eri, nbasis)
    elif eri.dim() == 3:
        nbasis_mu, nbasis_nu, _naux = eri.shape
        if nbasis_mu != nbasis_nu:
            raise ValueError(
                f"eri first two dims must match (nbasis, nbasis); got ({nbasis_mu}, {nbasis_nu})"
            )
        nbasis = nbasis_mu
    else:
        raise ValueError(
            "eri must be 3D (nbasis, nbasis, naux) or packed 2D (npair, naux); "
            f"got shape {tuple(eri.shape)}"
        )
    if C_occ.shape[0] != nbasis:
        raise ValueError(f"C_occ must have shape ({nbasis}, nocc); got {tuple(C_occ.shape)}")
    if C_vir.shape[0] != nbasis:
        raise ValueError(f"C_vir must have shape ({nbasis}, nvir); got {tuple(C_vir.shape)}")
    _, nocc = C_occ.shape
    _, nvir = C_vir.shape
    packed = eri.dim() == 2
    if _dispatch._offload_host(
        nocc * nbasis * nbasis * eri.shape[-1],
        _dispatch._AO_TO_MO_HOST_OFFLOAD_MACS_PACKED
        if packed
        else _dispatch._AO_TO_MO_HOST_OFFLOAD_MACS,
        eri,
        C_occ,
        C_vir,
    ):
        # Host in, host out through the device (#56): upload the orbitals
        # here and the integrals below (in the caller's layout), run the
        # device route, download into cached host memory.
        out = _on_device(lambda co, cv: ao_to_mo_transform(eri, co, cv), C_occ, C_vir)
        if out is not None:
            return _dispatch._to_host(out, "cpu")
    if eri.device.type == "cpu" and C_occ.device.type != "cpu":
        # Upload in the caller's layout: PySCF's (naux, npair) goes up as is
        # (a host-side transpose would cost more than the upload saves).
        if packed and not eri.is_contiguous() and eri.T.is_contiguous():
            eri = eri.T.to(C_occ.device).T
        else:
            eri = eri.contiguous().to(C_occ.device)

    route = _ao_to_mo_route(eri, nocc, nvir, nbasis)
    if packed and route != "trn2":
        # aten / legacy consume the full square.
        return ao_to_mo_transform(unpack_tril(eri, nbasis), C_occ, C_vir)
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
                if packed:
                    return _d._nki_ao_to_mo_trn2_packed(eri, C_occ, C_vir)
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
            if route == "trn2":
                _d._note_native_kernel_failure(("ao2mo", tuple(eri.shape), nocc, nvir))
    if packed:
        eri = unpack_tril(eri, nbasis)
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
