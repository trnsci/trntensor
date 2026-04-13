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

from .nki.dispatch import HAS_NKI, _use_nki


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

    On Trainium the entire sum is one fused NKI dispatch: contraction
    accumulates in PSUM, the spin-adapted numerator and denominator are
    built in SBUF, and the final reduction folds into a scalar
    accumulator — no intermediate ``T`` tensor is ever materialized to
    HBM. On CPU the reference path falls back to a Python loop of
    einsum calls (same as ``examples/df_mp2_einsum.py``).

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

    if _use_nki() and HAS_NKI:
        # NotImplementedError propagates — it signals an unsupported shape,
        # not a transient kernel failure. Other exceptions fall back to CPU
        # unless TRNTENSOR_REQUIRE_NKI=1.
        from .nki.dispatch import _nki_mp2_energy

        if nvir > 128 or _naux > 128:
            raise NotImplementedError(
                f"mp2_energy NKI path requires nvir ≤ 128 and naux ≤ 128 "
                f"(got nvir={nvir}, naux={_naux}). K/M tiling not yet implemented."
            )
        try:
            return _nki_mp2_energy(B, eps_occ, eps_vir)
        except Exception:
            import os

            if os.environ.get("TRNTENSOR_REQUIRE_NKI", "").lower() in ("1", "true", "yes"):
                raise
    return _cpu_mp2_energy(B, eps_occ, eps_vir)


def _cpu_mp2_energy(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation. Same loop as ``examples/df_mp2_einsum.py``."""
    nocc = B.shape[0]
    e = B.new_zeros(())
    for i in range(nocc):
        for j in range(nocc):
            T = torch.einsum("ap,bp->ab", B[i], B[j])
            denom = eps_occ[i] + eps_occ[j] - eps_vir.unsqueeze(1) - eps_vir.unsqueeze(0)
            e = e + (T * (2 * T - T.T) / denom).sum()
    return e
