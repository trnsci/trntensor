"""
NKI dispatch for tensor contractions.

The primary NKI target is fused multi-index contractions that avoid
materializing intermediates. For example, the DF-MP2 energy contraction
    E = Σ_{ijab} B_ia^P B_jb^P / Δ_{ijab}
can be tiled across (i,j) pairs with B slices loaded once to SBUF.

On the Tensor Engine, each (a,b) block is a matmul: B[i] @ B[j]^T.
Fusing the denominator division into the PSUM accumulation avoids
a separate element-wise pass over the output.
"""

from __future__ import annotations

import os

import torch

from ._kernels import HAS_NKI, TILE_K, TILE_M, TILE_N

# Re-raise kernel exceptions instead of falling back when set. Used by the
# validation suite to surface silent kernel breakage during iteration.
_REQUIRE_NKI = os.environ.get("TRNTENSOR_REQUIRE_NKI", "").lower() in ("1", "true", "yes")

_backend = "auto"


def set_backend(backend: str):
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError(
            "NKI backend requires neuronxcc. Install with: pip install 'trntensor[neuron]'"
        )
    _backend = backend


def get_backend() -> str:
    return _backend


def _use_nki() -> bool:
    # Env-var override takes precedence so benchmarks can sweep backends
    # without code changes (e.g. TRNTENSOR_FORCE_BACKEND=pytorch).
    forced = os.environ.get("TRNTENSOR_FORCE_BACKEND", "").lower()
    if forced == "nki":
        return True
    if forced == "pytorch":
        return False
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _to_xla(*tensors):
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


def nki_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """2D matmul ``A @ B`` routed through NKI when available.

    Pads M, K up to tile multiples and N up to ``TILE_N`` when N exceeds
    it (single-N-tile path skips padding). Result is sliced back to the
    original ``(M, N)``. Falls back to ``torch.matmul`` on any kernel
    failure unless ``TRNTENSOR_REQUIRE_NKI=1``.
    """
    if not (_use_nki() and HAS_NKI):
        return torch.matmul(A, B)

    from ._kernels import matmul_kernel

    M, K = A.shape
    _, N = B.shape
    M_pad = _round_up(M, TILE_M)
    K_pad = _round_up(K, TILE_K)
    N_pad = N if N <= TILE_N else _round_up(N, TILE_N)
    needs_pad = (M_pad != M) or (K_pad != K) or (N_pad != N)

    try:
        if needs_pad:
            A_p = torch.zeros(M_pad, K_pad, dtype=A.dtype, device=A.device)
            A_p[:M, :K] = A
            B_p = torch.zeros(K_pad, N_pad, dtype=B.dtype, device=B.device)
            B_p[:K, :N] = B
            (a, b), orig_device = _to_xla(A_p.contiguous(), B_p.contiguous())
        else:
            (a, b), orig_device = _to_xla(A.contiguous(), B.contiguous())
        c = matmul_kernel(a, b)
        result = c.to(orig_device)
        return result[:M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.matmul(A, B)


def nki_batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched 2D matmul ``A @ B`` over a leading batch dim.

    Routes through ``batched_matmul_kernel`` on Trainium. Pads M, K
    and N to tile multiples per slice; batch dim is passed through
    unchanged. Falls back to ``torch.bmm`` on any kernel failure
    unless ``TRNTENSOR_REQUIRE_NKI=1``.
    """
    if not (_use_nki() and HAS_NKI):
        return torch.bmm(A, B)

    from ._kernels import batched_matmul_kernel

    Bsz, M, K = A.shape
    _, _, N = B.shape
    M_pad = _round_up(M, TILE_M)
    K_pad = _round_up(K, TILE_K)
    N_pad = N if N <= TILE_N else _round_up(N, TILE_N)
    needs_pad = (M_pad != M) or (K_pad != K) or (N_pad != N)

    try:
        if needs_pad:
            A_p = torch.zeros(Bsz, M_pad, K_pad, dtype=A.dtype, device=A.device)
            A_p[:, :M, :K] = A
            B_p = torch.zeros(Bsz, K_pad, N_pad, dtype=B.dtype, device=B.device)
            B_p[:, :K, :N] = B
            (a, b), orig_device = _to_xla(A_p.contiguous(), B_p.contiguous())
        else:
            (a, b), orig_device = _to_xla(A.contiguous(), B.contiguous())
        c = batched_matmul_kernel(a, b)
        result = c.to(orig_device)
        return result[:, :M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.bmm(A, B)
