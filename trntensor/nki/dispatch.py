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

import numpy as np
import torch

from ._kernels import HAS_NKI, TILE_K, TILE_M, TILE_N

if HAS_NKI:
    import nki  # re-imported here so `nki.simulate` is reachable from dispatch

# Re-raise kernel exceptions instead of falling back when set. Used by the
# validation suite to surface silent kernel breakage during iteration.
_REQUIRE_NKI = os.environ.get("TRNTENSOR_REQUIRE_NKI", "").lower() in ("1", "true", "yes")

# Route dispatch through nki.simulate(kernel)(np_args) on CPU instead of
# XLA → NEFF → hardware. Catches Python-trace-level errors (bad kwargs,
# shape mismatches) in seconds. MLIR verifier errors remain hardware-only.
_USE_SIMULATOR = os.environ.get("TRNTENSOR_USE_SIMULATOR", "").lower() in (
    "1",
    "true",
    "yes",
)


def _use_simulator() -> bool:
    return _USE_SIMULATOR and HAS_NKI


# Total-FLOP threshold below which we skip NKI and use torch.matmul / torch.bmm
# directly. The NKI wrapper has ~1 ms of XLA dispatch overhead per call; for
# small contractions the PyTorch path finishes in less time than the overhead
# alone. Calibrated on trn1.2xlarge: matmul_2048 (8.6 GFLOPs) wins with NKI,
# matmul_1024 (1.07 GFLOPs) loses. Set the threshold at 2 GFLOPs conservatively;
# override with TRNTENSOR_MIN_NKI_FLOPS.
_MIN_NKI_FLOPS = int(os.environ.get("TRNTENSOR_MIN_NKI_FLOPS", "2000000000"))

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
    """Move tensors to the XLA device for NKI dispatch.

    Fast path: if every tensor is already on an XLA device, return them
    unchanged and report the XLA device as ``orig``. That means
    pre-pinned tensors pay no transfer cost per dispatch, and results
    stay on XLA when inputs were on XLA — the caller decides when to
    pull back with ``from_xla``.
    """
    if all(t.device.type == "xla" for t in tensors):
        return list(tensors), tensors[0].device

    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


def to_xla(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to the XLA (Trainium) device.

    Pre-pinning operands via ``to_xla`` and only pulling results back
    with ``from_xla`` when needed eliminates host↔device transfer cost
    from per-dispatch overhead. For pipelines where the same operand
    is consumed by multiple trntensor calls — DF-MP2 (same ``B`` into
    ``ao_to_mo_transform`` and ``mp2_energy``), ``(i,j)`` loops reusing
    ``B_i``, any multi-step workflow — residency is the practical fix
    for dispatch overhead.

    Raises ``RuntimeError`` on hosts where ``nki`` isn't importable;
    no-op when the tensor is already on an XLA device.
    """
    if tensor.device.type == "xla":
        return tensor
    if not HAS_NKI:
        raise RuntimeError(
            "to_xla requires the NKI runtime. Install trntensor on a Trainium "
            "instance (AWS Deep Learning AMI) or use the CPU path."
        )
    # Route through the internal _to_xla helper so residency uses exactly
    # the same path as per-kernel dispatch — keeps torch_xla initialization
    # consistent across entry points.
    xla_tensors, _ = _to_xla(tensor)
    return xla_tensors[0]


def from_xla(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor back to CPU. No-op if already on CPU."""
    if tensor.device.type == "cpu":
        return tensor
    return tensor.to("cpu")


def nki_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """2D matmul ``A @ B`` routed through NKI when available.

    Pads M, K up to tile multiples and N up to ``TILE_N`` when N exceeds
    it (single-N-tile path skips padding). Result is sliced back to the
    original ``(M, N)``. Falls back to ``torch.matmul`` on any kernel
    failure unless ``TRNTENSOR_REQUIRE_NKI=1``.
    """
    if not (_use_nki() and HAS_NKI):
        return torch.matmul(A, B)

    M, K = A.shape
    _, N = B.shape
    # Skip NKI when dispatch overhead would exceed kernel work.
    if M * K * N < _MIN_NKI_FLOPS:
        return torch.matmul(A, B)

    from ._kernels import matmul_kernel

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
            A_feed = A_p.contiguous()
            B_feed = B_p.contiguous()
        else:
            A_feed = A.contiguous()
            B_feed = B.contiguous()

        if _use_simulator():
            out_np = nki.simulate(matmul_kernel)(A_feed.cpu().numpy(), B_feed.cpu().numpy())
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            (a, b), orig_device = _to_xla(A_feed, B_feed)
            c = matmul_kernel(a, b)
            result = c.to(orig_device)
        return result[:M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.matmul(A, B)


def _nki_mp2_energy(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Dispatch ``mp2_energy_kernel`` and reduce the per-pair partial.

    The kernel returns a ``(nocc, nocc)`` tensor of pair contributions
    which we sum on the host into a scalar. Host-side reduction of a
    small matrix (``nocc`` is typically ≤ 100) is cheap compared to
    the fused contract+reduce that the kernel handles.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    from ._kernels import mp2_energy_kernel

    nocc, nvir, naux = B.shape
    # Single-tile constraints of the kernel. Caller gets a clear error
    # rather than a cryptic compile failure.
    if nvir > 128 or naux > 128:
        raise NotImplementedError(
            f"mp2_energy_kernel requires nvir ≤ 128 and naux ≤ 128 "
            f"(got nvir={nvir}, naux={naux}). K/M tiling not yet implemented."
        )

    # Reshape ε vectors to 2D at the dispatch boundary. NKI's
    # partition-dim inference is ambiguous on 1D tensor slices when
    # the input is pre-pinned on XLA (bites mp2_energy_kernel's
    # nl.load calls). 2D at entry makes the layout explicit regardless
    # of residency state. See #38 and the trnblas equivalent pattern.
    B_feed = B.contiguous()
    eo_feed = eps_occ.reshape(-1, 1).contiguous()
    ev_feed = eps_vir.reshape(-1, 1).contiguous()

    if _use_simulator():
        out_np = nki.simulate(mp2_energy_kernel)(
            B_feed.cpu().numpy(), eo_feed.cpu().numpy(), ev_feed.cpu().numpy()
        )
        return torch.from_numpy(np.asarray(out_np)).to(B.device).sum()

    (b, eo, ev), orig_device = _to_xla(B_feed, eo_feed, ev_feed)
    partial = mp2_energy_kernel(b, eo, ev)
    return partial.to(orig_device).sum()


def _nki_ao_to_mo_transform(
    eri: torch.Tensor,
    C_occ: torch.Tensor,
    C_vir: torch.Tensor,
) -> torch.Tensor:
    """Dispatch ``ao_to_mo_transform_kernel``.

    Single-tile path — shape constraints enforced by the caller
    (``trntensor.quantum.ao_to_mo_transform``).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    from ._kernels import ao_to_mo_transform_kernel

    eri_feed = eri.contiguous()
    C_occ_feed = C_occ.contiguous()
    C_vir_feed = C_vir.contiguous()

    if _use_simulator():
        out_np = nki.simulate(ao_to_mo_transform_kernel)(
            eri_feed.cpu().numpy(),
            C_occ_feed.cpu().numpy(),
            C_vir_feed.cpu().numpy(),
        )
        return torch.from_numpy(np.asarray(out_np)).to(eri.device)

    (e, co, cv), orig_device = _to_xla(eri_feed, C_occ_feed, C_vir_feed)
    B = ao_to_mo_transform_kernel(e, co, cv)
    return B.to(orig_device)


def nki_batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched 2D matmul ``A @ B`` over a leading batch dim.

    Routes through ``batched_matmul_kernel`` on Trainium. Pads M, K
    and N to tile multiples per slice; batch dim is passed through
    unchanged. Falls back to ``torch.bmm`` on any kernel failure
    unless ``TRNTENSOR_REQUIRE_NKI=1``.
    """
    if not (_use_nki() and HAS_NKI):
        return torch.bmm(A, B)

    Bsz, M, K = A.shape
    _, _, N = B.shape
    if Bsz * M * K * N < _MIN_NKI_FLOPS:
        return torch.bmm(A, B)

    from ._kernels import batched_matmul_kernel

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
            A_feed = A_p.contiguous()
            B_feed = B_p.contiguous()
        else:
            A_feed = A.contiguous()
            B_feed = B.contiguous()

        if _use_simulator():
            out_np = nki.simulate(batched_matmul_kernel)(A_feed.cpu().numpy(), B_feed.cpu().numpy())
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            (a, b), orig_device = _to_xla(A_feed, B_feed)
            c = batched_matmul_kernel(a, b)
            result = c.to(orig_device)
        return result[:, :M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.bmm(A, B)
