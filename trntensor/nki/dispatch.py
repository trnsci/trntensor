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


# FLOP thresholds for NKI dispatch.
#
# Profiler findings (trn1.2xlarge, 2026-04-17, scripts/run_neuron_profile.sh):
#   - XLA kernel submission latency: ~0.67 ms fixed overhead per dispatch
#   - host→device transfer: ~3.5 MB/ms (scales with tensor size)
#   - device→host transfer: ~5.5 MB/ms
#
# Without XLA residency (operands on CPU):
#   - NKI barely beats torch.matmul at 1024² (2.94 ms vs 2.94 ms)
#   - 2× speedup at 1536² (4.51 ms vs 9.44 ms)
#   Crossover: ~2 GFLOPs
#
# With XLA residency (operands pre-pinned via to_xla):
#   - Eliminates host→device transfer; crossover drops to ~900 MFLOPs
#   - 1024² wins (1.60 ms vs 2.90 ms = 1.8×)
#   - 2048² wins (4.12 ms vs 23.3 ms = 5.65×)
#   Crossover: ~1 GFLOPs
#
# Two thresholds:
#   _MIN_NKI_FLOPS        — operands on CPU (default: 2 GFLOPs)
#   _MIN_NKI_FLOPS_PINNED — operands already on XLA (default: 1 GFLOPs)
#
# Override with env vars or scripts/autotune_dispatch.py --write-cache.
def _load_min_nki_flops() -> tuple[int, int]:
    """Return (threshold_unpinned, threshold_pinned)."""
    env_unpinned = os.environ.get("TRNTENSOR_MIN_NKI_FLOPS", "")
    env_pinned = os.environ.get("TRNTENSOR_MIN_NKI_FLOPS_PINNED", "")
    cache_path = os.environ.get(
        "TRNTENSOR_AUTOTUNE_CACHE", "/var/tmp/trntensor-autotune/threshold.json"
    )
    cache_unpinned = cache_pinned = None
    try:
        import json

        with open(cache_path) as f:
            data = json.load(f)
        cache_unpinned = int(data.get("trntensor_min_nki_flops", 2_000_000_000))
        cache_pinned = int(data.get("trntensor_min_nki_flops_pinned", 1_000_000_000))
    except (FileNotFoundError, KeyError, ValueError, OSError):
        pass
    unpinned = int(env_unpinned) if env_unpinned else (cache_unpinned or 2_000_000_000)
    pinned = int(env_pinned) if env_pinned else (cache_pinned or 1_000_000_000)
    return unpinned, pinned


_MIN_NKI_FLOPS, _MIN_NKI_FLOPS_PINNED = _load_min_nki_flops()

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

    Fast path: if every tensor is already on an XLA device, call
    ``xm.mark_step()`` to force any pending lazy XLA computations to
    materialize before the next kernel invocation. Without this, the
    XLA lazy evaluator can try to fuse consecutive trntensor kernels
    into one graph, which the NKI compiler cannot lower (it produces
    shape ambiguities or trn2-only instructions on trn1).

    Otherwise this is a regular host→XLA transfer.
    """
    if all(t.device.type == "xla" for t in tensors):
        import torch_xla.core.xla_model as xm

        xm.mark_step()
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
    # Use a lower threshold when operands are already on XLA — no host→device
    # transfer cost. Crossover measured at ~900 MFLOPs vs ~2 GFLOPs unpinned.
    already_pinned = A.device.type == "xla" and B.device.type == "xla"
    flop_threshold = _MIN_NKI_FLOPS_PINNED if already_pinned else _MIN_NKI_FLOPS
    if flop_threshold > M * K * N:
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

    Pads nbasis to the nearest multiple of TILE_K (128) so the kernel's
    K-tile loops always see clean tile boundaries. Padding is zero-filled,
    so the extra rows/cols contribute nothing to the contraction result.
    Shape constraints are enforced by the caller
    (``trntensor.quantum.ao_to_mo_transform``).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    from ._kernels import ao_to_mo_transform_kernel

    nbasis = eri.shape[0]
    nbasis_pad = _round_up(nbasis, TILE_K)
    needs_nbasis_pad = nbasis_pad != nbasis

    if needs_nbasis_pad:
        naux = eri.shape[2]
        nocc = C_occ.shape[1]
        nvir = C_vir.shape[1]
        eri_p = torch.zeros(nbasis_pad, nbasis_pad, naux, dtype=eri.dtype, device=eri.device)
        eri_p[:nbasis, :nbasis, :] = eri
        C_occ_p = torch.zeros(nbasis_pad, nocc, dtype=C_occ.dtype, device=C_occ.device)
        C_occ_p[:nbasis, :] = C_occ
        C_vir_p = torch.zeros(nbasis_pad, nvir, dtype=C_vir.dtype, device=C_vir.device)
        C_vir_p[:nbasis, :] = C_vir
        eri_feed = eri_p.contiguous()
        C_occ_feed = C_occ_p.contiguous()
        C_vir_feed = C_vir_p.contiguous()
    else:
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
    already_pinned = A.device.type == "xla" and B.device.type == "xla"
    flop_threshold = _MIN_NKI_FLOPS_PINNED if already_pinned else _MIN_NKI_FLOPS
    if Bsz * M * K * N < flop_threshold:
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
