"""
NKI dispatch for tensor contractions.

The primary NKI target is fused multi-index contractions that avoid
materializing intermediates. For example, the DF-MP2 energy contraction
    E = Σ_{ijab} B_ia^P B_jb^P / Δ_{ijab}
can be tiled across (i,j) pairs with B slices loaded once to SBUF.

On the Tensor Engine, each (a,b) block is a matmul: B[i] @ B[j]^T.
Fusing the denominator division into the PSUM accumulation avoids
a separate element-wise pass over the output.

Every kernel launch goes through ``_launch``, which picks the native
TorchNeuron custom op (``trntensor::<name>`` on ``device="neuron"``, see
``_native.py``) or the PyTorch/XLA path (``_to_xla``), detected lazily on
the first launch by ``_get_dispatch_path``.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import os
import sys
import threading
import warnings

import numpy as np
import torch

from ._kernels import (
    HAS_NKI,
    TILE_K,
    TILE_M,
    TILE_N,
    ao_to_mo_blocked_nub,
    ao_to_mo_blocked_sbuf_bytes,
    ao_to_mo_blocked_tp,
    ao_to_mo_trn2_nub,
    ao_to_mo_trn2_tiling,
    df_group_chunks,
    df_groups,
    df_half_tiling,
)

if HAS_NKI:
    import nki  # re-imported here so `nki.simulate` is reachable from dispatch

# Re-raise kernel exceptions instead of falling back when set. Used by the
# validation suite to surface silent kernel breakage during iteration.
_REQUIRE_NKI = os.environ.get("TRNTENSOR_REQUIRE_NKI", "").lower() in ("1", "true", "yes")
# Keep float64 / complex128 that is already on the ``neuron`` device in device
# aten (fp32 precision) under ``"auto"`` instead of computing it on the host.
# See ``_fp64_host_guard``.
_FP64_ON_DEVICE = os.environ.get("TRNTENSOR_FP64_ON_DEVICE", "").lower() in ("1", "true", "yes")

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


# Stochastic rounding on NKI 0.7+ is a process-wide Neuron Runtime setting,
# not a per-instruction one (#41). NKI 0.7 removed ``nisa.activation``'s
# ``round_mode`` kwarg, so ``use_sr=True`` cannot select SR by itself. The
# runtime reads this variable once at init: when it is ``"1"`` every
# FP32->BF16/FP16 downcast in the process rounds stochastically (including
# ``use_sr=False`` / ``precision="fast"``); otherwise every downcast is
# round-to-nearest-even. Measured on Trn2, SDK 2.33 / NKI 0.7.0. See
# aws-neuron-sdk about-neuron/arch/neuron-features/rounding-modes.rst.
SR_ENV_VAR = "NEURON_RT_STOCHASTIC_ROUNDING_EN"

_sr_warning_emitted = False


def sr_runtime_enabled() -> bool:
    """Return True if the Neuron Runtime is configured for stochastic rounding.

    Checks ``NEURON_RT_STOCHASTIC_ROUNDING_EN == "1"`` in the current
    environment. The runtime only reads the variable at initialization, so
    it must be set before the process first touches a Neuron device; setting
    it afterwards makes this return True without changing rounding.
    """
    return os.environ.get(SR_ENV_VAR) == "1"


def _warn_if_sr_unavailable() -> None:
    """Emit a one-time ``RuntimeWarning`` when SR is requested on the native
    NKI path but the runtime is not configured for it. Computation proceeds
    with round-to-nearest-even."""
    global _sr_warning_emitted
    if _sr_warning_emitted or sr_runtime_enabled():
        return
    _sr_warning_emitted = True
    warnings.warn(
        'trntensor: precision="sr" requested on the NKI path, but stochastic '
        f"rounding requires {SR_ENV_VAR}=1 in the environment at process start "
        "(it is a Neuron Runtime setting read once at init, not a per-kernel "
        "option on NKI 0.7+). Without it, results are round-to-nearest. Note "
        "that with it set, ALL downcasts in the process round stochastically, "
        'including precision="fast". This warning is shown once per process.',
        RuntimeWarning,
        stacklevel=3,
    )


# FLOP thresholds for NKI dispatch.
#
# XLA path (PyTorch/XLA lineage, Trn1). Profiler findings (trn1.2xlarge,
# 2026-04-17, scripts/run_neuron_profile.sh):
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
# XLA thresholds (unchanged):
#   _MIN_NKI_FLOPS        — operands on CPU (default: 2 GFLOPs)
#   _MIN_NKI_FLOPS_PINNED — operands already on XLA (default: 1 GFLOPs)
#
# Native path (TorchNeuron, Trn2/Trn3). Measured on trn2.3xlarge, SDK 2.33 /
# NKI 0.7.0, one NeuronCore, warm latency, 2026-10-30
# (scripts/bench_native_einsum.py, scripts/autotune_dispatch.py; #41):
#   - aten ``torch.matmul`` / ``torch.bmm`` on device-resident ``neuron``
#     tensors beat the trntensor NKI matmul / batched-matmul kernels at every
#     measured size, fp32 and bf16, 128³ to 4096³ (e.g. 4096³ bf16: 1.38 ms
#     aten vs 6.82 ms NKI; 1024³ fp32: 0.20 ms vs 0.55 ms). The eager aten
#     launch floor is ~0.13 ms, the NKI custom-op floor ~0.24 ms.
#   - For host operands, offloading to aten on the device (upload, matmul,
#     download) beats host torch from ~0.5 G multiply-adds (M·K·N) and beats
#     NKI-with-transfer at every size.
# So on the native path the two thresholds mean:
#   _MIN_NKI_FLOPS_NATIVE        — host operands: M·K·N at which the work leaves
#                                  the host for the device (default 500 M).
#   _MIN_NKI_FLOPS_NATIVE_PINNED — M·K·N at which the NKI kernel replaces aten
#                                  on the device (default: never, NKI_NEVER).
# Below the pinned threshold, device work runs as aten on ``neuron``. See
# _native_route. Set either to a number (env / cache) to re-enable NKI.
#
# Thresholds compare the multiply-add count M·K·N (B·M·K·N for bmm).
# _nki_flop_threshold(pinned) returns the threshold for the active path.
# Override with env vars or scripts/autotune_dispatch.py --write-cache; the
# value "never" (or "inf") disables a threshold.
NKI_NEVER = 2**63 - 1
_DEFAULT_MIN_NKI_FLOPS = 2_000_000_000
_DEFAULT_MIN_NKI_FLOPS_PINNED = 1_000_000_000
_DEFAULT_MIN_NKI_FLOPS_NATIVE = 500_000_000
_DEFAULT_MIN_NKI_FLOPS_NATIVE_PINNED = NKI_NEVER


def _load_threshold_cache() -> dict:
    cache_path = os.environ.get(
        "TRNTENSOR_AUTOTUNE_CACHE", "/var/tmp/trntensor-autotune/threshold.json"
    )
    try:
        import json

        with open(cache_path) as f:
            data = json.load(f)
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_threshold(value) -> int:
    """int, or ``NKI_NEVER`` for "never" / "inf" (NKI disabled)."""
    if isinstance(value, str) and value.strip().lower() in ("never", "inf", "infinity"):
        return NKI_NEVER
    return int(value)


def _threshold(env_var: str, cache: dict, cache_key: str, default: int) -> int:
    env = os.environ.get(env_var, "")
    if env:
        return _parse_threshold(env)
    try:
        cached = cache.get(cache_key)
        return _parse_threshold(cached) if cached else default
    except (TypeError, ValueError):
        return default


def _load_min_nki_flops() -> tuple[int, int]:
    """Return the XLA-path (threshold_unpinned, threshold_pinned)."""
    cache = _load_threshold_cache()
    unpinned = _threshold(
        "TRNTENSOR_MIN_NKI_FLOPS", cache, "trntensor_min_nki_flops", _DEFAULT_MIN_NKI_FLOPS
    )
    pinned = _threshold(
        "TRNTENSOR_MIN_NKI_FLOPS_PINNED",
        cache,
        "trntensor_min_nki_flops_pinned",
        _DEFAULT_MIN_NKI_FLOPS_PINNED,
    )
    return unpinned, pinned


def _load_min_nki_flops_native() -> tuple[int, int]:
    """Return the native-path (threshold_unpinned, threshold_pinned)."""
    cache = _load_threshold_cache()
    unpinned = _threshold(
        "TRNTENSOR_MIN_NKI_FLOPS_NATIVE",
        cache,
        "trntensor_min_nki_flops_native",
        _DEFAULT_MIN_NKI_FLOPS_NATIVE,
    )
    pinned = _threshold(
        "TRNTENSOR_MIN_NKI_FLOPS_NATIVE_PINNED",
        cache,
        "trntensor_min_nki_flops_native_pinned",
        _DEFAULT_MIN_NKI_FLOPS_NATIVE_PINNED,
    )
    return unpinned, pinned


_MIN_NKI_FLOPS, _MIN_NKI_FLOPS_PINNED = _load_min_nki_flops()
_MIN_NKI_FLOPS_NATIVE, _MIN_NKI_FLOPS_NATIVE_PINNED = _load_min_nki_flops_native()


def _on_native_path() -> bool:
    return not _use_simulator() and _get_dispatch_path() == "native"


def _nki_flop_threshold(pinned: bool) -> int:
    """FLOP threshold (M·K·N) below which a contraction stays on torch.

    Path-aware: the native TorchNeuron path uses the native thresholds; the
    XLA path, the simulator, and hosts without a device path use the XLA
    values. ``pinned`` means the operands already live on the path's device.
    """
    if _on_native_path():
        return _MIN_NKI_FLOPS_NATIVE_PINNED if pinned else _MIN_NKI_FLOPS_NATIVE
    return _MIN_NKI_FLOPS_PINNED if pinned else _MIN_NKI_FLOPS


def _min_nki_flops(*operands: torch.Tensor) -> int:
    """:func:`_nki_flop_threshold` for these operands' residency."""
    device_type = _NEURON_DEVICE if _on_native_path() else "xla"
    return _nki_flop_threshold(pinned=all(t.device.type == device_type for t in operands))


def _route(flops: int, pinned: bool) -> str:
    """Executor for an ``M·K·N``-sized matmul/bmm: "nki", "aten" or "host".

    "host" is torch on the operands' own device (CPU operands stay on CPU,
    device operands run aten on the device). "aten" (native path only) moves
    host operands to the ``neuron`` device, runs aten there and brings the
    result back. "nki" launches the trntensor kernel.

    XLA path / simulator: "nki" at or above the residency-specific threshold,
    else "host" (unchanged). Native path: device-resident operands use "nki"
    at or above the pinned threshold, else "host" (aten on device); host
    operands stay on the host below the unpinned (offload) threshold and
    otherwise go to the device, as "nki" at or above the pinned threshold and
    as "aten" below it. ``TRNTENSOR_HOST_ROUTE=host|device`` replaces the
    offload threshold for host operands (#56).
    """
    if not _on_native_path():
        return "nki" if flops >= _nki_flop_threshold(pinned) else "host"
    nki_ok = flops >= _MIN_NKI_FLOPS_NATIVE_PINNED
    if pinned:
        return "nki" if nki_ok else "host"
    choice = _host_route_choice()
    if choice == "host" or (choice == "auto" and flops < _MIN_NKI_FLOPS_NATIVE):
        return "host"
    return "nki" if nki_ok else "aten"


def _route_for(flops: int, *operands: torch.Tensor) -> str:
    """:func:`_route` for these operands' residency.

    On the native path, host operands of a dtype the device does not compute
    natively (float64, complex) stay on the host: the ``neuron`` device's
    engines top out at fp32 (#56)."""
    native = _on_native_path()
    device_type = _NEURON_DEVICE if native else "xla"
    pinned = all(t.device.type == device_type for t in operands)
    route = _route(flops, pinned=pinned)
    if native and route == "aten" and any(t.dtype not in _OFFLOAD_DTYPES for t in operands):
        return "host"
    return route


def _aten_on_device(fn, *tensors: torch.Tensor) -> torch.Tensor:
    """Run ``fn`` (an aten op) on the ``neuron`` device; result on the caller's device."""
    orig = tensors[0].device
    return _to_host(fn(*(t.to(_NEURON_DEVICE) for t in tensors)), orig)


# ── Host-input routing (#56) ────────────────────────────────────────────────
#
# Native path, ``"auto"`` backend, every operand on the host: once the work
# reaches the entry point's threshold, the operands are uploaded, the call
# runs on the ``neuron`` device by the device-resident policy, and the result
# comes back through the caching host allocator (`_to_host`). Measured on Trn2
# (SDK 2.33, one logical core at LNC=2, fp32, warm medians, host in -> host
# out, tables on #56), with 2 host threads on two physical cores:
#
# - mp2_energy, work nocc²·nvir²·naux: the device route costs ~3 ms at small
#   sizes (upload, launch floor, scalar readback). Host aten takes 3.8 ms at
#   1.05e8 and 5.1 ms at 1.5e8. Larger problems gain 40-120x: 646 ms -> 16 ms
#   at nocc=32, nvir=224, naux=1024, and 15.8 s -> 135 ms at 64/448/1536.
# - ao_to_mo_transform, stage-1 work nocc·nbasis²·naux:
#   - full eri: host 6.2 vs device 8.9 ms at 1.3e8, 9.4 vs 7.1 ms at 2.5e8,
#     then 1.2-4.2x. The upload of the full square sets the device time.
#   - packed eri: on the host it is unpacked first, which makes the host
#     route 1.3-12x slower than uploading the packed rows (half the bytes).
#     Host wins at 6e6, the device from 3e7.
# - einsum with a "torch" or "path" plan and 2+ operands: the host runs bmm
#   either way, so the crossover is matmul's, 5e8. Patterns with little
#   reuse (under _EINSUM_HOST_MIN_INTENSITY multiply-adds per byte moved:
#   elementwise products at 0.08, outer products at 0.25) stay on the host.
#   They measured 1.4-2.9x faster uploaded, but only because 2 host threads
#   stream memory slower than the link. Host bandwidth grows with threads and
#   the link does not, and these patterns fill device memory.
#
# The host side is compute-bound and scales with threads, so like trnblas#65
# the thresholds scale by torch.get_num_threads() / _HOST_MODEL_THREADS. The
# matmul/bmm offload threshold (_MIN_NKI_FLOPS_NATIVE, #41; re-measured here
# at the same crossover) keeps its own env/cache override and is not scaled.
#
# TRNTENSOR_HOST_ROUTE=host|device|auto overrides the size tests for host
# operands, matmul/bmm included. Explicit backends keep their route:
# set_backend("pytorch") never leaves the host, and set_backend("nki") runs
# the kernels, which upload host operands themselves. Only float32 /
# bfloat16 / float16 tensors are uploaded; float64 and complex stay on the
# host, since the device computes in at most fp32.
_HOST_ROUTE_CHOICES = ("auto", "host", "device")
_HOST_MODEL_THREADS = 2
_MP2_HOST_OFFLOAD_MACS = 1.5e8
_AO_TO_MO_HOST_OFFLOAD_MACS = 2.0e8
_AO_TO_MO_HOST_OFFLOAD_MACS_PACKED = 3.0e7
_EINSUM_HOST_OFFLOAD_MACS = 5.0e8
_EINSUM_HOST_MIN_INTENSITY = 0.5
_OFFLOAD_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _host_route_choice() -> str:
    choice = os.environ.get("TRNTENSOR_HOST_ROUTE", "auto").strip().lower() or "auto"
    if choice not in _HOST_ROUTE_CHOICES:
        raise ValueError(
            f"TRNTENSOR_HOST_ROUTE must be one of {_HOST_ROUTE_CHOICES}; got {choice!r}"
        )
    return choice


def _host_thread_scale() -> float:
    return max(1, torch.get_num_threads()) / _HOST_MODEL_THREADS


def _offload_host(
    work: float, threshold: float, *tensors: torch.Tensor, floor: float = 0.0
) -> bool:
    """True when host operands doing ``work`` multiply-adds should be uploaded
    and run on the ``neuron`` device: native path, ``"auto"`` backend (not
    forced NKI), every operand a CPU tensor of an uploadable dtype, and
    ``work`` at least ``threshold`` (scaled by the host's thread count) and
    ``floor`` (not scaled). ``TRNTENSOR_HOST_ROUTE=host|device`` replaces the
    size test."""
    if not (_use_nki() and HAS_NKI) or not _on_native_path() or _nki_forced():
        return False
    if _NEURON_DEVICE == "cpu":  # CPU standing in for the device (tests)
        return False
    if any(t.device.type != "cpu" or t.dtype not in _OFFLOAD_DTYPES for t in tensors):
        return False
    choice = _host_route_choice()
    if choice != "auto":
        return choice == "device"
    return work >= max(threshold * _host_thread_scale(), floor)


# ── Host results through the caching host allocator ─────────────────────────
#
# Same fix as trnblas#61/#64. On Trn2 (SDK 2.33, native TorchNeuron) a
# device→host copy into fresh pageable memory (`.to("cpu")`, `torch.empty` +
# `copy_`) runs at ~4 GB/s, because the host page-faults every page it
# touches. A copy into memory that is already faulted in runs at the link
# rate, 13-14 GB/s. `torch.empty(..., pin_memory=True)` comes from the
# accelerator's caching host allocator: each call still returns a fresh tensor
# (a block is recycled only after the tensor that owned it is freed, so results
# never alias), and recycled blocks come back already faulted in. Uploads from
# pageable memory already run at the link rate and are unchanged.
#
# Costs: blocks are rounded up to a power of two, and freed blocks stay cached
# until `torch.accelerator.empty_host_cache()`. TRNTENSOR_PINNED_OUTPUT=0 turns
# this off (plain `.to("cpu")`). Results under `_PINNED_OUTPUT_MIN_BYTES`
# always take `.to(device)`. The XLA path and the simulator are unchanged.
_PINNED_OUTPUT_MIN_BYTES = 1 << 20


def _pinned_output_enabled() -> bool:
    value = os.environ.get("TRNTENSOR_PINNED_OUTPUT", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _use_host_cache(device, nbytes: int) -> bool:
    """True when a CPU result of `nbytes` should come from the caching host
    allocator: native path, CPU destination, large enough, not disabled."""
    return (
        torch.device(device).type == "cpu"
        and nbytes >= _PINNED_OUTPUT_MIN_BYTES
        and _pinned_output_enabled()
        and not _use_simulator()
        and _get_dispatch_path() == "native"
    )


def _host_empty(shape, dtype: torch.dtype, device) -> torch.Tensor:
    """`torch.empty(shape, dtype=dtype, device=device)`; a large CPU tensor on the
    native path comes from the caching host allocator (fresh, already faulted in)."""
    shape = tuple(shape)
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if _use_host_cache(device, nbytes):
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    return torch.empty(shape, dtype=dtype, device=device)


def _to_host(t: torch.Tensor, device) -> torch.Tensor:
    """`t.to(device)`. A large device → CPU copy lands in a fresh tensor from
    the caching host allocator, about 3.5x faster than `.to("cpu")` (trnblas#61)."""
    if t.device.type == "cpu" or not _use_host_cache(device, t.numel() * t.element_size()):
        return t.to(device)
    return _host_empty(t.shape, t.dtype, device).copy_(t)


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


def _nki_forced() -> bool:
    """True when the caller explicitly asked for NKI (``set_backend("nki")`` or
    ``TRNTENSOR_FORCE_BACKEND=nki``), as opposed to ``"auto"``."""
    forced = os.environ.get("TRNTENSOR_FORCE_BACKEND", "").lower()
    if forced:
        return forced == "nki"
    return _backend == "nki"


# Native-path kernel retirement (#51). On Trn2 these trntensor kernels measured
# slower than aten at every size, including a rewrite with the program_id
# split and Tensor Engine transposes, so native ``"auto"`` never runs them.
# ``set_backend("nki")`` still runs them, with a one-time FutureWarning; a
# later release routes them to aten on the native path. The XLA path (Trn1)
# keeps them unchanged and does not warn.
_native_kernel_deprecations_emitted: set[str] = set()

NATIVE_KERNEL_DEPRECATION_MESSAGE = (
    "trntensor: the NKI {name} kernel is deprecated on the native TorchNeuron "
    "path (Trn2/Trn3) and will be removed from it in a future release; {why} "
    "(trnsci/trntensor#51). The PyTorch/XLA path (Trn1) is unaffected. This "
    "warning is shown once per kernel per process."
)
_ATEN_IS_FASTER = (
    'aten on the neuron device is faster at every measured size, and set_backend("auto") '
    "already uses it"
)


def _warn_native_kernel_deprecated(name: str, why: str = _ATEN_IS_FASTER) -> None:
    if name in _native_kernel_deprecations_emitted:
        return
    _native_kernel_deprecations_emitted.add(name)
    warnings.warn(
        NATIVE_KERNEL_DEPRECATION_MESSAGE.format(name=name, why=why),
        FutureWarning,
        stacklevel=3,
    )


def _use_fused_kernel() -> bool:
    """Whether the fused ``ao_to_mo_transform`` kernel should run under the
    current settings (``mp2_energy`` routes through :func:`_mp2_energy_route`).

    On the native TorchNeuron path, ``"auto"`` uses the aten formulation
    (#41, #51); the kernel still runs when NKI is forced. Other paths keep
    the kernel under ``"auto"``.
    """
    if not (_use_nki() and HAS_NKI):
        return False
    if _on_native_path():
        return _nki_forced()
    return True


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


# ── Device dispatch path: native TorchNeuron vs XLA ──────────────────────────
#
# Two Neuron PyTorch lineages coexist:
#   native  torch-neuronx 2.14+ (TorchNeuron, Trn2/Trn3): Trainium is a
#           PyTorch `neuron` device; kernels are custom ops registered via
#           `torch_neuronx.nki_op` (see `_native.py`).
#   xla     torch-neuronx 2.9 lineage (PyTorch/XLA; Trn1 and Inf2 stay
#           here): kernels dispatch on the torch_xla device via `_to_xla()`.
# Anything else is "cpu": no device path, so a kernel launch outside the
# simulator goes through `_to_xla` and fails there exactly as it did before
# the native path existed (the host wrappers then fall back to torch unless
# TRNTENSOR_REQUIRE_NKI=1). Detection runs on the first launch, not at
# import, and the native probe never imports torch_xla.

_NEURON_DEVICE = "neuron"

# TorchNeuron's aten matmul with a transposed-view left operand returns wrong
# values (or a DMA abort) once the right operand reaches 2**32 bytes, fp32 and
# bf16, LNC=1 and 2 (torch-neuronx 2.14.3; aws-neuron-sdk#1448, #97). The
# guard starts at 2**32 and ignores the contraction dim. (This comment used to
# say "exactly 2**32 is correct". That measurement was correct by chance: the
# operand sat directly above an identical copy. With other data below it,
# exactly 2**32 is wrong too (#123, aws-neuron-sdk#1457).)
_ATEN_TRANSPOSED_LHS_MAX_BYTES = 1 << 32

# The same bug hits a *contiguous* left operand (#123, trnblas#112,
# aws-neuron-sdk#1457, trnsci/trnsci#140). An operand's memory rows are its DMA
# partition step, and once min(rows, 128) * row_bytes >= 2**32 (for B (K, N)
# with K >= 128, a row of 32 MiB or more: fp32 N >= 8,388,608), partitions
# 64-127 are read from 4 GiB below the right address. Measured on Trn2 (SDK
# 2.33, LNC=2, operand above a decoy, 2026-11-07): (96, 128) @ (128, 8,389,632)
# has rel err 0.79 with no error, and so do bmm and a long-row left operand.
# A device copy of a column slice of such a B is itself wrong, and so is a
# device `torch.cat` along dim 1 into a result with that layout. Contiguous
# blocks of whole rows spanning about 2 GiB are correct (also at offsets up
# to 6 GB), a `cat` of whole-row blocks along dim 0 is correct, and host round
# trips are bit-exact. So `_neuron_matmul` splits K (partial sums) or M / N
# (concatenated results) into row blocks of the long operand, and builds a
# result with that layout from blocks of its rows. Products below the rule are
# the same aten call as before.
_NEURON_SPAN_BYTES = 1 << 32
_NEURON_PARTITIONS = 128
# Rows per block: a span of at most 2 GiB + 1 MiB, so 64 rows of up to
# 32 MiB + 16 KiB (measured correct: K = 64 at 32 MiB + 4 KiB on #140, and
# 63-row blocks here) and fewer rows for longer ones. All of them sit in
# partitions 0-63, which the upstream analysis says are read correctly.
_NEURON_BLOCK_BYTES = (1 << 31) + (1 << 20)


def _even_step(total: int, step: int) -> int:
    """A block length of at most `step` that cuts `total` into near-equal blocks.

    Fewer distinct block shapes means fewer NEFF compiles: 128 rows in blocks of
    at most 64 is 64 + 64, not 63 + 63 + 2."""
    return -(-total // -(-total // step))


def _span_bytes(t: torch.Tensor) -> int:
    """Bytes from the first to the last element of `t`'s view, inclusive."""
    if t.numel() == 0:
        return 0
    return (
        1 + sum((n - 1) * abs(s) for n, s in zip(t.shape, t.stride(), strict=True))
    ) * t.element_size()


def _row_layout(t: torch.Tensor) -> tuple[int, int, int]:
    """(row dim, rows, row bytes) of the memory rows of a 2-D `t`.

    A row-major matrix has its memory rows along dim 0, and a transposed view
    (`X.T` of a row-major X) along dim 1. Row bytes is the row stride, or the
    row length if that is larger, so a slice of a wider matrix counts its
    parent's rows.
    """
    item = t.element_size()
    if t.stride(1) == 1 or (t.stride(0) != 1 and t.shape[1] == 1):
        return 0, t.shape[0], max(t.shape[1], abs(t.stride(0))) * item
    return 1, t.shape[1], max(t.shape[0], abs(t.stride(1))) * item


def _span_block(t: torch.Tensor) -> int:
    """0 if a 2-D `t` is a safe device matmul operand, else its rows per block.

    Unsafe means min(rows, 128) * row bytes >= 2**32 (see above). A block of
    that many memory rows spans at most 2 GiB + 1 MiB (and holds at least one row).
    """
    if t.dim() != 2 or _span_bytes(t) < _NEURON_SPAN_BYTES:
        return 0
    _, rows, row_bytes = _row_layout(t)
    if min(rows, _NEURON_PARTITIONS) * row_bytes < _NEURON_SPAN_BYTES:
        return 0
    return max(1, _NEURON_BLOCK_BYTES // row_bytes)


def _out_block(M: int, N: int, itemsize: int) -> int:
    """0 if a fresh row-major (M, N) result is safe, else its rows per block."""
    row_bytes = N * itemsize
    if min(M, _NEURON_PARTITIONS) * row_bytes < _NEURON_SPAN_BYTES:
        return 0
    return max(1, _NEURON_BLOCK_BYTES // row_bytes)


def _neuron_cat(parts: list, dim: int) -> torch.Tensor:
    """`torch.cat(parts, dim)` of 2-D device blocks, safe when the result has the bad layout.

    Measured on Trn2: a device `cat` along dim 0 into a (128, 8,389,632) fp32
    result is correct (whole-row blocks), along dim 1 it is wrong (rel err
    0.71). So a dim-1 result with that layout is assembled on the host and
    uploaded (an upload lands bit-exact). `_neuron_mm2d` blocks the output rows
    first, so its dim-1 concatenations never need that.
    """
    if len(parts) == 1:
        return parts[0]
    M = sum(p.shape[0] for p in parts) if dim == 0 else parts[0].shape[0]
    N = sum(p.shape[1] for p in parts) if dim == 1 else parts[0].shape[1]
    if dim != 0 and _on_neuron_device(parts[0]) and _out_block(M, N, parts[0].element_size()):
        return torch.cat([p.cpu() for p in parts], dim).to(parts[0].device)
    return torch.cat(parts, dim)


def _neuron_mm_unsafe(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True when device `torch.matmul(a, b)` needs `_neuron_matmul`'s blocking.

    Cheap and conservative: False whenever neither operand nor the result
    spans 4 GiB, so ordinary products keep their exact aten call.
    """
    if a.dim() < 2 or b.dim() < 2 or not (_on_neuron_device(a) or _on_neuron_device(b)):
        return False
    try:
        batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    except RuntimeError:
        return False  # torch.matmul raises its own error
    out_bytes = math.prod(batch) * a.shape[-2] * b.shape[-1] * a.element_size()
    if max(_span_bytes(a), _span_bytes(b), out_bytes) < _NEURON_SPAN_BYTES:
        return False
    M, N = a.shape[-2], b.shape[-1]
    rows = math.prod(batch) * M
    return bool(_batched_block(a) or _batched_block(b) or _out_block(rows, N, a.element_size()))


def _batched_block(t: torch.Tensor) -> int:
    """`_span_block` of a (batched) operand, its matrices' rows counted together.

    Conservative for batches: the memory rows of all matrices count toward the
    128 partitions, as for an elementwise op on the whole tensor.
    """
    if t.dim() == 2:
        return _span_block(t)
    if _span_bytes(t) < _NEURON_SPAN_BYTES:
        return 0
    _, rows, row_bytes = _row_layout(t[(0,) * (t.dim() - 2)])
    rows *= math.prod(t.shape[:-2])
    if min(rows, _NEURON_PARTITIONS) * row_bytes < _NEURON_SPAN_BYTES:
        return 0
    return max(1, _NEURON_BLOCK_BYTES // row_bytes)


def _neuron_mm2d(a: torch.Tensor, b: torch.Tensor, mm=None) -> torch.Tensor:
    """2-D `a @ b` on the device, cut into row blocks wherever trnblas#112 applies.

    `mm` computes each safe block (default: aten)."""
    M, K = a.shape
    N = b.shape[1]
    ob = _out_block(M, N, a.element_size())
    if ob and ob < M:  # the result itself: blocks of output rows
        ob = _even_step(M, ob)
        return _neuron_cat([_neuron_mm2d(a[i : i + ob], b, mm) for i in range(0, M, ob)], 0)
    ab = _span_block(a)
    if ab:
        if _row_layout(a)[0] == 0 and ab < M:  # long rows of A: split M
            ab = _even_step(M, ab)
            return _neuron_cat([_neuron_mm2d(a[i : i + ab], b, mm) for i in range(0, M, ab)], 0)
        if _row_layout(a)[0] == 1 and ab < K:  # A = X.T, long rows of X: split K
            return _neuron_mm_ksum(a, b, ab, mm)
    bb = _span_block(b)
    if bb:
        if _row_layout(b)[0] == 0 and bb < K:  # long rows of B: split K
            return _neuron_mm_ksum(a, b, bb, mm)
        if _row_layout(b)[0] == 1 and bb < N:  # B = Y.T, long rows of Y: split N
            bb = _even_step(N, bb)
            return _neuron_cat([_neuron_mm2d(a, b[:, j : j + bb], mm) for j in range(0, N, bb)], 1)
    if mm is not None and not _neuron_mm_unsafe(a, b):
        return mm(a, b)
    return torch.matmul(a, b)


def _neuron_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`torch.bmm(a, b)`, blocked like `_neuron_matmul` when #123 applies."""
    if a.dim() == 3 and b.dim() == 3 and _neuron_mm_unsafe(a, b):
        return _neuron_matmul(a, b)
    return torch.bmm(a, b)


def _neuron_mm_ksum(a: torch.Tensor, b: torch.Tensor, step: int, mm=None) -> torch.Tensor:
    """Sum over K blocks of `step` of a[:, k] @ b[k], in order."""
    out = None
    step = _even_step(a.shape[1], step)
    for k in range(0, a.shape[1], step):
        ak, bk = a[:, k : k + step], b[k : k + step]
        # The other operand's slice is small and strided: give it a dense copy.
        if not _span_block(ak):
            ak = ak.contiguous()
        if not _span_block(bk):
            bk = bk.contiguous()
        part = _neuron_mm2d(ak, bk, mm)
        out = part if out is None else out + part
    return out


def _neuron_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`torch.matmul(a, b)`, correct on the `neuron` device at any size (trnblas#112).

    The same call as before unless an operand or the
    result has a partition span of 4 GiB or more; then the product is cut into
    row blocks of the long operand (see `_NEURON_SPAN_BYTES`). Splitting K
    changes the fp32 summation order of those products only. Batched operands
    then run matrix by matrix. Host tensors are never affected.
    """
    if not _neuron_mm_unsafe(a, b):
        return torch.matmul(a, b)
    if a.dim() == 2 and b.dim() == 2:
        return _neuron_mm2d(a, b)
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    ae = a.expand(*batch, *a.shape[-2:])
    be = b.expand(*batch, *b.shape[-2:])
    M, N = a.shape[-2], b.shape[-1]
    parts = [_neuron_mm2d(ae[idx], be[idx]) for idx in np.ndindex(*batch)]
    return _neuron_cat(parts, 0).reshape(*batch, M, N)


_dispatch_path: str | None = None
_native_ops: dict | None = None

# Dormant XLA deprecation hook (trnsci/trnsci#35). Off unless the suite-wide
# TRNSCI_XLA_DEPRECATION=1 is set; the suite turns it on at TorchNeuron GA.
XLA_DEPRECATION_ENV_VAR = "TRNSCI_XLA_DEPRECATION"
XLA_DEPRECATION_MESSAGE = (
    "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
    "Trn1/Inf2 remain supported in maintenance mode."
)
_xla_deprecation_emitted = False
_api_deprecations_emitted: set[str] = set()


def _xla_deprecation_enabled() -> bool:
    return os.environ.get(XLA_DEPRECATION_ENV_VAR, "") == "1"


def _warn_xla_path_deprecated() -> None:
    global _xla_deprecation_emitted
    if _xla_deprecation_emitted or not _xla_deprecation_enabled():
        return
    _xla_deprecation_emitted = True
    warnings.warn(XLA_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=4)


def _native_available() -> bool:
    try:
        import torch_neuronx
    except Exception:
        return False
    # The XLA-lineage torch_neuronx has no `nki_op`.
    if not hasattr(torch_neuronx, "nki_op"):
        return False
    try:
        torch.device(_NEURON_DEVICE)  # torch_neuronx renames PrivateUse1 to "neuron"
        return torch_neuronx.device_count() > 0
    except Exception:
        return False


def _detect_dispatch_path() -> str:
    # TRNSCI_NKI_DISPATCH=auto|native|xla is shared by every trnsci library, so
    # one variable pins the whole suite on hosts that have both lineages.
    forced = os.environ.get("TRNSCI_NKI_DISPATCH", "auto").strip().lower()
    if forced == "native":
        if not _native_available():
            raise RuntimeError(
                "TRNSCI_NKI_DISPATCH=native, but no TorchNeuron backend with a "
                "'neuron' device is available"
            )
        return "native"
    if forced == "xla":
        if importlib.util.find_spec("torch_xla") is None:
            raise RuntimeError("TRNSCI_NKI_DISPATCH=xla, but torch_xla is not installed")
        return "xla"
    if forced != "auto":
        raise ValueError(f"TRNSCI_NKI_DISPATCH must be auto, native, or xla; got {forced!r}")
    if _native_available():
        return "native"
    if importlib.util.find_spec("torch_xla") is not None:
        return "xla"
    return "cpu"


def _get_dispatch_path() -> str:
    """Return "native", "xla", or "cpu" (cached after the first call).

    Selecting the XLA path emits the dormant one-time ``DeprecationWarning``
    when ``TRNSCI_XLA_DEPRECATION=1`` is set.
    """
    global _dispatch_path
    if _dispatch_path is None:
        _dispatch_path = _detect_dispatch_path()
    if _dispatch_path == "xla":
        _warn_xla_path_deprecated()
    return _dispatch_path


# ── Kernel operand dtypes: one guard at the kernel boundary ──────────────────
#
# Trainium has no fp64 datapath and the trntensor kernels are written for
# fp32 (bf16 / fp16 where listed). Through 0.23.0 float64 (and integer, bool,
# complex) tensors could reach them unchanged (float64 chemistry: through
# 0.22.0, then cast by #88) and come back inf, garbage or
# in the wrong dtype (#87, the same class as trnblas#68, trnfft#109/#131,
# trnsolver#83, trnsparse#78/#79). Two layers close it:
#
# - The host wrappers decide per dtype before any launch: the matmul family
#   runs torch for a dtype the kernel does not take
#   (:func:`_matmul_kernel_dtypes_ok`); the chemistry routes compute float16
#   (and bf16 on the df_* kernels) in fp32 and cast the result back, and take
#   aten for integer, bool and complex operands (:func:`_chem_kernel_operands`).
#   float64 follows the suite fp64 policy (:func:`_fp64_to_kernel`). Under
#   ``TRNTENSOR_REQUIRE_NKI=1`` a dtype that would take torch instead of the
#   kernel raises ``NotImplementedError`` (:func:`_no_kernel_for_dtype`).
# - Every launch (native op, XLA call, simulator) checks its operands against
#   ``KERNEL_OPERAND_DTYPES`` and raises ``TypeError`` otherwise, so no kernel
#   is reached with a dtype it does not take even if a wrapper forgets the
#   first layer. A native op without a table entry cannot be loaded.

_MATMUL_DTYPES = frozenset({torch.float32, torch.bfloat16, torch.float16})
_CHEM_DTYPES = frozenset({torch.float32, torch.bfloat16})  # mp2_energy, ao_to_mo
_F32_DTYPES = frozenset({torch.float32})
_INDEX_DTYPES = frozenset({torch.uint32})  # gather images (`_tril_gather_images`)

# Allowed dtypes per positional tensor operand, keyed by the op name `_launch`
# uses (and "all_reduce" for `collectives.nki_all_reduce_sum`).
KERNEL_OPERAND_DTYPES: dict[str, tuple[frozenset, ...]] = {
    "matmul": (_MATMUL_DTYPES,) * 2,
    "batched_matmul": (_MATMUL_DTYPES,) * 2,
    "mp2_energy": (_CHEM_DTYPES,) * 4,
    "mp2_energy_row": (_CHEM_DTYPES,) * 6,
    "ao_to_mo_transform": (_CHEM_DTYPES,) * 3,
    "ao_to_mo_trn2": (_CHEM_DTYPES,) * 3,
    "ao_to_mo_trn2_packed": (_CHEM_DTYPES, _INDEX_DTYPES, _CHEM_DTYPES, _CHEM_DTYPES),
    # fp32 only: the blocked kernel (#112) was validated on device in fp32.
    "ao_to_mo_trn2_packed_blocked": (_F32_DTYPES, _INDEX_DTYPES, _F32_DTYPES, _F32_DTYPES),
    # The df_* kernels run fp32 only: bf16 was never validated on device, and
    # the df_coulomb host tail mixed a bf16 B with the kernel's fp32 d.
    "df_half_packed": (_F32_DTYPES, _INDEX_DTYPES, _F32_DTYPES),
    "df_exchange_syrk": (_F32_DTYPES,) * 2,
    "df_coulomb_d": (_F32_DTYPES,) * 2,
    "df_coulomb_j": (_F32_DTYPES,) * 2,
    "all_reduce": (_MATMUL_DTYPES,),
}
# The output takes the first operand's dtype: every operand must share it.
_SAME_DTYPE_KERNELS = frozenset({"matmul", "batched_matmul"})


def _check_kernel_operands(op_name: str, tensors) -> None:
    """Raise ``TypeError`` unless every operand dtype is one ``op_name`` takes."""
    allowed = KERNEL_OPERAND_DTYPES.get(op_name)
    if allowed is None:
        raise TypeError(f"trntensor kernel {op_name!r} has no KERNEL_OPERAND_DTYPES entry")
    tensors = tuple(tensors)
    dts = [t.dtype for t in tensors]
    ok = len(dts) == len(allowed) and all(d in a for d, a in zip(dts, allowed, strict=True))
    if ok and op_name in _SAME_DTYPE_KERNELS:
        ok = len(set(dts)) == 1
    if not ok:
        want = ", ".join("/".join(sorted(str(d) for d in a)) for a in allowed)
        raise TypeError(
            f"trntensor kernel {op_name!r} takes operands ({want}); got "
            f"({', '.join(str(d) for d in dts)}). This is a trntensor bug: the "
            "host wrapper should have cast the operands or run torch."
        )


def _require_nki_now() -> bool:
    return _REQUIRE_NKI or os.environ.get("TRNTENSOR_REQUIRE_NKI", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _no_kernel_for_dtype(what: str, allowed, *tensors: torch.Tensor) -> None:
    """Under ``TRNTENSOR_REQUIRE_NKI=1``, refuse a dtype the kernels do not take.

    Without it the caller runs torch (the pytorch backend's computation)
    instead. Matches trnblas's convention (trnblas#69).
    """
    if _require_nki_now():
        dts = sorted({str(t.dtype) for t in tensors})
        ok = "/".join(sorted(str(d) for d in allowed))
        raise NotImplementedError(
            f"trntensor {what}: the NKI kernels take {ok} operands, got {dts}. "
            "Unset TRNTENSOR_REQUIRE_NKI to compute in torch."
        )


# ── float64 at a kernel entry: the suite policy (#92, trnsci/trnsci#35) ─────
#
# Trainium computes float64 at fp32 precision, so a float64 operand on the
# kernel route cannot get float64 accuracy there:
# - ``"auto"``: torch on the operands' device (the pytorch backend's
#   computation), at float64 accuracy on the host;
# - explicit ``set_backend("nki")`` / ``TRNTENSOR_FORCE_BACKEND=nki``: cast to
#   fp32, run the kernel, warn once per op (UserWarning), and return the dtype
#   the pytorch backend returns;
# - ``TRNTENSOR_REQUIRE_NKI=1``: ``NotImplementedError`` (takes precedence).
_fp64_warned: set[str] = set()

FP64_REDUCED_PRECISION_MESSAGE = (
    "trntensor {what}: float64 operands are computed by the NKI kernel in fp32 "
    '(Trainium has no fp64 datapath) because set_backend("nki") was requested; '
    "the result is returned as {out} but has fp32 accuracy. Use "
    'set_backend("auto") to compute float64 in torch at float64 accuracy. '
    "This warning is shown once per op per process."
)


def _has_fp64(*tensors: torch.Tensor) -> bool:
    return any(t.dtype == torch.float64 for t in tensors)


def _fp64_to_kernel(what: str, *tensors: torch.Tensor) -> bool:
    """Decide a float64 operand set at a kernel entry (the fp64 policy above).

    Returns True when the caller should cast to fp32 and run the kernel
    (explicit NKI; warns once per ``what``), False when it should run torch
    (``"auto"``). Raises ``NotImplementedError`` under
    ``TRNTENSOR_REQUIRE_NKI=1``, before any cast.
    """
    if _require_nki_now():
        raise NotImplementedError(
            f"trntensor {what}: float64 operands would run the NKI kernel at fp32 "
            "precision (Trainium has no fp64 datapath), and TRNTENSOR_REQUIRE_NKI=1 "
            "does not allow that downcast. Pass float32 operands to run the kernel, "
            "or unset TRNTENSOR_REQUIRE_NKI to compute float64 in torch."
        )
    if not _nki_forced():
        return False
    if what not in _fp64_warned:
        _fp64_warned.add(what)
        warnings.warn(
            FP64_REDUCED_PRECISION_MESSAGE.format(what=what, out="float64"),
            UserWarning,
            stacklevel=3,
        )
    return True


# ── float64 already on the neuron device (#92, trnsci/trnsci#35) ────────────
#
# Under ``"auto"`` float64 runs torch on the operands' device. For operands on
# the ``neuron`` device that is device aten, and device aten computes float64
# at fp32 precision (measured on Trn2), so ``"auto"`` would not keep its
# float64-accuracy promise for device-resident data. Every public entry point
# that can compute on float64 / complex128 is wrapped in ``_fp64_host_guard``:
# under ``"auto"``, when an operand is float64 or complex128 on the ``neuron``
# device, the neuron-resident operands are copied to the host, the call runs
# there (host float64 already stays in host torch), and the results go back to
# the operands' device in their float64 / complex128 dtype. One UserWarning per
# op per process. ``TRNTENSOR_FP64_ON_DEVICE=1`` keeps device aten (fp32
# precision) for callers who prefer speed. Explicit ``"nki"`` and
# ``"pytorch"`` backends are not affected. ``TRNTENSOR_REQUIRE_NKI=1`` (NKI
# backends) raises NotImplementedError for float64 / complex128 on the neuron
# device at every guarded entry point, with no host copy and ahead of
# ``TRNTENSOR_FP64_ON_DEVICE`` (`_refuse_device_fp64`): float64 on a kernel
# route raised already, and the routes without a kernel (small or non-matmul
# ``einsum`` contractions, the decompositions, ...) ran device aten at fp32
# precision with no error.
_FP64_HOST_DTYPES = (torch.float64, torch.complex128)
_fp64_host_warned: set[str] = set()
_fp64_host_state = threading.local()  # .active: inside a guarded host call

FP64_HOST_MESSAGE = (
    "trntensor {what}: float64 on the neuron device is computed on the host for "
    "float64 accuracy (device arithmetic is fp32); pass float32 to stay on device, "
    "or set TRNTENSOR_FP64_ON_DEVICE=1 to keep device aten at fp32 precision. "
    "This warning is shown once per op per process."
)


def _fp64_on_device_now() -> bool:
    return _FP64_ON_DEVICE or os.environ.get("TRNTENSOR_FP64_ON_DEVICE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _auto_backend() -> bool:
    """True under ``"auto"`` (no ``TRNTENSOR_FORCE_BACKEND=nki|pytorch``)."""
    forced = os.environ.get("TRNTENSOR_FORCE_BACKEND", "").lower()
    return forced not in ("nki", "pytorch") and _backend == "auto"


def _on_neuron_device(t: torch.Tensor) -> bool:
    """True when ``t`` lives on the TorchNeuron ``neuron`` device."""
    return t.device.type == "neuron"


def _fp64_to_host(t: torch.Tensor) -> torch.Tensor:
    return t.to("cpu")


def _fp64_to_device(t: torch.Tensor, device) -> torch.Tensor:
    return t.to(device)


def _map_tensors(x, fn):
    """``x`` with ``fn`` applied to every tensor in it: tensors, tuples (incl.
    namedtuples), lists, dicts and :class:`~trntensor.parallel.ShardedTensor`."""
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, tuple):
        items = [_map_tensors(v, fn) for v in x]
        return type(x)(*items) if hasattr(x, "_fields") else tuple(items)
    if isinstance(x, list):
        return [_map_tensors(v, fn) for v in x]
    if isinstance(x, dict):
        return {k: _map_tensors(v, fn) for k, v in x.items()}
    from ..parallel import ShardedTensor

    if isinstance(x, ShardedTensor):
        return ShardedTensor([_map_tensors(c, fn) for c in x.chunks], x.partition_dim)
    return x


def _tensor_version(t: torch.Tensor) -> int | None:
    try:
        return t._version
    except RuntimeError:  # inference tensors do not track versions
        return None


def _tensors_in(x) -> list[torch.Tensor]:
    """Every tensor in ``x`` (the structures :func:`_map_tensors` walks)."""
    found: list[torch.Tensor] = []

    def visit(t):
        found.append(t)
        return t

    _map_tensors(x, visit)
    return found


def _fp64_device_operands(args, kwargs) -> list[torch.Tensor]:
    """The float64 / complex128 tensors on the ``neuron`` device in a call."""
    return [
        t
        for t in _tensors_in((args, kwargs))
        if t.dtype in _FP64_HOST_DTYPES and _on_neuron_device(t)
    ]


def _fp64_device_compute(args, kwargs) -> list[torch.Tensor]:
    """The neuron-resident tensors of a call that would compute in float64 /
    complex128 on the device (any backend; empty inside a guarded host call)."""
    if getattr(_fp64_host_state, "active", False):
        return []
    dt = kwargs.get("dtype")
    if dt is not None:
        if _resolve_compute_dtype(dt) not in _FP64_HOST_DTYPES:
            return []  # an explicit compute dtype (einsum's dtype=) decides instead
        # einsum(dtype=float64) asks for a float64 computation even when the device
        # operands are fp32 / bf16: the device would compute it at fp32 precision.
        return [t for t in _tensors_in((args, kwargs)) if _on_neuron_device(t)]
    return _fp64_device_operands(args, kwargs)


def _fp64_host_wanted(args, kwargs) -> list[torch.Tensor]:
    if not _auto_backend() or _require_nki_now() or _fp64_on_device_now():
        return []
    return _fp64_device_compute(args, kwargs)


def _refuse_fp64_on_device(what: str, dtypes) -> None:
    raise NotImplementedError(
        f"trntensor {what}: no NKI kernel computes {', '.join(sorted(set(dtypes)))} on the "
        "neuron device at float64 accuracy (device arithmetic is fp32). Unset "
        "TRNTENSOR_REQUIRE_NKI to compute on the host at float64 accuracy, or pass float32."
    )


def _refuse_device_fp64(what: str, args, kwargs) -> None:
    """Under ``TRNTENSOR_REQUIRE_NKI=1`` (NKI backends), NotImplementedError for
    a float64 / complex128 computation on the neuron device (see above)."""
    if not _require_nki_now() or not _use_nki():
        return
    found = _fp64_device_compute(args, kwargs)
    if found:
        dt = kwargs.get("dtype")
        dts = [str(_resolve_compute_dtype(dt))] if dt is not None else []
        _refuse_fp64_on_device(
            what, dts + [str(t.dtype) for t in found if t.dtype in _FP64_HOST_DTYPES]
        )


def _device_is_neuron(device) -> bool:
    """True for the TorchNeuron ``neuron`` device (patched in the CPU tests)."""
    return torch.device(device).type == "neuron"


def _fp64_host_devices(what: str, dtype: torch.dtype, devices) -> list | None:
    """``[cpu]`` when a ``dtype`` computation that a caller places on
    ``devices`` itself (``trntensor.stream``) must run on the host instead:
    float64 / complex128, a ``neuron`` device among ``devices``, and the host
    path applies (``"auto"``, no REQUIRE_NKI / FP64_ON_DEVICE, not nested).
    Warns once per op, as :func:`_fp64_host_guard`. Otherwise None; under
    ``TRNTENSOR_REQUIRE_NKI=1`` NotImplementedError, as the guard."""
    if dtype not in _FP64_HOST_DTYPES or not any(_device_is_neuron(d) for d in devices):
        return None
    if getattr(_fp64_host_state, "active", False):
        return None
    if _require_nki_now() and _use_nki():
        _refuse_fp64_on_device(what, [str(dtype)])
    if not _auto_backend() or _fp64_on_device_now():
        return None
    if what not in _fp64_host_warned:
        _fp64_host_warned.add(what)
        warnings.warn(FP64_HOST_MESSAGE.format(what=what), UserWarning, stacklevel=3)
    return [torch.device("cpu")]


def _resolve_compute_dtype(dt):
    """einsum's ``dtype=`` (a torch dtype or a name such as "f64") as a torch dtype."""
    if isinstance(dt, torch.dtype):
        return dt
    from ..einsum import _resolve_dtype

    try:
        return _resolve_dtype(dt)
    except (TypeError, ValueError, KeyError):
        return None


def _fp64_host_guard(what: str, results_on_host: bool = False):
    """Decorate a public entry point with the device-resident float64 host path.

    See the section comment above. ``results_on_host`` is for entry points
    whose contract already returns host tensors (``multi_einsum``). Operands
    the call mutates in place (``einsum(out=)``) are copied back into the
    caller's device tensor, which is returned in place of its host copy.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _refuse_device_fp64(what, args, kwargs)
            found = _fp64_host_wanted(args, kwargs)
            if not found:
                return fn(*args, **kwargs)
            device = found[0].device
            if what not in _fp64_host_warned:
                _fp64_host_warned.add(what)
                warnings.warn(FP64_HOST_MESSAGE.format(what=what), UserWarning, stacklevel=2)
            moved: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}

            def to_host(t):
                if not _on_neuron_device(t):
                    return t
                if id(t) not in moved:
                    h = _fp64_to_host(t)
                    moved[id(t)] = (t, h, _tensor_version(h))
                return moved[id(t)][1]

            h_args, h_kwargs = _map_tensors((args, kwargs), to_host)
            prev = getattr(_fp64_host_state, "active", False)
            _fp64_host_state.active = True
            try:
                out = fn(*h_args, **h_kwargs)
            finally:
                _fp64_host_state.active = prev
            returned = {id(t) for t in _tensors_in(out)}
            originals = {}
            for orig, host, version in moved.values():
                # Written in place by the call (an inference tensor has no
                # version counter: then only a returned operand counts).
                now = _tensor_version(host)
                if (now != version) if version is not None else id(host) in returned:
                    orig.copy_(host)
                    originals[id(host)] = orig
            if results_on_host:
                return out

            def back(t):
                if id(t) in originals:
                    return originals[id(t)]
                return _fp64_to_device(t, device)

            return _map_tensors(out, back)

        return wrapper

    return deco


def _fp64_stays_on_host(*tensors: torch.Tensor) -> bool:
    """True when host float64 / complex128 must not be uploaded to the device:
    every backend but explicit NKI computes it in host torch."""
    return not _nki_forced() and any(t.dtype in _FP64_HOST_DTYPES for t in tensors)


def _matmul_kernel_dtypes_ok(*tensors: torch.Tensor) -> bool:
    """True when the matmul / batched-matmul kernels take these operands
    (one shared fp32 / bf16 / fp16 dtype). Otherwise the wrapper runs torch:
    float64 keeps its full precision there, and integer, bool, complex and
    mixed dtypes get exactly the pytorch backend's result (or its error)."""
    dts = {t.dtype for t in tensors}
    return len(dts) == 1 and dts <= _MATMUL_DTYPES


def _chem_kernel_operands(
    what: str, *tensors: torch.Tensor, allowed: frozenset = _CHEM_DTYPES
) -> tuple | None:
    """Operands for the chemistry kernels, or None to take aten.

    float64 follows the fp64 policy (:func:`_fp64_to_kernel`): aten under
    ``"auto"``, fp32 kernel with a one-time warning under explicit NKI,
    ``NotImplementedError`` under ``TRNTENSOR_REQUIRE_NKI=1``. Other real
    floating dtypes outside ``allowed`` (float16; bf16 for the fp32-only df_*
    kernels) are computed in fp32 (a lossless upcast). The caller casts the
    result back to the aten route's dtype. Strides are kept
    (``preserve_format``) so layout detection still sees the caller's layout.
    Integer, bool and complex operands take aten (the pytorch backend's
    computation); under ``TRNTENSOR_REQUIRE_NKI=1`` they raise
    ``NotImplementedError``.
    """
    if not all(t.dtype.is_floating_point for t in tensors):
        _no_kernel_for_dtype(what, allowed, *tensors)
        return None
    if _has_fp64(*tensors) and not _fp64_to_kernel(what, *tensors):
        return None  # "auto": aten at float64 accuracy
    return tuple(t if t.dtype in allowed else t.to(torch.float32) for t in tensors)


def _get_native_ops() -> dict:
    """Import `_native` (registering the trntensor:: custom ops) on first use.

    Every op must have a ``KERNEL_OPERAND_DTYPES`` entry: a new kernel without
    one fails here rather than taking unchecked operands.
    """
    global _native_ops
    if _native_ops is None:
        from . import _native

        missing = sorted(set(_native.OPS) - set(KERNEL_OPERAND_DTYPES))
        if missing:
            raise TypeError(f"trntensor native ops without KERNEL_OPERAND_DTYPES: {missing}")
        _native_ops = _native.OPS
    return _native_ops


def _simulate(kernel, op_name: str, *tensors: torch.Tensor, **consts):
    """``nki.simulate(kernel)`` on host numpy copies, after the dtype check."""
    _check_kernel_operands(op_name, tensors)
    return nki.simulate(kernel)(*(t.cpu().numpy() for t in tensors), **consts)


# ── Eager call handles (native path) ─────────────────────────────────────────
#
# Same fix as trnblas#92. A `trntensor::` custom-op call costs ~0.10 ms of host
# time before the kernel is queued; an aten op costs ~0.02 ms (trn2, SDK 2.33).
# Most of the gap is per-call work that never changes for a given kernel:
#   - nki's `Kernel.__call__` rebuilds the torch handle on every call
#     (`inspect.signature`, a fresh `TorchNativeKernel` hashed into the
#     registry, LNC resolution): ~0.06 ms;
#   - the `torch.library` custom-op layer: ~0.01-0.02 ms.
# So eager calls on plain neuron tensors build a handle once per kernel with
# the public `torch_neuronx.wrap_nki` and call it directly (~0.02 ms). The
# handle is gridless, so NKI resolves the LNC grid itself.
# `**consts` (`use_sr`, `nub`, `tp`, `tail`, `banks`) pass through it by keyword,
# unchanged, as through the op.
# Only ops whose body is a single kernel call qualify: `_native.KERNELS`
# names the kernel each one calls, and the handle wraps that kernel (looked
# up when called, as the op body does; not the caller's), so it runs exactly
# what the op body runs. Ops that are not listed keep the op.
# Under torch.compile, for tensor subclasses (fake / functional), tensors that
# require grad, and int64 inputs (the custom op applies the TorchNeuron int64
# policy), the custom op is used as before.
# `TRNTENSOR_NKI_EAGER_HANDLE=0` turns the fast path off.

_EAGER_HANDLE_ENABLED: bool = os.environ.get(
    "TRNTENSOR_NKI_EAGER_HANDLE", "1"
).strip().lower() not in ("0", "false", "no", "off")
# id(kernel) -> (kernel, handle); holding the kernel keeps its id unique.
_eager_handles: dict[int, tuple[object, object]] = {}
_wrap_nki: object = None  # torch_neuronx.wrap_nki, resolved on first use; False if absent


def _eager_handle(kernel):
    """Torch call handle for `kernel`, built once; None if unavailable."""
    entry = _eager_handles.get(id(kernel))
    if entry is not None:
        return entry[1]
    global _wrap_nki
    if _wrap_nki is None:
        try:
            import torch_neuronx

            _wrap_nki = getattr(torch_neuronx, "wrap_nki", False)
        except ImportError:
            _wrap_nki = False
    if _wrap_nki is False or not hasattr(kernel, "func"):
        return None  # no torch_neuronx.wrap_nki, or not an @nki.jit kernel
    handle = _wrap_nki(kernel)
    _eager_handles[id(kernel)] = (kernel, handle)
    return handle


def _eager_call_ok(tensors) -> bool:
    """True when calling the cached handle is equivalent to the custom op."""
    if not _EAGER_HANDLE_ENABLED or torch.compiler.is_compiling():
        return False
    for t in tensors:
        if (
            type(t) is not torch.Tensor
            or t.device.type != _NEURON_DEVICE
            or t.requires_grad
            or t.dtype == torch.int64
        ):
            return False
    return True


def _native_kernel(op_name: str):
    """`(kernel, conversions)` for the kernel the `trntensor::<op_name>` op body
    calls (looked up now, as the body does), or None for an op that is not a
    single kernel call. `conversions[i]` is what the body applies to
    positional argument i (`int`, `tuple`, `float`), or None to pass it as is."""
    native = sys.modules.get(__package__ + "._native")  # loaded by _get_native_ops
    entry = None if native is None else native.KERNELS.get(op_name)
    if entry is None:
        return None
    kernel = getattr(entry[0], entry[1], None)
    return None if kernel is None else (kernel, entry[2] if len(entry) > 2 else ())


def _native_call(op_name: str, args, kwargs=None):
    """Run `trntensor::<op_name>` on `args` (tensors already on the device, then
    the op's static arguments) and `kwargs` (static
    keyword arguments, passed as is): through the cached handle when
    that is equivalent (see above), else the custom op."""
    kwargs = kwargs or {}
    # Every native launch, through the handle or the op, checks operand
    # dtypes first (trnsci/trnsci#35); `_launch` callers are checked twice.
    _check_kernel_operands(op_name, [a for a in args if isinstance(a, torch.Tensor)])
    ops = _get_native_ops()
    if _eager_call_ok([a for a in args if isinstance(a, torch.Tensor)]):
        entry = _native_kernel(op_name)
        handle = None if entry is None else _eager_handle(entry[0])
        if handle is not None:
            conv = entry[1]
            if conv:
                args = [
                    conv[i](a) if i < len(conv) and conv[i] is not None else a
                    for i, a in enumerate(args)
                ]
            return handle(*args, **kwargs)
    return ops[op_name](*args, **kwargs)


def _launch(kernel, op_name: str, *tensors: torch.Tensor, **consts) -> torch.Tensor:
    """Run one kernel launch on the Neuron device; return it on the caller's device.

    ``op_name`` is the ``trntensor::<op_name>`` custom op used on the native
    path (eager calls on plain tensors skip the op layer through a cached
    handle, see ``_native_call``); the XLA path calls ``kernel`` directly on torch_xla tensors.
    ``consts`` are compile-time kernel arguments (``use_sr``), passed by
    keyword on both paths. The result comes back on the device of
    ``tensors[0]``, so device-resident operands keep the result on the device.
    Operand dtypes are checked against ``KERNEL_OPERAND_DTYPES`` on every path.
    """
    _check_kernel_operands(op_name, tensors)
    if _get_dispatch_path() == "native":
        orig = tensors[0].device
        out = _native_call(op_name, [t.to(_NEURON_DEVICE) for t in tensors], consts)
        return _to_host(out, orig)
    xs, orig = _to_xla(*tensors)
    return kernel(*xs, **consts).to(orig)


# Device types that count as "on Trainium" (either lineage).
_DEVICE_TYPES = ("xla", _NEURON_DEVICE)


def _is_device_tensor(t: torch.Tensor) -> bool:
    """True if ``t`` already lives on a Neuron device (either lineage)."""
    return t.device.type in _DEVICE_TYPES


def _warn_api_deprecated(name: str, replacement: str) -> None:
    if name in _api_deprecations_emitted or not _xla_deprecation_enabled():
        return
    _api_deprecations_emitted.add(name)
    warnings.warn(
        f"trntensor.{name} is deprecated; use trntensor.{replacement}, which works "
        "on both the native TorchNeuron and PyTorch/XLA paths. " + XLA_DEPRECATION_MESSAGE,
        DeprecationWarning,
        stacklevel=3,
    )


def _require_nki(fn_name: str) -> None:
    if not HAS_NKI:
        raise RuntimeError(
            f"{fn_name} requires the NKI runtime. Install trntensor on a Trainium "
            "instance (AWS Deep Learning AMI) or use the CPU path."
        )


def _to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Place ``tensor`` on the active dispatch path's device (never warns)."""
    if _is_device_tensor(tensor):
        return tensor
    _require_nki("to_device")
    if _get_dispatch_path() == "native":
        return tensor.to(_NEURON_DEVICE)
    xla_tensors, _ = _to_xla(tensor)
    return xla_tensors[0]


def _from_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move ``tensor`` back to CPU (never warns). No-op if already on CPU."""
    if tensor.device.type == "cpu":
        return tensor
    return _to_host(tensor, "cpu")


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to the Trainium device of the active dispatch path.

    On TorchNeuron (native, Trn2/Trn3) this is ``tensor.to("neuron")``. On
    the PyTorch/XLA lineage (Trn1) it is the XLA device, exactly as
    :func:`to_xla`. trntensor calls on device-resident operands return
    device-resident results; pull them back with :func:`from_device`.

    Pre-pinning operands and only pulling results back when needed
    eliminates host↔device transfer cost from per-dispatch overhead. For
    pipelines where the same operand is consumed by multiple trntensor
    calls — DF-MP2 (same ``B`` into ``ao_to_mo_transform`` and
    ``mp2_energy``), ``(i,j)`` loops reusing ``B_i``, any multi-step
    workflow — residency is the practical fix for dispatch overhead.

    Raises ``RuntimeError`` on hosts where ``nki`` isn't importable;
    no-op when the tensor is already on a Neuron device.
    """
    return _to_device(tensor)


def from_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor back to CPU. No-op if already on CPU."""
    return _from_device(tensor)


def to_xla(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to the Trainium device (PyTorch/XLA-era name).

    Kept for compatibility; prefer :func:`to_device`. On the PyTorch/XLA
    lineage (Trn1) this moves the tensor to the XLA device, unchanged. On
    TorchNeuron (native, Trn2/Trn3) there is no XLA device, so placement is
    automatic: the tensor goes to the ``neuron`` device and residency works
    the same way.

    Pre-pinning operands via ``to_xla`` and only pulling results back
    with ``from_xla`` when needed eliminates host↔device transfer cost
    from per-dispatch overhead.

    Raises ``RuntimeError`` on hosts where ``nki`` isn't importable;
    no-op when the tensor is already on an XLA device. With
    ``TRNSCI_XLA_DEPRECATION=1`` set, emits a one-time
    ``DeprecationWarning`` pointing at :func:`to_device`.
    """
    _warn_api_deprecated("to_xla", "to_device")
    if tensor.device.type == "xla":
        return tensor
    _require_nki("to_xla")
    if _get_dispatch_path() == "native":
        return tensor.to(_NEURON_DEVICE)
    # Route through the internal _to_xla helper so residency uses exactly
    # the same path as per-kernel dispatch — keeps torch_xla initialization
    # consistent across entry points.
    xla_tensors, _ = _to_xla(tensor)
    return xla_tensors[0]


def from_xla(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor back to CPU. No-op if already on CPU.

    Kept for compatibility; prefer :func:`from_device` (same behaviour on
    either lineage). With ``TRNSCI_XLA_DEPRECATION=1`` set, emits a
    one-time ``DeprecationWarning``.
    """
    _warn_api_deprecated("from_xla", "from_device")
    return _from_device(tensor)


def _stochastic_round_cpu(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """CPU stochastic rounding from float32 to ``dtype``.

    Named stand-in for the per-tile ``nisa.activation(..., round_mode='stochastic')``
    applied at the PSUM→SBUF boundary in the NKI matmul kernel. Enables
    ``precision='sr'`` to be tested on CPU without hardware.

    Rounds each element to its lower or upper neighbor in ``dtype`` with
    probability proportional to the fractional distance from the lower neighbor.
    Mean rounding error is zero (Connolly–Higham–Mary, SIAM J. Sci. Comput. 2021).
    When ``x`` is already in ``dtype`` this is a no-op.
    """
    if x.dtype == dtype:
        return x
    x32 = x.float()
    x_lo = x32.to(dtype)  # round-nearest lower neighbor
    x_lo_f = x_lo.float()
    ulp = (x_lo_f.abs() * torch.finfo(dtype).eps).clamp_min(torch.finfo(dtype).tiny)
    x_hi = (x_lo_f + torch.sign(x32 - x_lo_f) * ulp).to(dtype)
    x_hi_f = x_hi.float()
    gap = (x_hi_f - x_lo_f).abs().clamp_min(1e-38)
    p = ((x32 - x_lo_f).abs() / gap).clamp(0.0, 1.0)
    return torch.where(torch.rand_like(x32) < p, x_hi, x_lo)


def _ozaki_split_cpu(A: torch.Tensor, k: int = 2) -> list[torch.Tensor]:
    """Split FP32 tensor into k BF16-precision summands (stored as FP32).

    CPU stand-in for the per-tile Ogita–Rump–Oishi error-free split used by
    the trnblas Phase 2 double-double GEMM kernel (trnblas#22). Each summand
    holds only BF16-precision information; their sum reconstructs A to FP32
    working precision. The subtraction at each stage is exact by Sterbenz's
    lemma: since s = round_bfloat16(remainder) satisfies |s - remainder| ≤
    u_bf·|s|, the ratio |remainder|/|s| lies in [0.5, 2], meeting the lemma's
    condition for exact FP32 subtraction.
    """
    remainder = A.float()
    summands: list[torch.Tensor] = []
    for _ in range(k):
        s = remainder.to(torch.bfloat16).float()
        summands.append(s)
        remainder = remainder - s
    return summands


def _ozaki_matmul_cpu(A: torch.Tensor, B: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Ozaki-k matmul: accumulates k² BF16-precision products in FP32.

    CPU stand-in for nki_matmul with the trnblas Phase 2 double-double GEMM
    kernel (trnblas#22). Splits each FP32 operand into k BF16-precision
    summands, computes k² matmuls of those summands, and sums the results in
    FP32. Achieves forward error O(K · u_bf^k · |C|) vs O(K · u_bf · |C|)
    for a single BF16 matmul — substantially more accurate for large K.

    The fused NKI version (trnblas#22) holds both main and correction tiles
    simultaneously in PSUM rather than dispatching k² separate kernel calls.
    """
    A_splits = _ozaki_split_cpu(A, k)
    B_splits = _ozaki_split_cpu(B, k)
    result = torch.zeros(A.shape[0], B.shape[1], dtype=torch.float32, device=A.device)
    for Ai in A_splits:
        for Bj in B_splits:
            result = result + torch.matmul(Ai, Bj)
    return result


_DEKKER_SPLIT = 134217729.0  # 2**27 + 1: splits a float64 into two 26-bit halves
_DD_CHUNK_ELEMS = 1 << 22  # products held at once by _dd_matmul_fp64_cpu


def _two_sum(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Knuth TwoSum: s + e == a + b exactly, s = fl(a + b)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _fast_two_sum(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dekker Fast2Sum (|a| >= |b|): s + e == a + b exactly."""
    s = a + b
    return s, b - (s - a)


def _two_prod(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dekker TwoProduct (no FMA): p + e == a * b exactly, barring over/underflow."""
    p = a * b
    t = _DEKKER_SPLIT * a
    ah = t - (t - a)
    al = a - ah
    t = _DEKKER_SPLIT * b
    bh = t - (t - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _dw_add(xh, xl, yh, yl) -> tuple[torch.Tensor, torch.Tensor]:
    """Accurate double-word addition (Joldes-Muller-Popescu 2017, Alg. 6).

    Relative error at most 3u^2 / (1 - 4u) of the exact sum, u = 2**-53.
    """
    sh, sl = _two_sum(xh, yh)
    th, tl = _two_sum(xl, yl)
    vh, vl = _fast_two_sum(sh, sl + th)
    return _fast_two_sum(vh, tl + vl)


def _dd_matmul_fp64_cpu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``A @ B`` for float64 operands with double-double (double-word) accumulation.

    CPU stand-in for ``precision="dd"`` on float64 operands (#84). Each
    product ``a_ik b_kj`` is formed exactly as a double-word with TwoProduct,
    and the K products are summed with accurate double-word addition
    (pairwise inside a chunk of K, sequentially across chunks), then rounded
    once to float64. The accumulation error is at most about
    ``3 K u^2 (|A||B|)_ij`` with u = 2**-53, so the result is within
    ``K u^2 + u_out`` of the exact product to first order: in practice it is
    the correctly rounded value or one ulp from it.

    Batched ``(..., M, K) @ (..., K, N)`` operands are supported. Non-finite
    operands, or products that overflow, fall back to ``torch.matmul`` for
    the affected outputs (inf / nan semantics match plain float64).
    """
    if not (torch.isfinite(A).all() and torch.isfinite(B).all()):
        return torch.matmul(A, B)
    K = A.shape[-1]
    out_shape = torch.broadcast_shapes(A.shape[:-1] + (1,), B.shape[:-2] + (1, B.shape[-1]))
    hi = torch.zeros(out_shape, dtype=torch.float64, device=A.device)
    lo = torch.zeros_like(hi)
    if K == 0:
        return hi
    kc = max(1, min(K, _DD_CHUNK_ELEMS // max(1, hi.numel())))
    for k0 in range(0, K, kc):
        a = A[..., :, k0 : k0 + kc].unsqueeze(-1)  # (..., M, kc, 1)
        b = B[..., k0 : k0 + kc, :].unsqueeze(-3)  # (..., 1, kc, N)
        ph, pl = _two_prod(a, b)  # (..., M, kc, N)
        while ph.shape[-2] > 1:
            if ph.shape[-2] % 2:
                pad = torch.zeros_like(ph[..., :1, :])
                ph = torch.cat([ph, pad], dim=-2)
                pl = torch.cat([pl, pad], dim=-2)
            ph, pl = _dw_add(ph[..., 0::2, :], pl[..., 0::2, :], ph[..., 1::2, :], pl[..., 1::2, :])
        hi, lo = _dw_add(hi, lo, ph[..., 0, :], pl[..., 0, :])
    out = hi + lo
    bad = ~torch.isfinite(out)
    if bad.any():
        out = torch.where(bad, torch.matmul(A, B), out)
    return out


def nki_matmul(A: torch.Tensor, B: torch.Tensor, use_sr: bool = False) -> torch.Tensor:
    """2D matmul ``A @ B`` routed through NKI when available.

    Pads M, K up to tile multiples and N up to ``TILE_N`` when N exceeds
    it (single-N-tile path skips padding). Result is sliced back to the
    original ``(M, N)``. Falls back to ``torch.matmul`` on any kernel
    failure unless ``TRNTENSOR_REQUIRE_NKI=1``.

    ``use_sr=True`` on the native NKI path rounds stochastically only when
    ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1`` was set at process start; otherwise
    a one-time ``RuntimeWarning`` is emitted and the result is
    round-to-nearest-even. The CPU fallback emulates SR in software.
    """
    if not (_use_nki() and HAS_NKI):
        result = _neuron_matmul(A, B)
        return _stochastic_round_cpu(result, A.dtype) if use_sr else result

    M, K = A.shape
    _, N = B.shape
    # Residency- and path-aware executor choice: see _route.
    route = _route_for(M * K * N, A, B)
    if route != "nki":
        if use_sr and (route == "aten" or _is_device_tensor(A)):
            _warn_if_sr_unavailable()
        if route == "aten":
            return _aten_on_device(_neuron_matmul, A, B)
        result = _neuron_matmul(A, B)
        return _stochastic_round_cpu(result, A.dtype) if use_sr else result
    if _neuron_mm_unsafe(A, B):
        # trnblas#112 / #123: no blocked kernel; the blocked aten product.
        return _neuron_matmul(A, B)

    if not _matmul_kernel_dtypes_ok(A, B):
        if A.dtype == B.dtype == torch.float64 and _fp64_to_kernel("matmul", A, B):
            # Explicit NKI: the fp32 kernel, result in float64 (the fp64 policy).
            return nki_matmul(A.float(), B.float(), use_sr=use_sr).to(torch.float64)
        # float64 under "auto" / integer / bool / complex / mixed: torch, as
        # the pytorch backend (float64 reached the fp32 kernel through 0.23.0).
        _no_kernel_for_dtype("matmul", _MATMUL_DTYPES, A, B)
        result = torch.matmul(A, B)
        return _stochastic_round_cpu(result, A.dtype) if use_sr else result

    if _on_native_path():
        _warn_native_kernel_deprecated("matmul")

    from ._kernels import matmul_kernel

    M_pad = _round_up(M, TILE_M)
    K_pad = _round_up(K, TILE_K)
    N_pad = N if N <= TILE_N else _round_up(N, TILE_N)
    needs_pad = (M_pad != M) or (K_pad != K) or (N_pad != N)

    if use_sr and not _use_simulator():
        _warn_if_sr_unavailable()

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
            out_np = _simulate(matmul_kernel, "matmul", A_feed, B_feed, use_sr=use_sr)
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            result = _launch(matmul_kernel, "matmul", A_feed, B_feed, use_sr=use_sr)
        return result[:M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        result = torch.matmul(A, B)
        return _stochastic_round_cpu(result, A.dtype) if use_sr else result


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

    # Reshape ε vectors to 2D at the dispatch boundary. NKI's
    # partition-dim inference is ambiguous on 1D tensor slices when
    # the input is pre-pinned on XLA (bites mp2_energy_kernel's
    # nl.load calls). 2D at entry makes the layout explicit regardless
    # of residency state. See #38 and the trnblas equivalent pattern.
    B_feed = B.contiguous()
    eo_feed = eps_occ.reshape(-1, 1).contiguous()
    ev_feed = eps_vir.reshape(-1, 1).contiguous()

    # Pad NVIR to a TILE_M multiple and NAUX to a TILE_K multiple so the
    # kernel's affine_range loops always see clean tile boundaries.
    # Zero-padding is safe: padded B entries are 0, so their contribution
    # to term = T*(2T-T.T)/Δ is 0 regardless of Δ.
    nvir_pad = _round_up(nvir, TILE_M)
    naux_pad = _round_up(naux, TILE_K)
    needs_pad = nvir_pad != nvir or naux_pad != naux
    if needs_pad:
        B_p = torch.zeros(nocc, nvir_pad, naux_pad, dtype=B_feed.dtype, device=B_feed.device)
        B_p[:, :nvir, :naux] = B_feed
        ev_p = torch.zeros(nvir_pad, 1, dtype=ev_feed.dtype, device=ev_feed.device)
        ev_p[:nvir, :] = ev_feed
        B_feed = B_p.contiguous()
        ev_feed = ev_p.contiguous()

    # Kernel layout (#41): B as (nocc, naux, nvir) so every (P, a) tile is a
    # plain DMA (no load_transpose2d), and the ε_i / ε_b operands
    # pre-broadcast across the TILE_M partitions so the denominator is built
    # without any partition-dim broadcast or reshape on chip.
    B_feed = B_feed.transpose(1, 2).contiguous()
    eo_b = eo_feed.reshape(1, -1).expand(TILE_M, -1).contiguous()
    ev_b = ev_feed.reshape(1, -1).expand(TILE_M, -1).contiguous()

    if _use_simulator():
        out_np = _simulate(mp2_energy_kernel, "mp2_energy", B_feed, eo_b, ev_feed, ev_b)
        return torch.from_numpy(np.asarray(out_np)).to(B.device).sum()

    partial = _launch(mp2_energy_kernel, "mp2_energy", B_feed, eo_b, ev_feed, ev_b)
    return partial.sum()


# ── Native MP2 energy: per-row symmetric kernel, both cores at LNC=2 (#51) ──
#
# SBUF bytes per partition `mp2_energy_row_kernel` keeps live: B[i].T and
# B[j].T (NK * NVIR each) plus the T strip block (NA * NVIR), all fp32. Trn2
# has 192 KiB per partition; leave room for the compiler's temporaries.
_MP2_ROW_SBUF_BUDGET = 144 * 1024


def _mp2_row_dims(nocc: int, nvir: int, naux: int) -> dict:
    """Padded sizes, j-window length and per-d weights for `mp2_energy_row_kernel`.

    Row i takes j = i + d (mod nocc) for d = 0..nocc//2, so every unordered
    pair {i, j} is visited by exactly one (i, d), twice for d = nocc/2 when
    nocc is even. e_ij = e_ji, so off-diagonal d get weight 2 except that
    doubled d, which gets 1. The window is padded to an even length (one j
    per LNC=2 program) with weight-0 entries.
    """
    nvir_pad = _round_up(nvir, TILE_M)
    naux_pad = _round_up(naux, TILE_K)
    half = nocc // 2
    nd = half + 1
    nd_pad = nd + (nd % 2)
    weights = [2.0] * nd_pad
    weights[0] = 1.0
    if half > 0 and nocc % 2 == 0:
        weights[half] = 1.0
    for d in range(nd, nd_pad):
        weights[d] = 0.0
    na, nk = nvir_pad // TILE_M, naux_pad // TILE_K
    return dict(
        nvir_pad=nvir_pad,
        naux_pad=naux_pad,
        nd_pad=nd_pad,
        weights=weights,
        sbuf_bytes=4 * nvir_pad * (2 * nk + na),
        row_flop=2.0 * nd * nvir_pad * nvir_pad * naux_pad,
    )


# Native `mp2_energy` routing (#51). The row kernel launches once per occupied
# row, so it pays ~nocc custom-op launch floors and wins only once each row
# carries enough work. TRNTENSOR_MP2_ENERGY overrides: auto | rows | aten |
# legacy (the original single-launch kernel; deprecated on the native path).
_MP2_ENERGY_CHOICES = ("auto", "rows", "aten", "legacy")
# Shapes whose Trn2 kernel launch failed in this process (e.g. its input
# staging hit NRT_RESOURCE with the device nearly full). "auto" sends them
# straight to aten instead of paying the failed attempt again: at nocc=96,
# nvir=672, naux=2304 with 20 of 24 GB held, each failed attempt cost ~2.3 s
# on top of aten's 1.3 s (#51).
_native_kernel_failed: set[tuple] = set()


def _note_native_kernel_failure(key: tuple) -> None:
    _native_kernel_failed.add(key)


_MP2_ROW_MIN_ROW_FLOP = 3.0e9


# Device memory the row kernel's staging may use, relative to the logical
# core's capacity: leave this much for NEFFs and runtime scratch. With 18 of
# 24 GiB held at nocc=96, nvir=672, naux=2304 (~1.7 GiB of staging) the kernel
# ran; with 20 GiB held its staging failed with NRT_RESOURCE and left the
# runtime thrashing for minutes (#51).
_MP2_ROW_HEADROOM_BYTES = 3 * 2**30


def _mp2_row_staging_bytes(B: torch.Tensor, dims: dict) -> int:
    nocc = B.shape[0]
    padded = nocc * dims["nvir_pad"] * dims["naux_pad"] * 4
    window = (nocc + dims["nd_pad"]) * dims["nvir_pad"] * dims["naux_pad"] * 4
    return padded + window


# Cleared when a gc pass in `_device_has_room` freed nothing; set again by
# every row-kernel call (which can leave a collectable cycle behind).
_gc_may_free = True


def _device_has_room(device: torch.device, nbytes: int) -> bool:
    """False when this process's live device allocations plus ``nbytes`` would
    leave less than ``_MP2_ROW_HEADROOM_BYTES`` of the device. Live, not
    reserved: blocks the caching allocator holds free (e.g. the previous call's
    staging) are reusable. True when the backend does not report memory. It
    only sees this process's allocations."""
    try:
        mod = getattr(torch, _NEURON_DEVICE)
        idx = device.index if device.index is not None else mod.current_device()
        total = mod.get_device_properties(idx).total_memory
        if mod.memory_allocated(idx) + nbytes <= total - _MP2_ROW_HEADROOM_BYTES:
            return True
        # The first launch of a kernel shape leaves its inputs in a reference
        # cycle (measured on TorchNeuron, #51: the ~1 GiB staging buffer at
        # nocc=96 stays allocated until a gc pass). Collect once and recheck,
        # unless the last collection freed nothing and no row kernel has run
        # since (a full gc pass cost ~0.1 s per call with the device full).
        global _gc_may_free
        if not _gc_may_free:
            return False
        import gc

        before = mod.memory_allocated(idx)
        gc.collect()
        after = mod.memory_allocated(idx)
        _gc_may_free = after < before
        return after + nbytes <= total - _MP2_ROW_HEADROOM_BYTES
    except Exception:
        return True


def _mp2_energy_route(B: torch.Tensor) -> str:
    """'rows', 'legacy' or 'aten' for :func:`trntensor.quantum.mp2_energy`.

    XLA path / simulator: 'legacy' (the original kernel) whenever NKI is in
    use, as before. Native path: device-resident ``B`` uses the row kernel
    when it fits in SBUF and each row launch carries at least
    ``_MP2_ROW_MIN_ROW_FLOP`` and the device has room for its input staging,
    else aten; ``set_backend("nki")`` uses the row kernel whenever it fits in
    SBUF. Under ``"auto"``, :func:`trntensor.quantum.mp2_energy` uploads a
    large host ``B`` first and then asks again (#56); a host ``B`` that
    reaches this function gets 'aten', on the host.
    """
    if not (_use_nki() and HAS_NKI):
        return "aten"
    if not _on_native_path():
        return "legacy"
    choice = os.environ.get("TRNTENSOR_MP2_ENERGY", "auto").strip().lower()
    if choice not in _MP2_ENERGY_CHOICES:
        raise ValueError(
            f"TRNTENSOR_MP2_ENERGY must be one of {_MP2_ENERGY_CHOICES}; got {choice!r}"
        )
    if choice != "auto":
        return choice
    dims = _mp2_row_dims(*B.shape)
    fits = dims["sbuf_bytes"] <= _MP2_ROW_SBUF_BUDGET
    if _nki_forced():
        return "rows" if fits else "aten"
    if B.device.type != _NEURON_DEVICE or ("mp2", tuple(B.shape)) in _native_kernel_failed:
        return "aten"
    if not fits or dims["row_flop"] < _MP2_ROW_MIN_ROW_FLOP:
        return "aten"
    return "rows" if _device_has_room(B.device, _mp2_row_staging_bytes(B, dims)) else "aten"


def _mp2_row_feeds(B: torch.Tensor, eps_occ: torch.Tensor, eps_vir: torch.Tensor, dims: dict):
    """Device operands for `mp2_energy_row_kernel`: B[j].T for j = 0..nocc-1
    followed by the first ``nd_pad`` again (each row's wrapped j window is then
    a contiguous slice), the matching eps window and the padded eps_vir column
    / row. All fp32. B[i].T is a view of the same buffer, so the feed costs
    about 1.5x the padded B instead of 3x (it matters near the HBM limit)."""
    nocc, nvir, naux = B.shape
    nv, na = dims["nvir_pad"], dims["naux_pad"]
    B = B.float()
    if (nv, na) != (nvir, naux):
        Bp = torch.zeros(nocc, nv, na, dtype=B.dtype, device=B.device)
        Bp[:, :nvir, :naux] = B
    else:
        Bp = B
    nd_pad = dims["nd_pad"]
    BpT = Bp.permute(0, 2, 1)  # (nocc, naux_pad, nvir_pad) view
    n_win = nocc + nd_pad
    reps = -(-n_win // nocc)  # > 2 only when nd_pad > nocc (nocc ≤ 2)
    if reps <= 2:
        Bx = torch.cat([BpT, BpT[:nd_pad]], dim=0).contiguous()
    else:
        Bx = torch.cat([BpT] * reps, dim=0)[:n_win].contiguous()
    del Bp, BpT
    Bt = Bx[:nocc]
    eo = eps_occ.to(device=B.device, dtype=B.dtype).reshape(1, nocc)
    ex = torch.cat([eo] * reps, dim=1)[:, :n_win].contiguous()
    ev = torch.zeros(nv, dtype=B.dtype, device=B.device)
    ev[:nvir] = eps_vir.to(device=B.device, dtype=B.dtype)
    return Bt, Bx, eo, ex, ev.reshape(nv, 1).contiguous(), ev.reshape(1, nv).contiguous()


def _mp2_row_reference(B: torch.Tensor, eps_occ: torch.Tensor, eps_vir: torch.Tensor):
    """fp64 torch emulation of the row kernel's (i, d) walk and weights (CPU tests)."""
    nocc = B.shape[0]
    dims = _mp2_row_dims(*B.shape)
    Bt, Bx, eo, ex, evc, evr = _mp2_row_feeds(B, eps_occ, eps_vir, dims)
    w = torch.tensor(dims["weights"], dtype=torch.float64)
    total = torch.zeros((), dtype=torch.float64)
    for i in range(nocc):
        for d in range(dims["nd_pad"]):
            bi, bj = Bt[i].double(), Bx[i + d].double()
            T = bi.T @ bj
            denom = float(eo[0, i]) + float(ex[0, i + d]) - evc.double() - evr.double()
            total = total + w[d] * (T * (2.0 * T - T.T) / denom).sum()
    return total


def _nki_mp2_energy_rows(
    B: torch.Tensor, eps_occ: torch.Tensor, eps_vir: torch.Tensor
) -> torch.Tensor:
    """MP2 energy via `mp2_energy_row_kernel`, one launch per occupied row (native).

    Never materializes T (nocc² nvir²) in device memory: each launch holds one
    B[i].T, one B[j].T and one T strip block in SBUF. Device memory beyond the
    inputs is the doubled, padded, transposed copy of B (2 * nocc * naux * nvir)
    and a (128, ND) partial per row.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    nocc = B.shape[0]
    dims = _mp2_row_dims(*B.shape)
    if dims["sbuf_bytes"] > _MP2_ROW_SBUF_BUDGET:
        raise ValueError(
            f"mp2_energy_row needs {dims['sbuf_bytes']} SBUF bytes per partition "
            f"(budget {_MP2_ROW_SBUF_BUDGET}) for B of shape {tuple(B.shape)}"
        )
    orig = B.device
    if B.device.type != _NEURON_DEVICE:
        B = B.to(_NEURON_DEVICE)
    Bt, Bx, eo, ex, evc, evr = _mp2_row_feeds(B, eps_occ, eps_vir, dims)
    nd_pad = dims["nd_pad"]
    global _gc_may_free
    _gc_may_free = True
    acc = None
    for i in range(nocc):
        p = _native_call(
            "mp2_energy_row",
            [
                Bt[i],
                Bx[i : i + nd_pad],
                eo[:, i : i + 1].contiguous(),
                ex[:, i : i + nd_pad].contiguous(),
                evc,
                evr,
            ],
        )
        acc = p if acc is None else acc + p
    w = torch.tensor(dims["weights"], dtype=acc.dtype).to(acc.device)
    return (acc.sum(0) * w).sum().to(orig)


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
    naux = eri.shape[2]
    nocc = C_occ.shape[1]
    nvir = C_vir.shape[1]

    # Pad nbasis to a TILE_K (128) multiple — the kernel's step-1 ν-tile and
    # μ/ν K-tile loops are all TILE_K-wide (#41).
    # Pad nvir to a TILE_N multiple — the kernel's step-2 a-tile loop requires
    # this. Slice the result back to (nocc, nvir, naux) after kernel returns.
    nbasis_pad = _round_up(nbasis, TILE_K)
    nvir_pad = _round_up(nvir, TILE_N)
    needs_pad = nbasis_pad != nbasis or nvir_pad != nvir

    if needs_pad:
        eri_p = torch.zeros(nbasis_pad, nbasis_pad, naux, dtype=eri.dtype, device=eri.device)
        eri_p[:nbasis, :nbasis, :] = eri
        C_occ_p = torch.zeros(nbasis_pad, nocc, dtype=C_occ.dtype, device=C_occ.device)
        C_occ_p[:nbasis, :] = C_occ
        C_vir_p = torch.zeros(nbasis_pad, nvir_pad, dtype=C_vir.dtype, device=C_vir.device)
        C_vir_p[:nbasis, :nvir] = C_vir
        eri_feed = eri_p.contiguous()
        C_occ_feed = C_occ_p.contiguous()
        C_vir_feed = C_vir_p.contiguous()
    else:
        eri_feed = eri.contiguous()
        C_occ_feed = C_occ.contiguous()
        C_vir_feed = C_vir.contiguous()

    if _use_simulator():
        out_np = _simulate(
            ao_to_mo_transform_kernel, "ao_to_mo_transform", eri_feed, C_occ_feed, C_vir_feed
        )
        result = torch.from_numpy(np.asarray(out_np)).to(eri.device)
        return result[:, :nvir, :] if nvir_pad != nvir else result

    result = _launch(
        ao_to_mo_transform_kernel, "ao_to_mo_transform", eri_feed, C_occ_feed, C_vir_feed
    )
    return result[:, :nvir, :] if nvir_pad != nvir else result


# Native `ao_to_mo_transform` routing (#51). TRNTENSOR_AO_TO_MO overrides:
# auto | trn2 | aten | legacy (the original per-P kernel; deprecated on the
# native path).
_AO_TO_MO_CHOICES = ("auto", "trn2", "aten", "legacy")
# SBUF bytes per partition `ao_to_mo_trn2_kernel` keeps live (C_occ, C_vir,
# one stage-1 eri block and X block, one stage-2 X group and B block); same
# budget as df_exchange's launch 1 (of 192 KiB per partition). It was 144 KiB;
# n_ao = 970, nocc = 161, packed naux = 4608 needs 158 KiB and compiles and
# runs at 118 ms, where aten runs out of device memory (#80).
_AO_TO_MO_SBUF_BUDGET = 176 * 1024
# Under "auto" the kernel needs stage-1 work nocc·nbasis²·naux of at least
# this many multiply-adds. Below it, the custom-op launch floor and the
# padding copies lose to aten: nbasis=128, nocc=16, naux=256 (6.7e7) measured
# 0.57 ms against 0.49 ms; nbasis=256, nocc=32, naux=512 (1.1e9) 0.92 ms
# against 1.23 ms (#51).
_AO_TO_MO_MIN_MACS = 5.0e8
# Two more conditions from the nocc > 128 measurements (#80, Trn2, LNC=2):
# - A full eri whose nbasis or naux is not a 128 multiple stays on aten: the
#   kernel pads a copy of the whole square first, and that copy alone costs
#   more than aten. 300/100/700 (6.3e9 multiply-adds) took 5.6 ms against
#   2.0 ms, 300/129/700 6.3 against 2.6 ms; the unpadded 384/129/768 2.7
#   against 4.2 ms. Packed eri pads nbasis only in the gather indices.
# - A packed eri with nbasis padded wastes the padded rows' Tensor Engine
#   work, so it needs more work: 160/129/256 (8.5e8) took 2.1 against 1.6 ms,
#   192/150/384 (2.1e9) 2.1 against 2.2 ms, 300/129/700 (8.1e9) 4.1 against
#   5.5 ms.
_AO_TO_MO_MIN_MACS_PADDED = 2.0e9


def _ao_to_mo_trn2_sbuf_bytes(
    nbasis: int, nocc: int, nvir: int, naux: int, itemsize: int, packed: bool = False
) -> int:
    nkb = _round_up(nbasis, TILE_K) // TILE_K
    tp, nub = ao_to_mo_trn2_tiling(_round_up(naux, TILE_K), packed)
    # Stage 2 moves up to 512 columns (NI orbitals of one chunk) per matmul.
    gw = max(1, min(nocc, 512 // tp)) * tp
    cols = nkb * (nocc + _round_up(nvir, TILE_M) + nub * tp + gw) + nub * tp + gw
    # The packed kernel also keeps one chunk's gather indices (uint32).
    idx = 4 * nkb * nkb * TILE_K if packed else 0
    return itemsize * cols + idx


def _ao_to_mo_route(eri: torch.Tensor, nocc: int, nvir: int, nbasis: int | None = None) -> str:
    """'trn2', 'legacy' or 'aten' for :func:`trntensor.quantum.ao_to_mo_transform`.

    XLA path / simulator: 'legacy' whenever NKI is in use (unchanged).
    Native path: the Trn2 kernel for device-resident ``eri`` when
    ``nocc ≤ 128``, its SBUF working set fits and the stage-1 work reaches
    ``_AO_TO_MO_MIN_MACS``, else aten; ``set_backend("nki")`` drops the size
    and residency conditions. Under ``"auto"``,
    :func:`trntensor.quantum.ao_to_mo_transform` uploads large host inputs
    first and then asks again (#56); a host ``eri`` that reaches this
    function gets 'aten', on the host.

    A 2-D ``eri`` is the packed lower triangle ``(npair, naux)`` of an
    ``nbasis`` basis (#54); then 'trn2' means ``ao_to_mo_trn2_packed_kernel``
    and 'legacy' is not offered (the caller unpacks for it).
    """
    if not (_use_nki() and HAS_NKI):
        return "aten"
    if not _on_native_path():
        return "legacy" if _use_fused_kernel() else "aten"
    choice = os.environ.get("TRNTENSOR_AO_TO_MO", "auto").strip().lower()
    if choice not in _AO_TO_MO_CHOICES:
        raise ValueError(f"TRNTENSOR_AO_TO_MO must be one of {_AO_TO_MO_CHOICES}; got {choice!r}")
    if choice != "auto":
        return choice
    packed = eri.dim() == 2
    nb = (nbasis if packed else eri.shape[0]) or 0
    naux = eri.shape[-1]
    fits = (
        _ao_to_mo_trn2_sbuf_bytes(nb, nocc, nvir, naux, eri.element_size(), packed)
        <= _AO_TO_MO_SBUF_BUDGET
    )
    if not fits and packed:
        # The blocked kernel's SBUF does not grow with n_ao (#112).
        fits = _ao_to_mo_packed_variant(nb, nocc, nvir, naux, eri.dtype) == "blocked"
    if not fits:
        return "aten"
    if _nki_forced():
        return "trn2"
    if ("ao2mo", tuple(eri.shape), nocc, nvir) in _native_kernel_failed:
        return "aten"
    if not packed and (nb % TILE_K or naux % TILE_K):
        # The kernel would pad a copy of the full square first (#80).
        return "aten"
    macs = nocc * nb**2 * naux
    big = macs >= (_AO_TO_MO_MIN_MACS_PADDED if nb % TILE_K else _AO_TO_MO_MIN_MACS)
    return "trn2" if big and eri.device.type == _NEURON_DEVICE else "aten"


_AO_TO_MO_PACKED_CHOICES = ("auto", "resident", "blocked")


def _ao_to_mo_packed_variant(
    nbasis: int, nocc: int, nvir: int, naux: int, dtype: torch.dtype
) -> str | None:
    """Which packed Trn2 AO→MO kernel runs: 'resident' (``ao_to_mo_trn2_packed_kernel``),
    'blocked' (``ao_to_mo_trn2_packed_blocked_kernel``, #112) or None (neither).

    ``"auto"`` keeps the resident kernel wherever its SBUF working set fits
    (it was measured there, #54/#80, and is up to ~2x faster at mid sizes:
    14 against 18-27 ms at n_ao 970 / nocc 161 / naux 512) and takes the
    blocked one beyond, from n_ao ≈ 1100-1300 at nocc ≈ n_ao/10, where
    C_vir, the gather indices and the stage-1 eri column pass the budget.
    The blocked kernel runs fp32 only.
    ``TRNTENSOR_AO_TO_MO_PACKED=auto|resident|blocked`` overrides.
    """
    choice = os.environ.get("TRNTENSOR_AO_TO_MO_PACKED", "auto").strip().lower()
    if choice not in _AO_TO_MO_PACKED_CHOICES:
        raise ValueError(
            f"TRNTENSOR_AO_TO_MO_PACKED must be one of {_AO_TO_MO_PACKED_CHOICES}; got {choice!r}"
        )
    itemsize = torch.empty((), dtype=dtype).element_size()
    resident = (
        _ao_to_mo_trn2_sbuf_bytes(nbasis, nocc, nvir, naux, itemsize, True) <= _AO_TO_MO_SBUF_BUDGET
    )
    tp = ao_to_mo_blocked_tp(naux) or ao_to_mo_blocked_tp(_round_up(naux, TILE_K))
    blocked = (
        dtype == torch.float32
        and ao_to_mo_blocked_sbuf_bytes(nocc, tp, _round_up(nbasis, TILE_K))
        <= _AO_TO_MO_SBUF_BUDGET
    )
    if choice == "resident" or (choice == "auto" and resident):
        return "resident"
    return "blocked" if blocked else None


def _ao_to_mo_pad_occ(C_occ: torch.Tensor, naux_pad: int, packed: bool) -> torch.Tensor:
    """``C_occ`` with a zero orbital appended when the kernel splits the
    occupied index across programs and ``nocc`` is odd.

    ``ao_to_mo_trn2_kernel`` splits the aux chunks across the two programs
    of an LNC=2 core; with an odd chunk count and ``nocc > 128`` it splits
    the occupied index instead, which needs an even ``nocc`` (#80). The
    zero orbital gives a zero row of ``B``, sliced off (a view) on return.
    """
    nocc = C_occ.shape[1]
    tp = ao_to_mo_trn2_tiling(naux_pad, packed)[0]
    if nocc <= TILE_M or nocc % 2 == 0 or (naux_pad // tp) % 2 == 0:
        return C_occ
    co = torch.zeros(C_occ.shape[0], nocc + 1, dtype=C_occ.dtype, device=C_occ.device)
    co[:, :nocc] = C_occ
    return co


def _nki_ao_to_mo_trn2(
    eri: torch.Tensor,
    C_occ: torch.Tensor,
    C_vir: torch.Tensor,
) -> torch.Tensor:
    """Dispatch ``ao_to_mo_trn2_kernel`` (native path, #51).

    Pads nbasis, nvir and naux to 128 multiples (zero padding contributes
    nothing) and slices the result back to ``(nocc, nvir, naux)``.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    nbasis, _, naux = eri.shape
    nocc = C_occ.shape[1]
    nvir = C_vir.shape[1]
    nbp, nvp, nxp = _round_up(nbasis, TILE_K), _round_up(nvir, TILE_M), _round_up(naux, TILE_K)
    if (nbp, nxp) != (nbasis, naux):
        eri_p = torch.zeros(nbp, nbp, nxp, dtype=eri.dtype, device=eri.device)
        eri_p[:nbasis, :nbasis, :naux] = eri
        eri = eri_p
    if nbp != nbasis:
        co = torch.zeros(nbp, nocc, dtype=C_occ.dtype, device=C_occ.device)
        co[:nbasis] = C_occ
        C_occ = co
    if (nbp, nvp) != (nbasis, nvir):
        cv = torch.zeros(nbp, nvp, dtype=C_vir.dtype, device=C_vir.device)
        cv[:nbasis, :nvir] = C_vir
        C_vir = cv
    C_occ = _ao_to_mo_pad_occ(C_occ, nxp, packed=False)
    out = _launch(None, "ao_to_mo_trn2", eri.contiguous(), C_occ.contiguous(), C_vir.contiguous())
    if out.shape[0] != nocc:
        out = out[:nocc]
    return out[:, :nvir, :naux] if (nvp, nxp) != (nvir, naux) else out


def _tril_rows(nbasis: int) -> torch.Tensor:
    """``(nbasis, nbasis)`` int64: packed row ``max(μ,ν)(max+1)/2 + min`` of ``(μ, ν)``."""
    r = torch.arange(nbasis, dtype=torch.int64)
    hi = torch.maximum(r[:, None], r[None, :])
    lo = torch.minimum(r[:, None], r[None, :])
    return hi * (hi + 1) // 2 + lo


def _tril_gather_images(nbasis: int, nbp: int, naux: int, tp: int, device=None) -> torch.Tensor:
    """Gather indices for ``ao_to_mo_trn2_packed_kernel``: ``(NPC, 128, NKB·nbp)`` uint32.

    The kernel views the packed ``(npair, naux)`` tensor as ``(npair·NPC,
    tp)`` rows, so aux chunk ``c`` of pair ``r`` is row ``NPC·r + c``. One
    batched indirect DMA fills the SBUF block ``e[p, u, :] = eri[k·128 + p,
    v0 + u, chunk c]`` (``p < 128``, ``u < NUB``) from a ``(128, NUB)``
    index block that the DMA reads column-major and writes row-major: the
    ``j``-th row (``j = m·128 + q`` for index ``[q, m]``) lands at ``p = j //
    NUB``, ``u = j % NUB``. Block ``(k, v0)`` sits at columns ``k·nbp + v0``.

    Rows/columns past ``nbasis`` (padding) read pair 0; the zero padding of
    ``C_occ``/``C_vir`` cancels them. Cached per shape and ``device``
    (``4·NPC·nbp²`` bytes, so repeat calls skip the upload).
    """
    key = (nbasis, nbp, naux, tp, str(device))
    hit = _gather_image_cache.get(key)
    if hit is not None:
        return hit
    npc = naux // tp
    nub = ao_to_mo_trn2_nub(tp)
    nkb = nbp // TILE_K
    rows = torch.zeros(nbp, nbp, dtype=torch.int64)
    rows[:nbasis, :nbasis] = _tril_rows(nbasis)
    q = torch.arange(TILE_K)[:, None]
    m = torch.arange(nub)[None, :]
    j = m * TILE_K + q  # (128, NUB)
    dp, du = j // nub, j % nub
    k = torch.arange(nkb)[:, None, None, None]
    v0 = (torch.arange(nbp // nub) * nub)[None, :, None, None]
    mu = k * TILE_K + dp[None, None]  # (NKB, NB/NUB, 128, NUB)
    nu = v0 + du[None, None]
    base = rows[mu, nu].permute(2, 0, 1, 3).reshape(TILE_K, nkb * nbp)
    img = base[None] * npc + torch.arange(npc)[:, None, None]
    if int(img.max()) >= 2**31:
        raise ValueError("packed eri too large for 32-bit gather indices")
    hit = img.to(torch.int32).view(torch.uint32).contiguous()
    if device is not None:
        hit = hit.to(device)
    _gather_image_cache[key] = hit
    return hit


_gather_image_cache: dict[tuple, torch.Tensor] = {}


def _tril_gather_images_blocked(
    nbasis: int, nbp: int, naux: int, tp: int, nub: int, device=None
) -> torch.Tensor:
    """Gather indices for ``ao_to_mo_trn2_packed_blocked_kernel``:
    ``(NPC, nbp/NUB, 128, NKB·NUB)`` uint32 (#112).

    The same ``(128, NUB)`` index blocks as :func:`_tril_gather_images`
    (rows ``NPC·pair + c`` of the ``(npair·NPC, tp)`` view, read column-major
    and written row-major), regrouped so that for chunk ``c`` and ν block
    ``vb`` the blocks of consecutive μ tiles ``k`` are contiguous (columns
    ``k·NUB``): one small DMA brings a whole μ block's indices. Cached per
    shape and device (``4·NPC·nbp²`` bytes).
    """
    key = ("blk", nbasis, nbp, naux, tp, nub, str(device))
    hit = _gather_image_cache.get(key)
    if hit is not None:
        return hit
    npc = naux // tp
    nkb = nbp // TILE_K
    nvb = nbp // nub
    if (nbasis * (nbasis + 1) // 2) * npc >= 2**31:
        raise ValueError("packed eri too large for 32-bit gather indices")
    rows = torch.zeros(nbp, nbp, dtype=torch.int64)
    rows[:nbasis, :nbasis] = _tril_rows(nbasis)
    q = torch.arange(TILE_K)[:, None]
    m = torch.arange(nub)[None, :]
    j = m * TILE_K + q  # (128, NUB)
    dp, du = j // nub, j % nub
    k = torch.arange(nkb)[None, :, None, None]
    v0 = (torch.arange(nvb) * nub)[:, None, None, None]
    mu = k * TILE_K + dp[None, None]  # (NVB, NKB, 128, NUB)
    nu = v0 + du[None, None]
    base = rows[mu, nu].permute(0, 2, 1, 3).reshape(nvb, TILE_K, nkb * nub)
    hit = (base[None] * npc + torch.arange(npc)[:, None, None, None]).to(torch.int32)
    hit = hit.view(torch.uint32).contiguous()
    if device is not None:
        hit = hit.to(device)
    _gather_image_cache[key] = hit
    return hit


def _packed_tp(naux: int) -> int:
    return ao_to_mo_trn2_tiling(naux, packed=True)[0]


def _nki_ao_to_mo_trn2_packed(
    eri_p: torch.Tensor,
    C_occ: torch.Tensor,
    C_vir: torch.Tensor,
) -> torch.Tensor:
    """Dispatch ``ao_to_mo_trn2_packed_kernel`` on packed ``(npair, naux)`` integrals (#54).

    Pads naux to a 128 multiple (a copy of the packed tensor, half the size
    of the full one) and nbasis/nvir through ``C_occ``/``C_vir`` and the
    gather indices; slices the result back to ``(nocc, nvir, naux)``.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    npair, naux = eri_p.shape
    nbasis, nocc = C_occ.shape
    nvir = C_vir.shape[1]
    if _ao_to_mo_packed_variant(nbasis, nocc, nvir, naux, eri_p.dtype) == "blocked":
        return _nki_ao_to_mo_trn2_packed_blocked(eri_p, C_occ, C_vir)
    nbp, nvp, nxp = _round_up(nbasis, TILE_K), _round_up(nvir, TILE_M), _round_up(naux, TILE_K)
    if nxp != naux:
        ep = torch.zeros(npair, nxp, dtype=eri_p.dtype, device=eri_p.device)
        ep[:, :naux] = eri_p
        eri_p = ep
    if nbp != nbasis:
        co = torch.zeros(nbp, nocc, dtype=C_occ.dtype, device=C_occ.device)
        co[:nbasis] = C_occ
        C_occ = co
    if (nbp, nvp) != (nbasis, nvir):
        cv = torch.zeros(nbp, nvp, dtype=C_vir.dtype, device=C_vir.device)
        cv[:nbasis, :nvir] = C_vir
        C_vir = cv
    dev = eri_p.device if eri_p.device.type == _NEURON_DEVICE else None
    tp = _packed_tp(nxp)
    img = _tril_gather_images(nbasis, nbp, nxp, tp, dev)
    C_occ = _ao_to_mo_pad_occ(C_occ, nxp, packed=True)
    out = _launch(
        None,
        "ao_to_mo_trn2_packed",
        eri_p.contiguous().view(npair * (nxp // tp), tp),
        img,
        C_occ.contiguous(),
        C_vir.contiguous(),
    )
    if out.shape[0] != nocc:
        out = out[:nocc]
    return out[:, :nvir, :naux] if (nvp, nxp) != (nvir, naux) else out


def _nki_ao_to_mo_trn2_packed_blocked(
    eri_p: torch.Tensor,
    C_occ: torch.Tensor,
    C_vir: torch.Tensor,
) -> torch.Tensor:
    """Dispatch ``ao_to_mo_trn2_packed_blocked_kernel`` (#112).

    The aux chunk width is a divisor of naux (:func:`ao_to_mo_blocked_tp`),
    so the packed tensor goes in without a padding copy unless naux has no
    divisor in ``[64, 512]``. nbasis and nvir pad through ``C_occ``/``C_vir``
    and the gather indices, and an odd nocc above 128 gets a zero orbital
    when the programs split the occupied index (odd chunk count).
    """
    npair, naux = eri_p.shape
    nbasis, nocc = C_occ.shape
    nvir = C_vir.shape[1]
    nbp, nvp = _round_up(nbasis, TILE_K), _round_up(nvir, TILE_M)
    tp, nxp = ao_to_mo_blocked_tp(naux), naux
    if not tp:
        nxp = _round_up(naux, TILE_K)
        tp = ao_to_mo_blocked_tp(nxp)
        ep = torch.zeros(npair, nxp, dtype=eri_p.dtype, device=eri_p.device)
        ep[:, :naux] = eri_p
        eri_p = ep
    nocp = nocc + 1 if nocc > TILE_M and nocc % 2 and (nxp // tp) % 2 else nocc
    if (nbp, nocp) != (nbasis, nocc):
        co = torch.zeros(nbp, nocp, dtype=C_occ.dtype, device=C_occ.device)
        co[:nbasis, :nocc] = C_occ
        C_occ = co
    if (nbp, nvp) != (nbasis, nvir):
        cv = torch.zeros(nbp, nvp, dtype=C_vir.dtype, device=C_vir.device)
        cv[:nbasis, :nvir] = C_vir
        C_vir = cv
    dev = eri_p.device if eri_p.device.type == _NEURON_DEVICE else None
    nub = ao_to_mo_blocked_nub(tp, nocp)
    img = _tril_gather_images_blocked(nbasis, nbp, nxp, tp, nub, dev)
    out = _launch(
        None,
        "ao_to_mo_trn2_packed_blocked",
        eri_p.contiguous().view(npair * (nxp // tp), tp),
        img,
        C_occ.contiguous(),
        C_vir.contiguous(),
    )
    if out.shape[0] != nocc:
        out = out[:nocc]
    return out[:, :nvir, :naux] if (nvp, nxp) != (nvir, naux) else out


def _df_gather_images(
    nbasis: int, nbp: int, naux: int, tp: int, nub: int, nvc: int, device=None
) -> torch.Tensor:
    """Gather indices for ``df_half_packed_kernel``: ``(NPC, NVG, 128, NKB·NVC)`` uint32.

    Chunk ``c`` of pair ``r`` starts at element ``r·naux + c·tp`` of the
    packed tensor, which the kernel views as ``(npair·naux / W, W)`` rows,
    ``W = gcd(naux, tp)``: its index is row ``(r·naux + c·tp) / W``. With
    ``tp`` dividing naux that is row ``NPC·r + c`` of the ``(npair·NPC, tp)``
    view (:func:`_tril_gather_images`); otherwise the last chunk is the
    short ``naux mod tp`` one. ``(128, NUB)`` blocks are read column-major
    and written row-major, regrouped so one contiguous DMA per (chunk, ν
    group of ``nvc``) brings every μ-tile's indices for that group. Cached
    per shape and device (``4·NPC·nbp²`` bytes).
    """
    key = ("dfk", nbasis, nbp, naux, tp, nub, nvc, str(device))
    hit = _gather_image_cache.get(key)
    if hit is not None:
        return hit
    w0 = math.gcd(naux, tp)
    npc = -(-naux // tp)
    nkb = nbp // TILE_K
    nvg = nbp // nvc
    rows = torch.zeros(nbp, nbp, dtype=torch.int64)
    rows[:nbasis, :nbasis] = _tril_rows(nbasis)
    q = torch.arange(TILE_K)[:, None]
    m = torch.arange(nub)[None, :]
    j = m * TILE_K + q  # (128, NUB)
    dp, du = j // nub, j % nub
    k = torch.arange(nkb)[:, None, None, None]
    v0 = (torch.arange(nbp // nub) * nub)[None, :, None, None]
    mu = k * TILE_K + dp[None, None]  # (NKB, NB/NUB, 128, NUB)
    nu = v0 + du[None, None]
    base = rows[mu, nu].reshape(nkb, nvg, nvc // nub, TILE_K, nub)
    base = base.permute(1, 3, 0, 2, 4).reshape(nvg, TILE_K, nkb * nvc)
    img = (base[None] * naux + (torch.arange(npc) * tp)[:, None, None, None]) // w0
    if int(img.max()) >= 2**31:
        raise ValueError("packed eri too large for 32-bit gather indices")
    hit = img.to(torch.int32).view(torch.uint32).contiguous()
    if device is not None:
        hit = hit.to(device)
    _gather_image_cache[key] = hit
    return hit


_DF_EXCHANGE_CHOICES = ("auto", "trn2", "aten")
# Under "auto" the kernels need nocc·nbasis²·naux of at least this many
# multiply-adds (stage 1). Trn2, one LNC=2 core, warm (#60): water
# (24/116/5, 3.3e5) 1.32 ms kernels vs 1.16 ms aten; benzene (114/558/21,
# 1.5e8) 1.35 vs 2.24 ms; n_ao 250 (5.0e9) 2.2 vs 9.0 ms.
_DF_EXCHANGE_MIN_MACS = 1.0e7


def _df_exchange_route(B: torch.Tensor, nocc: int, nbasis: int) -> str:
    """'trn2' or 'aten' for :func:`trntensor.quantum.df_exchange` on packed ``B``.

    'trn2' needs the native path, NKI in use, ``nocc ≤ 512`` and a kernel
    tiling (``df_half_tiling``: an aux chunk width ≤ 512 and an SBUF fit).
    Under ``"auto"`` it also needs ``B`` on the device and ``nocc·nbasis²·naux
    ≥ _DF_EXCHANGE_MIN_MACS``; ``set_backend("nki")`` drops those two.
    ``TRNTENSOR_DF_EXCHANGE=auto|trn2|aten`` overrides.
    """
    if not (_use_nki() and HAS_NKI) or not _on_native_path():
        return "aten"
    choice = os.environ.get("TRNTENSOR_DF_EXCHANGE", "auto").strip().lower()
    if choice not in _DF_EXCHANGE_CHOICES:
        raise ValueError(
            f"TRNTENSOR_DF_EXCHANGE must be one of {_DF_EXCHANGE_CHOICES}; got {choice!r}"
        )
    if choice != "auto":
        return choice
    naux = B.shape[1]
    if naux == 0 or nocc == 0:  # nothing to tile (df_half_tiling divides by naux)
        return "aten"
    if nocc > 512 or df_half_tiling(naux, _round_up(nbasis, TILE_K), nocc, B.shape[0]) is None:
        return "aten"
    if _nki_forced():
        return "trn2"
    if ("dfk", tuple(B.shape), nocc) in _native_kernel_failed:
        return "aten"
    big = nocc * nbasis**2 * naux >= _DF_EXCHANGE_MIN_MACS
    return "trn2" if big and B.device.type == _NEURON_DEVICE else "aten"


def _df_k_banks() -> int:
    return int(os.environ.get("TRNTENSOR_DF_K_BANKS", "6"))


def _nki_df_exchange(B: torch.Tensor, C_occ: torch.Tensor) -> torch.Tensor:
    """K = Σ_{i,P} X X^T from packed ``(npair, naux)`` integrals on the native path (#60).

    Per group of aux chunks (``df_groups``), ``df_half_packed_kernel``
    gathers and half-transforms into ``Xt`` and ``df_exchange_syrk_kernel``
    adds that group into the two running partial Ks; ``K`` is their sum.
    nbasis is padded through ``C_occ`` and the gather indices, nocc to an
    even count with zero orbitals (the SYRK splits it across programs);
    naux is not padded: the chunks are ``tp`` wide (a divisor of naux or a
    128 multiple) and the last group's last chunk may be the short
    ``naux mod tp`` one.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    npair, naux = B.shape
    nbasis, nocc = C_occ.shape
    nbp = _round_up(nbasis, TILE_K)
    nop = nocc + nocc % 2
    tiling = df_half_tiling(naux, nbp, nop, npair)
    if tiling is None:
        raise ValueError(f"df_exchange: no kernel tiling for naux={naux}, nbasis={nbasis}")
    tp, nub, nvc = tiling
    tail = naux % tp
    if (nbp, nop) != (nbasis, nocc):
        co = torch.zeros(nbp, nop, dtype=C_occ.dtype, device=C_occ.device)
        co[:nbasis, :nocc] = C_occ
        C_occ = co
    C_occ = C_occ.contiguous()
    dev = B.device if B.device.type == _NEURON_DEVICE else None
    img = _df_gather_images(nbasis, nbp, naux, tp, nub, nvc, dev)
    npc = -(-naux // tp)
    grp = df_group_chunks(npc, nbp, tp, nub, nvc)
    w0 = math.gcd(naux, tp)
    eri_v = B.contiguous().view(npair * naux // w0, w0)
    kin = torch.zeros(2, nbp, nbp, dtype=torch.float32, device=B.device)
    banks = _df_k_banks()
    for c0, c1 in df_groups(naux, tp, grp):
        kw = {"nub": nub, "tp": tp, "tail": tail if c1 == npc else 0}
        xt = _launch(None, "df_half_packed", eri_v, img[c0:c1], C_occ, **kw)
        kin = _launch(None, "df_exchange_syrk", xt, kin, banks=banks)
        del xt
    K = kin[0] + kin[1]
    return K[:nbasis, :nbasis] if nbp != nbasis else K


# Under "auto" the J kernels need npair·naux of at least this many elements
# of B (each kernel streams B once). Trn2, LNC=2 (#60): 3.8e7 elements
# (n_ao 250) 2.90 ms vs 3.81 ms for two aten GEMVs; 2.2e9 (970) 46 vs 87 ms.
_DF_COULOMB_MIN_ELEMS = 1.0e7


def _df_coulomb_route(B: torch.Tensor) -> str:
    """'trn2' or 'aten' for :func:`trntensor.quantum.df_coulomb` on packed ``B``.

    'trn2' needs the native path, NKI in use and at least 256 packed rows;
    under ``"auto"`` also ``B`` on the device and ``npair·naux ≥
    _DF_COULOMB_MIN_ELEMS``. ``TRNTENSOR_DF_COULOMB=auto|trn2|aten`` overrides.
    """
    if not (_use_nki() and HAS_NKI) or not _on_native_path():
        return "aten"
    choice = os.environ.get("TRNTENSOR_DF_COULOMB", "auto").strip().lower()
    if choice not in _DF_EXCHANGE_CHOICES:
        raise ValueError(
            f"TRNTENSOR_DF_COULOMB must be one of {_DF_EXCHANGE_CHOICES}; got {choice!r}"
        )
    if choice != "auto":
        return choice
    if B.shape[0] < 2 * TILE_K:
        return "aten"
    if _nki_forced():
        return "trn2"
    if ("dfj", tuple(B.shape)) in _native_kernel_failed:
        return "aten"
    big = B.shape[0] * B.shape[1] >= _DF_COULOMB_MIN_ELEMS
    return "trn2" if big and B.device.type == _NEURON_DEVICE else "aten"


def _nki_df_coulomb(B: torch.Tensor, pw: torch.Tensor) -> torch.Tensor:
    """Packed ``J = B (Bᵀ pw)`` on the native path (#60): ``(npair,)``.

    ``df_coulomb_d_kernel`` and ``df_coulomb_j_kernel`` take the rows of
    ``2·(npair // 256)`` full 128-row tiles; the < 256 leftover rows are one
    small aten GEMV each way.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    npair, naux = B.shape
    out_dtype = B.dtype
    # The kernels take fp32. A float64 B went to them as is and
    # came back inf/garbage through 0.22.0; compute in fp32 and return B's
    # dtype, as the aten path does.
    if B.dtype != torch.float32:
        B = B.float()
    B = B.contiguous()
    pw = pw.reshape(npair, 1).to(B.dtype).contiguous()
    rows = (npair // (2 * TILE_K)) * 2 * TILE_K
    dpart = _launch(None, "df_coulomb_d", B, pw)
    d = dpart[0, 0] + dpart[1, 0]
    if rows < npair:
        d = d + B[rows:].T @ pw[rows:, 0]
    d_b = d.reshape(1, naux).expand(TILE_K, naux).contiguous()
    jt = _launch(None, "df_coulomb_j", B, d_b)
    j = jt.T.reshape(-1)
    if rows < npair:
        j = torch.cat([j, B[rows:] @ d])
    return j.to(out_dtype)


def nki_batched_matmul(A: torch.Tensor, B: torch.Tensor, use_sr: bool = False) -> torch.Tensor:
    """Batched 2D matmul ``A @ B`` over a leading batch dim.

    Routes through ``batched_matmul_kernel`` on Trainium. Pads M, K
    and N to tile multiples per slice; batch dim is passed through
    unchanged. Falls back to ``torch.bmm`` on any kernel failure
    unless ``TRNTENSOR_REQUIRE_NKI=1``.

    ``use_sr`` follows the same runtime rule as :func:`nki_matmul`: SR needs
    ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1`` at process start, else a one-time
    ``RuntimeWarning`` and round-to-nearest-even.
    """
    if not (_use_nki() and HAS_NKI):
        return _neuron_bmm(A, B)

    Bsz, M, K = A.shape
    _, _, N = B.shape
    route = _route_for(Bsz * M * K * N, A, B)
    if route == "aten":
        return _aten_on_device(_neuron_bmm, A, B)
    if route != "nki" or _neuron_mm_unsafe(A, B):
        return _neuron_bmm(A, B)

    if not _matmul_kernel_dtypes_ok(A, B):
        if A.dtype == B.dtype == torch.float64 and _fp64_to_kernel("batched_matmul", A, B):
            return nki_batched_matmul(A.float(), B.float(), use_sr=use_sr).to(torch.float64)
        _no_kernel_for_dtype("batched_matmul", _MATMUL_DTYPES, A, B)
        return torch.bmm(A, B)

    if _on_native_path():
        _warn_native_kernel_deprecated("batched_matmul")

    from ._kernels import batched_matmul_kernel

    M_pad = _round_up(M, TILE_M)
    K_pad = _round_up(K, TILE_K)
    N_pad = N if N <= TILE_N else _round_up(N, TILE_N)
    needs_pad = (M_pad != M) or (K_pad != K) or (N_pad != N)

    if use_sr and not _use_simulator():
        _warn_if_sr_unavailable()

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
            out_np = _simulate(
                batched_matmul_kernel, "batched_matmul", A_feed, B_feed, use_sr=use_sr
            )
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            result = _launch(batched_matmul_kernel, "batched_matmul", A_feed, B_feed, use_sr=use_sr)
        return result[:, :M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.bmm(A, B)
