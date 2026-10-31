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

import importlib.util
import os
import warnings

import numpy as np
import torch

from ._kernels import (
    HAS_NKI,
    TILE_K,
    TILE_M,
    TILE_N,
    ao_to_mo_trn2_nub,
    ao_to_mo_trn2_tiling,
)

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
    as "aten" below it.
    """
    if not _on_native_path():
        return "nki" if flops >= _nki_flop_threshold(pinned) else "host"
    nki_ok = flops >= _MIN_NKI_FLOPS_NATIVE_PINNED
    if pinned:
        return "nki" if nki_ok else "host"
    if flops < _MIN_NKI_FLOPS_NATIVE:
        return "host"
    return "nki" if nki_ok else "aten"


def _route_for(flops: int, *operands: torch.Tensor) -> str:
    """:func:`_route` for these operands' residency."""
    device_type = _NEURON_DEVICE if _on_native_path() else "xla"
    return _route(flops, pinned=all(t.device.type == device_type for t in operands))


def _aten_on_device(fn, *tensors: torch.Tensor) -> torch.Tensor:
    """Run ``fn`` (an aten op) on the ``neuron`` device; result on the caller's device."""
    orig = tensors[0].device
    return fn(*(t.to(_NEURON_DEVICE) for t in tensors)).to(orig)


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


def _get_native_ops() -> dict:
    """Import `_native` (registering the trntensor:: custom ops) on first use."""
    global _native_ops
    if _native_ops is None:
        from . import _native

        _native_ops = _native.OPS
    return _native_ops


def _launch(kernel, op_name: str, *tensors: torch.Tensor, **consts) -> torch.Tensor:
    """Run one kernel launch on the Neuron device; return it on the caller's device.

    ``op_name`` is the ``trntensor::<op_name>`` custom op used on the native
    path; the XLA path calls ``kernel`` directly on torch_xla tensors.
    ``consts`` are compile-time kernel arguments (``use_sr``), passed by
    keyword on both paths. The result comes back on the device of
    ``tensors[0]``, so device-resident operands keep the result on the device.
    """
    if _get_dispatch_path() == "native":
        orig = tensors[0].device
        op = _get_native_ops()[op_name]
        return op(*(t.to(_NEURON_DEVICE) for t in tensors), **consts).to(orig)
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
    return tensor.to("cpu")


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
        result = torch.matmul(A, B)
        return _stochastic_round_cpu(result, A.dtype) if use_sr else result

    M, K = A.shape
    _, N = B.shape
    # Residency- and path-aware executor choice: see _route.
    route = _route_for(M * K * N, A, B)
    if route != "nki":
        if use_sr and (route == "aten" or _is_device_tensor(A)):
            _warn_if_sr_unavailable()
        if route == "aten":
            return _aten_on_device(torch.matmul, A, B)
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
            out_np = nki.simulate(matmul_kernel)(
                A_feed.cpu().numpy(), B_feed.cpu().numpy(), use_sr=use_sr
            )
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
        out_np = nki.simulate(mp2_energy_kernel)(
            B_feed.cpu().numpy(),
            eo_b.cpu().numpy(),
            ev_feed.cpu().numpy(),
            ev_b.cpu().numpy(),
        )
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
    SBUF. Host ``B`` under ``"auto"`` stays on aten on the
    host.
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
    op = _get_native_ops()["mp2_energy_row"]
    global _gc_may_free
    _gc_may_free = True
    acc = None
    for i in range(nocc):
        p = op(
            Bt[i],
            Bx[i : i + nd_pad],
            eo[:, i : i + 1].contiguous(),
            ex[:, i : i + nd_pad].contiguous(),
            evc,
            evr,
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
        out_np = nki.simulate(ao_to_mo_transform_kernel)(
            eri_feed.cpu().numpy(),
            C_occ_feed.cpu().numpy(),
            C_vir_feed.cpu().numpy(),
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
# one stage-1 eri block, one X[i] strip); same budget as the MP2 kernel.
_AO_TO_MO_SBUF_BUDGET = 144 * 1024
# Under "auto" the kernel needs stage-1 work nocc·nbasis²·naux of at least
# this many multiply-adds. Below it, the custom-op launch floor and the
# padding copies lose to aten: nbasis=128, nocc=16, naux=256 (6.7e7) measured
# 0.57 ms against 0.49 ms; nbasis=256, nocc=32, naux=512 (1.1e9) 0.92 ms
# against 1.23 ms (#51).
_AO_TO_MO_MIN_MACS = 5.0e8


def _ao_to_mo_trn2_sbuf_bytes(
    nbasis: int, nocc: int, nvir: int, naux: int, itemsize: int, packed: bool = False
) -> int:
    nkb = _round_up(nbasis, TILE_K) // TILE_K
    tp, nub = ao_to_mo_trn2_tiling(_round_up(naux, TILE_K), packed)
    cols = nkb * (nocc + _round_up(nvir, TILE_M) + nub * tp + tp) + nub * tp
    # The packed kernel also keeps one chunk's gather indices (uint32).
    idx = 4 * nkb * nkb * TILE_K if packed else 0
    return itemsize * cols + idx


def _ao_to_mo_route(eri: torch.Tensor, nocc: int, nvir: int, nbasis: int | None = None) -> str:
    """'trn2', 'legacy' or 'aten' for :func:`trntensor.quantum.ao_to_mo_transform`.

    XLA path / simulator: 'legacy' whenever NKI is in use (unchanged).
    Native path: the Trn2 kernel for device-resident ``eri`` when
    ``nocc ≤ 128``, its SBUF working set fits and the stage-1 work reaches
    ``_AO_TO_MO_MIN_MACS``, else aten; ``set_backend("nki")`` drops the size
    and residency conditions. Host ``eri``
    under ``"auto"`` stays on aten on the host.

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
        nocc <= 128
        and _ao_to_mo_trn2_sbuf_bytes(nb, nocc, nvir, naux, eri.element_size(), packed)
        <= _AO_TO_MO_SBUF_BUDGET
    )
    if not fits:
        return "aten"
    if _nki_forced():
        return "trn2"
    if ("ao2mo", tuple(eri.shape), nocc, nvir) in _native_kernel_failed:
        return "aten"
    big = nocc * nb**2 * naux >= _AO_TO_MO_MIN_MACS
    return "trn2" if big and eri.device.type == _NEURON_DEVICE else "aten"


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
    out = _launch(None, "ao_to_mo_trn2", eri.contiguous(), C_occ.contiguous(), C_vir.contiguous())
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
    tp = _packed_tp(nxp)
    dev = eri_p.device if eri_p.device.type == _NEURON_DEVICE else None
    img = _tril_gather_images(nbasis, nbp, nxp, tp, dev)
    out = _launch(
        None,
        "ao_to_mo_trn2_packed",
        eri_p.contiguous().view(npair * (nxp // tp), tp),
        img,
        C_occ.contiguous(),
        C_vir.contiguous(),
    )
    return out[:, :nvir, :naux] if (nvp, nxp) != (nvir, naux) else out


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
        return torch.bmm(A, B)

    Bsz, M, K = A.shape
    _, _, N = B.shape
    route = _route_for(Bsz * M * K * N, A, B)
    if route == "aten":
        return _aten_on_device(torch.bmm, A, B)
    if route != "nki":
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
            out_np = nki.simulate(batched_matmul_kernel)(
                A_feed.cpu().numpy(), B_feed.cpu().numpy(), use_sr=use_sr
            )
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            result = _launch(batched_matmul_kernel, "batched_matmul", A_feed, B_feed, use_sr=use_sr)
        return result[:, :M, :N] if needs_pad else result
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.bmm(A, B)
