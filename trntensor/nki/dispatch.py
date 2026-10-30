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
# Native path (TorchNeuron, Trn2/Trn3). An eager launch of a registered
# ``trntensor::`` op costs ~0.2 ms, well under the XLA path's per-dispatch
# floor, so the fixed-overhead part of the crossover shrinks. Host↔device
# transfer costs the same on both paths, so the unpinned crossover moves
# less than the pinned one. These defaults are estimates from those two
# numbers, not measurements; re-measure on Trn2 with
# scripts/autotune_dispatch.py (which writes the *_native cache keys there):
#   _MIN_NKI_FLOPS_NATIVE        — operands on CPU (default: 1 GFLOPs)
#   _MIN_NKI_FLOPS_NATIVE_PINNED — operands already on "neuron" (default: 128 MFLOPs)
#
# _min_nki_flops(*operands) returns the threshold for the active path.
# Override with env vars or scripts/autotune_dispatch.py --write-cache.
_DEFAULT_MIN_NKI_FLOPS = 2_000_000_000
_DEFAULT_MIN_NKI_FLOPS_PINNED = 1_000_000_000
_DEFAULT_MIN_NKI_FLOPS_NATIVE = 1_000_000_000
_DEFAULT_MIN_NKI_FLOPS_NATIVE_PINNED = 128_000_000


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


def _threshold(env_var: str, cache: dict, cache_key: str, default: int) -> int:
    env = os.environ.get(env_var, "")
    if env:
        return int(env)
    try:
        cached = cache.get(cache_key)
        return int(cached) if cached else default
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
    # Use a lower threshold when operands are already on the device — no
    # host→device transfer cost. Path-aware: see _min_nki_flops.
    flop_threshold = _min_nki_flops(A, B)
    if flop_threshold > M * K * N:
        result = torch.matmul(A, B)
        return _stochastic_round_cpu(result, A.dtype) if use_sr else result

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
    flop_threshold = _min_nki_flops(A, B)
    if Bsz * M * K * N < flop_threshold:
        return torch.bmm(A, B)

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
