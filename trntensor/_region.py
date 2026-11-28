"""Compiled regions for the fused PyTorch paths (trnsci/trnsci#178).

The MLIP edge and node ops (``spherical_harmonics``, ``edge_chain``,
``equivariant_linear``) each have a forward and an analytic backward written
as a short chain of aten ops. Run eagerly on a device, every op is a launch.
Under a *region backend* the forward and the backward bodies each run as one
``torch.compile(..., dynamic=False)`` graph instead: one compiled graph per
direction per call site, with the analytic backward still registered through
``torch.autograd.Function`` (forces need the input gradients).

The backend is off by default. ``set_region_backend("neuron")`` (or
``TRNTENSOR_REGION_BACKEND=neuron``) turns it on for the TorchNeuron device;
on a CPU host ``"aot_eager"`` or ``"inductor"`` exercise the same tracing.
The NKI kernel route, where an op has one, takes precedence. Shapes are
static per compiled graph, so pad edge counts to a bucket (as the exemplar
does) to avoid a recompile per geometry.

When autograd records the backward (``create_graph=True``) the backward body
runs eagerly, so double backward differentiates the analytic formulas.
"""

from __future__ import annotations

import os
import threading
import types

import torch

_ENV = "TRNTENSOR_REGION_BACKEND"
_backend: str | None = os.environ.get(_ENV) or None
_compiled: dict = {}
_lock = threading.Lock()


def set_region_backend(name: str | None) -> None:
    """``torch.compile`` backend for the fused op bodies, or ``None`` (eager)."""
    global _backend
    _backend = name or None


def get_region_backend() -> str | None:
    return _backend


def _site_copy(fn, tag: int):
    """``fn`` with its own code object: TorchDynamo keeps compiled graphs (and
    counts recompiles) per code object, so each call site gets its own cache."""
    name = f"{fn.__name__}__site{tag}"
    kw = {"co_name": name}
    if hasattr(fn.__code__, "co_qualname"):  # Python >= 3.11
        kw["co_qualname"] = name
    code = fn.__code__.replace(**kw)
    f = types.FunctionType(code, fn.__globals__, name, fn.__defaults__, fn.__closure__)
    f.__kwdefaults__ = fn.__kwdefaults__
    return f


def region(fn, site=None):
    """``fn`` as one compiled graph under the region backend, else ``fn`` itself.

    ``site`` (hashable) separates call sites that specialise differently (a
    spec per layer, say), so each keeps its own graphs instead of
    recompiling one shared entry. Not used while autograd records
    (``torch.is_grad_enabled()`` inside a backward means ``create_graph=True``):
    the eager body is differentiable.
    """
    name = _backend
    if name is None or torch.is_grad_enabled():
        return fn
    key = (fn, site, name)
    with _lock:
        c = _compiled.get(key)
        if c is None:
            c = torch.compile(
                _site_copy(fn, len(_compiled)), backend=name, dynamic=False, fullgraph=True
            )
            if name in _DEVICE_BACKENDS:
                c = _on_device_only(fn, c, _DEVICE_BACKENDS[name])
            _compiled[key] = c
    return c


# Backends that compile for one device type. On TorchNeuron, a graph compiled
# with ``backend="neuron"`` and called with host tensors runs on the device and
# returns device tensors (seen on Trn2, window 8: a host fp64 reference under
# ``set_region_backend("neuron")`` came back on ``neuron:0``). A call with no
# tensor on that device runs the eager body instead; any call that touches the
# device still takes the compiled graph, so a device step never silently
# drops to eager.
_DEVICE_BACKENDS = {"neuron": "neuron"}


def _on_device_only(fn, compiled, device_type: str):
    from torch.utils._pytree import tree_leaves

    def call(*args, **kwargs):
        ts = [t for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
        if any(t.device.type == device_type for t in ts):
            return compiled(*args, **kwargs)
        return fn(*args, **kwargs)

    return call


def clear_region_cache() -> None:
    with _lock:
        _compiled.clear()
