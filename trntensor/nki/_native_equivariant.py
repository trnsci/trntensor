"""
Native TorchNeuron registration of the equivariant-contraction kernels.

Same lineage and conventions as ``_native.py`` (Neuron SDK 2.33+,
torch-neuronx 2.14, ``torch_neuronx.nki_op``), kept in its own module because
these are the first trntensor ops with a registered backward:

- ``trntensor::conv_tp``      out = conv_tp(x, y, w)        backward: ``trntensor::conv_tp_bwd``
- ``trntensor::conv_tp_bwd``  [∂x | ∂y | ∂w] in one tensor  (no gradient of its own)
- ``trntensor::symcon``       m-major symmetric contraction  backward: ``trntensor::symcon_bwd``
- ``trntensor::symcon_bwd``   ∂x                             (no gradient of its own)

The backward is attached with ``register_autograd`` on the ``CustomOpDef``
that ``nki_op`` returns (the trnsparse ``spmm_dense`` pattern), so it also
holds under ``torch.compile``. The constant operands (CG blocks, selections,
coupling and reduction matrices, per-node weight rows) get no gradient: they
are constants of the spec. The ``*_bwd`` ops are not differentiable
themselves; when autograd records the backward (``create_graph=True``,
training through forces) the backward formulas switch to the differentiable
aten statements in ``_equivariant.conv_tp_bwd_torch`` / ``symcon_bwd_torch``,
so double backward works, with first-order backward (forces) on the kernels.

Loaded by ``_equivariant._native_ops`` only after the native path is
selected. No ``from __future__ import annotations``: ``nki_op`` builds the op
schema from the live annotations. Kernels are looked up on
``_equivariant_kernels`` at call time, so registration traces nothing.
"""

import torch
from torch_neuronx import nki_op

from . import _equivariant as _eqh
from . import _equivariant_kernels as _ek


def _conv_tp_fake(x, y, w, cg, cgT, plan, zero_fill=0):
    return x.new_empty((x.shape[0], plan[0]))


@nki_op("trntensor::conv_tp", mutates_args={}, fake_fn=_conv_tp_fake)
def conv_tp(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    cg: torch.Tensor,
    cgT: torch.Tensor,
    plan: list[int],
    zero_fill: int = 0,
) -> torch.Tensor:
    return _ek.conv_tp_fwd_kernel(x, y, w, cg, plan=tuple(plan), zero_fill=zero_fill)


def _conv_tp_bwd_fake(g, x, y, w, cg, cgT, plan, need_dx=1, need_dy=1, need_dw=1):
    return x.new_empty((x.shape[0], x.shape[1] + y.shape[1] + w.shape[1]))


@nki_op("trntensor::conv_tp_bwd", mutates_args={}, fake_fn=_conv_tp_bwd_fake)
def conv_tp_bwd(
    g: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    cg: torch.Tensor,
    cgT: torch.Tensor,
    plan: list[int],
    need_dx: int = 1,
    need_dy: int = 1,
    need_dw: int = 1,
) -> torch.Tensor:
    return _ek.conv_tp_bwd_kernel(
        g, x, y, w, cg, cgT, plan=tuple(plan), need_dx=need_dx, need_dy=need_dy, need_dw=need_dw
    )


def _symcon_fake(x, wr, sel, selT, usym, usymT, s, sT):
    return x.new_empty((s.shape[1], x.shape[0] * x.shape[1]))


@nki_op("trntensor::symcon", mutates_args={}, fake_fn=_symcon_fake)
def symcon(
    x: torch.Tensor,
    wr: torch.Tensor,
    sel: torch.Tensor,
    selT: torch.Tensor,
    usym: torch.Tensor,
    usymT: torch.Tensor,
    s: torch.Tensor,
    sT: torch.Tensor,
) -> torch.Tensor:
    return _ek.symcon_fwd_kernel(x, wr, sel, usym, s)


def _symcon_bwd_fake(g, x, wr, sel, selT, usymT, sT):
    return x.new_empty(x.shape)


@nki_op("trntensor::symcon_bwd", mutates_args={}, fake_fn=_symcon_bwd_fake)
def symcon_bwd(
    g: torch.Tensor,
    x: torch.Tensor,
    wr: torch.Tensor,
    sel: torch.Tensor,
    selT: torch.Tensor,
    usymT: torch.Tensor,
    sT: torch.Tensor,
) -> torch.Tensor:
    return _ek.symcon_bwd_kernel(g, x, wr, sel, selT, usymT, sT)


# ── Autograd (register_autograd) ─────────────────────────────────────────────
#
# Module-level so the CPU tests can drive them through stand-in ops.


def split_conv_tp_grads(grads, x, y, w):
    """Views ``(∂x, ∂y, ∂w)`` of the backward op's concatenated output."""
    d1, d2 = x.shape[1], y.shape[1]
    return grads[:, :d1], grads[:, d1 : d1 + d2], grads[:, d1 + d2 : d1 + d2 + w.shape[1]]


def _conv_tp_setup_context(ctx, inputs, output):
    x, y, w, cg, cgT, plan, _zero_fill = inputs
    ctx.save_for_backward(x, y, w, cg, cgT)
    ctx.plan = list(plan)


def _conv_tp_backward(ctx, grad):
    x, y, w, cg, cgT = ctx.saved_tensors
    nx, ny, nw = ctx.needs_input_grad[:3]
    if not (nx or ny or nw):
        return (None,) * 7
    if torch.is_grad_enabled():
        # create_graph=True (training through forces): the *_bwd kernel has no
        # gradient, so build this backward from differentiable aten ops.
        dx, dy, dw = _eqh.conv_tp_bwd_torch(grad, x, y, w, cg, ctx.plan, nx, ny, nw)
        return (dx, dy, dw, None, None, None, None)
    grads = OPS["conv_tp_bwd"](
        grad.contiguous(), x, y, w, cg, cgT, ctx.plan, int(nx), int(ny), int(nw)
    )
    dx, dy, dw = split_conv_tp_grads(grads, x, y, w)
    return (dx if nx else None, dy if ny else None, dw if nw else None, None, None, None, None)


def _symcon_setup_context(ctx, inputs, output):
    x, wr, sel, selT, _usym, usymT, _s, sT = inputs
    ctx.save_for_backward(x, wr, sel, selT, usymT, sT)


def _symcon_backward(ctx, grad):
    if not ctx.needs_input_grad[0]:
        return (None,) * 8
    x, wr, sel, selT, usymT, sT = ctx.saved_tensors
    if torch.is_grad_enabled():  # create_graph=True: see _conv_tp_backward
        return (_eqh.symcon_bwd_torch(grad, x, wr, sel, usymT, sT),) + (None,) * 7
    dx = OPS["symcon_bwd"](grad.contiguous(), x, wr, sel, selT, usymT, sT)
    return (dx,) + (None,) * 7


OPS = {
    "conv_tp": conv_tp,
    "conv_tp_bwd": conv_tp_bwd,
    "symcon": symcon,
    "symcon_bwd": symcon_bwd,
}

conv_tp.register_autograd(_conv_tp_backward, setup_context=_conv_tp_setup_context)
symcon.register_autograd(_symcon_backward, setup_context=_symcon_setup_context)
