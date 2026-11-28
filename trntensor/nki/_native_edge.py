"""
Native TorchNeuron registration of the per-edge MLIP kernels (trnsci/trnsci#178).

Same pattern as ``_native_equivariant.py``:

- ``trntensor::sh``              spherical harmonics      backward: ``trntensor::sh_bwd``
- ``trntensor::sh_bwd``          ∂v                       (no gradient of its own)
- ``trntensor::edge_chain``      every head, side by side backward: ``trntensor::edge_chain_bwd``
- ``trntensor::edge_chain_bwd``  ∂r                       (no gradient of its own)

Operands are the kernels' partition-major 2-D views (``[128, M·F]``); the
static tables are ``list[int]`` / ``list[float]`` arguments. The backward is
attached with ``register_autograd``; under ``create_graph=True`` it switches
to the differentiable statements in ``_edge`` (double backward), with
first-order backward on the kernels. The packed weights get no gradient.

Loaded by ``_edge._native_ops`` only after the native path is selected. No
``from __future__ import annotations``: ``nki_op`` builds the op schema from
the live annotations.
"""

import torch
from torch_neuronx import nki_op

from . import _edge as _eh
from . import _edge_kernels as _xk


def _sh_fake(v2, lmax, normalize, eps, table, dtable):
    return v2.new_empty((v2.shape[0], (v2.shape[1] // 3) * (lmax + 1) * (lmax + 1)))


@nki_op("trntensor::sh", mutates_args={}, fake_fn=_sh_fake)
def sh(
    v2: torch.Tensor,
    lmax: int,
    normalize: int,
    eps: float,
    table: list[float],
    dtable: list[float],
) -> torch.Tensor:
    return _xk.sh_fwd_kernel(v2, lmax=lmax, normalize=normalize, eps=eps, table=tuple(table))


def _sh_bwd_fake(g2, v2, lmax, normalize, eps, dtable):
    return v2.new_empty(v2.shape)


@nki_op("trntensor::sh_bwd", mutates_args={}, fake_fn=_sh_bwd_fake)
def sh_bwd(
    g2: torch.Tensor,
    v2: torch.Tensor,
    lmax: int,
    normalize: int,
    eps: float,
    dtable: list[float],
) -> torch.Tensor:
    return _xk.sh_bwd_kernel(g2, v2, lmax=lmax, normalize=normalize, eps=eps, dtable=tuple(dtable))


def _edge_chain_fake(r2, r02, wpack, wtpack, dims, consts, bessel):
    _offs, DO = _xk.edge_out_offsets(dims)
    return r2.new_empty((r2.shape[0], r2.shape[1] * DO))


@nki_op("trntensor::edge_chain", mutates_args={}, fake_fn=_edge_chain_fake)
def edge_chain(
    r2: torch.Tensor,
    r02: torch.Tensor,
    wpack: torch.Tensor,
    wtpack: torch.Tensor,
    dims: list[int],
    consts: list[float],
    bessel: list[float],
) -> torch.Tensor:
    return _xk.edge_chain_fwd_kernel(
        r2, r02, wpack, dims=tuple(dims), consts=tuple(consts), bessel=tuple(bessel)
    )


def _edge_chain_bwd_fake(g2, r2, r02, wpack, wtpack, dims, consts, bessel):
    return r2.new_empty(r2.shape)


@nki_op("trntensor::edge_chain_bwd", mutates_args={}, fake_fn=_edge_chain_bwd_fake)
def edge_chain_bwd(
    g2: torch.Tensor,
    r2: torch.Tensor,
    r02: torch.Tensor,
    wpack: torch.Tensor,
    wtpack: torch.Tensor,
    dims: list[int],
    consts: list[float],
    bessel: list[float],
) -> torch.Tensor:
    return _xk.edge_chain_bwd_kernel(
        g2, r2, r02, wpack, wtpack, dims=tuple(dims), consts=tuple(consts), bessel=tuple(bessel)
    )


# ── Autograd (register_autograd) ─────────────────────────────────────────────


def _sh_setup_context(ctx, inputs, output):
    v2, lmax, normalize, eps, _table, dtable = inputs
    ctx.save_for_backward(v2)
    ctx.args = (lmax, normalize, eps, list(dtable))


def _sh_backward(ctx, grad):
    if not ctx.needs_input_grad[0]:
        return (None,) * 6
    (v2,) = ctx.saved_tensors
    lmax, normalize, eps, dtable = ctx.args
    if torch.is_grad_enabled():  # create_graph=True: differentiable aten
        return (_eh.sh_bwd_torch(grad, v2, lmax, normalize, eps, dtable),) + (None,) * 5
    dv = OPS["sh_bwd"](grad.contiguous(), v2, lmax, normalize, eps, dtable)
    return (dv,) + (None,) * 5


def _edge_chain_setup_context(ctx, inputs, output):
    r2, r02, wpack, wtpack, dims, consts, bessel = inputs
    ctx.save_for_backward(r2, r02, wpack, wtpack)
    ctx.args = (list(dims), list(consts), list(bessel))


def _edge_chain_backward(ctx, grad):
    if not ctx.needs_input_grad[0]:
        return (None,) * 7
    r2, r02, wpack, wtpack = ctx.saved_tensors
    dims, consts, bessel = ctx.args
    if torch.is_grad_enabled():  # create_graph=True: differentiable aten
        return (_eh.edge_chain_bwd_torch(grad, r2, r02, wpack, dims, consts, bessel),) + (None,) * 6
    dr = OPS["edge_chain_bwd"](grad.contiguous(), r2, r02, wpack, wtpack, dims, consts, bessel)
    return (dr,) + (None,) * 6


OPS = {
    "sh": sh,
    "sh_bwd": sh_bwd,
    "edge_chain": edge_chain,
    "edge_chain_bwd": edge_chain_bwd,
}

sh.register_autograd(_sh_backward, setup_context=_sh_setup_context)
edge_chain.register_autograd(_edge_chain_backward, setup_context=_edge_chain_setup_context)
