"""Host side of the equivariant-contraction kernels (trnsci/trntensor#138).

Routing, padding and autograd for ``trntensor.conv_tp`` and
``trntensor.symmetric_contraction`` on the NKI kernels in
``_equivariant_kernels.py``.

Autograd follows the suite pattern (trnsparse ``spmm``):

- native TorchNeuron path: ``trntensor::conv_tp`` and ``trntensor::symcon`` are
  ``nki_op`` custom ops whose backward is attached with ``register_autograd``
  (``_native.py``); the backward runs the ``*_bwd`` kernel op. Gradients reach
  ``x``, ``y`` and ``w`` (conv_tp) and ``x`` (symmetric contraction).
- XLA path and the CPU simulator: ``torch.autograd.Function`` wrappers here
  run the same kernels forward and backward.

Shapes: the edge count is padded to :func:`edge_bucket` (a multiple of 256,
coarse enough that an MD trajectory's small changes in E reuse one compiled
kernel) and the node count to a multiple of ``1024 / C``; padded rows are
zero and are sliced off. Only shapes are static: edge and species indices are
never compiled into a kernel.
"""

from __future__ import annotations

import numpy as np
import torch

from . import _equivariant_kernels as _ek
from . import dispatch as _d
from ._equivariant_kernels import (
    CONV_TP_TILE,
    EQUIVARIANT_SBUF_BUDGET,
    SYMCON_ROWS,
    conv_tp_sbuf_bytes,
    symcon_psum_banks,
)


def edge_bucket(E: int) -> int:
    """Padded edge count for the conv_tp kernels.

    A multiple of 256 (two programs of 128-edge tiles at LNC=2). Above 2048
    edges the step grows with E, ``2^(⌊log2 E⌋ − 3)``, so padding stays under
    12.5% while one compiled shape covers a range of E: 10,392 and 10,378
    (two geometries of the 216-atom slab) both run at 11,264, and 96,278 at
    98,304.
    """
    E = max(int(E), 1)
    step = 256
    if E > 2048:
        step = max(step, 1 << (E.bit_length() - 1 - 3))
    return -(-E // step) * step


def node_pad(N: int, C: int) -> int:
    """Padded node count for the symmetric-contraction kernels (N·C a multiple of 1024)."""
    per = max(1, (2 * SYMCON_ROWS) // C)
    return -(-max(int(N), 1) // per) * per


# ── routing ─────────────────────────────────────────────────────────────────


def _kernel_dtype_ok(*tensors) -> bool:
    return all(t.dtype in (torch.float32, torch.bfloat16, torch.float16) for t in tensors)


def _conv_tp_fits(spec) -> bool:
    lay = spec.layout
    return (
        lay.D2 <= CONV_TP_TILE
        and conv_tp_sbuf_bytes(lay.D1, lay.D2, lay.P * lay.U, lay.Dout, lay.Q, lay.U)
        <= EQUIVARIANT_SBUF_BUDGET
    )


def _symcon_fits(spec) -> bool:
    lay = spec.layout
    C = lay.C
    return (
        C <= 128
        and SYMCON_ROWS % C == 0
        and lay.L + 1 <= 128
        and lay.Dsum <= 128
        and symcon_psum_banks(lay.MK) <= 8
    )


def _route(what: str, fits: bool, tensors) -> bool:
    """True when the kernel should run (see ``trntensor.equivariant``)."""
    from ..equivariant import _auto_uses_kernels

    if not (_d._use_nki() and _d.HAS_NKI):
        return False
    forced = _d._nki_forced()
    if not forced:
        if not (_auto_uses_kernels() and _d._on_native_path()):
            return False
        if not all(_d._on_neuron_device(t) and t.dtype == torch.float32 for t in tensors):
            return False
    if not fits:
        if _d._require_nki_now():
            raise NotImplementedError(
                f"trntensor {what}: this spec exceeds the NKI kernel's SBUF/PSUM tiling; "
                "unset TRNTENSOR_REQUIRE_NKI to run the PyTorch path"
            )
        return False
    if not all(t.dtype.is_floating_point for t in tensors):
        return False
    if _d._has_fp64(*tensors):
        return _d._fp64_to_kernel(what, *tensors)
    return _kernel_dtype_ok(*tensors)


def conv_tp_kernel_route(x, y, w, spec) -> bool:
    if x.shape[0] == 0:
        return False
    return _route("conv_tp", _conv_tp_fits(spec), (x, y, w))


def symcon_kernel_route(x, species, spec) -> bool:
    if x.shape[0] == 0:
        return False
    return _route("symmetric_contraction", _symcon_fits(spec), (x,))


# ── launching ───────────────────────────────────────────────────────────────

_F32 = frozenset({torch.float32})
# Allowed dtypes per tensor operand of each kernel launch (the role of
# ``dispatch.KERNEL_OPERAND_DTYPES`` for these kernels; the host casts to fp32).
EQUIVARIANT_OPERAND_DTYPES: dict[str, tuple[frozenset, ...]] = {
    "conv_tp": (_F32,) * 4,  # x, y, w, cg
    "conv_tp_bwd": (_F32,) * 6,  # g, x, y, w, cg, cgT
    "symcon": (_F32,) * 5,  # x, wr, sel, usym, s
    "symcon_bwd": (_F32,) * 7,  # g, x, wr, sel, selT, usymT, sT
}


def check_operands(op_name: str, tensors) -> None:
    """``TypeError`` unless every operand dtype is one ``op_name`` takes."""
    allowed = EQUIVARIANT_OPERAND_DTYPES.get(op_name)
    if allowed is None:
        raise TypeError(f"trntensor kernel {op_name!r} has no operand dtype entry")
    dts = [t.dtype for t in tensors]
    if len(dts) != len(allowed) or not all(d in a for d, a in zip(dts, allowed, strict=True)):
        raise TypeError(
            f"trntensor kernel {op_name!r} takes {len(allowed)} float32 operands; got "
            f"({', '.join(str(d) for d in dts)}). This is a trntensor bug: the host "
            "wrapper should have cast the operands."
        )


_native_ops_table = None


def _native_ops() -> dict:
    """Import ``_native_equivariant`` (registering the ops) on first use."""
    global _native_ops_table
    if _native_ops_table is None:
        from . import _native_equivariant

        missing = sorted(set(_native_equivariant.OPS) - set(EQUIVARIANT_OPERAND_DTYPES))
        if missing:
            raise TypeError(f"equivariant native ops without an operand dtype entry: {missing}")
        _native_ops_table = _native_equivariant.OPS
    return _native_ops_table


def _native() -> bool:
    return _d.HAS_NKI and not _d._use_simulator() and _d._get_dispatch_path() == "native"


def _run(kernel, op_name: str, tensors, consts) -> tuple:
    """One kernel launch (simulator or XLA); outputs on ``tensors[0]``'s device."""
    check_operands(op_name, tensors)
    dev = tensors[0].device
    if _d._use_simulator():
        out = _d.nki.simulate(kernel)(*(t.detach().cpu().numpy() for t in tensors), **consts)
        outs = out if isinstance(out, (tuple, list)) else (out,)
        return tuple(torch.from_numpy(np.asarray(o)).to(dev) for o in outs)
    xs, orig = _d._to_xla(*tensors)
    out = kernel(*xs, **consts)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    return tuple(o.to(orig) for o in outs)


# ── differentiable statements of the backward kernels ─────────────────────
#
# The kernel backward is first order: the ``*_bwd`` kernels have no gradient of
# their own. When autograd builds a graph through the backward
# (``create_graph=True``, e.g. training on forces), the backward formulas
# below compute the same quantities from the same padded fp32 operands with
# differentiable aten ops, so double backward works on every path. First-order
# backward (forces in MD) keeps the kernel.


def conv_tp_bwd_torch(g, x, y, w, cg, plan, need_dx=True, need_dy=True, need_dw=True):
    """``(∂x, ∂y, ∂w)`` of the conv_tp kernel contract (differentiable aten)."""
    from ._equivariant_kernels import conv_tp_plan_paths

    E, D1 = x.shape
    paths = conv_tp_plan_paths(tuple(plan))
    U = w.shape[1] // len(paths)
    G = y @ cg
    dx_parts, dw_parts, dG_parts = {}, [], []
    for p, (x_off, d1, q_off, d3, o_off, _acc) in enumerate(paths):
        xv = x[:, x_off : x_off + U * d1].reshape(E, U, d1)
        gv = g[:, o_off : o_off + U * d3].reshape(E, U, d3)
        wp = w[:, p * U : (p + 1) * U]
        Gp = G[:, q_off : q_off + d1 * d3].reshape(E, d1, d3)
        h = torch.einsum("euk,eik->eui", gv, Gp)
        dw_parts.append((xv * h).sum(-1))
        part = (wp[:, :, None] * h).reshape(E, U * d1)
        dx_parts[x_off] = part if x_off not in dx_parts else dx_parts[x_off] + part
        dG_parts.append(torch.einsum("eui,euk->eik", xv * wp[:, :, None], gv).reshape(E, d1 * d3))
    dx = None
    if need_dx:
        cols, at = [], 0
        for x_off in sorted(dx_parts):
            if x_off > at:
                cols.append(x.new_zeros(E, x_off - at))
            cols.append(dx_parts[x_off])
            at = x_off + dx_parts[x_off].shape[1]
        if at < D1:
            cols.append(x.new_zeros(E, D1 - at))
        dx = torch.cat(cols, 1)
    dy = torch.cat(dG_parts, 1) @ cg.T if need_dy else None
    dw = torch.cat(dw_parts, 1) if need_dw else None
    return dx, dy, dw


def symcon_bwd_torch(g, x, wr, sel, usymT, sT):
    """``∂x`` of the symmetric-contraction kernel contract (differentiable aten).

    ``g [D, N·C]`` is the m-major cotangent of ``symcon_fwd_kernel``.
    """
    N, C, L = x.shape
    xa = torch.cat([x.reshape(N * C, L), x.new_ones(N * C, 1)], 1)
    X0, X1, X2 = (xa @ sel[:, j, :] for j in range(3))
    wrows = wr.permute(0, 2, 1).reshape(N * C, -1)
    R = ((g.T @ sT) * wrows) @ usymT
    dxa = (R * X1 * X2) @ sel[:, 0, :].T + (R * X0 * X2) @ sel[:, 1, :].T
    dxa = dxa + (R * X0 * X1) @ sel[:, 2, :].T
    return dxa[:, :L].reshape(N, C, L)


class _PadRows(torch.autograd.Function):
    """Zero-pad dim 0 to ``n`` rows; the backward is the row-prefix view.

    ``torch.nn.functional.pad``'s own backward is a second ``constant_pad_nd``
    (a copy of every input gradient, one launch each on the device); this one
    returns ``g[:rows]``, a view. Differentiable, so double backward holds.
    """

    @staticmethod
    def forward(ctx, t, n):
        ctx.rows = t.shape[0]
        return torch.nn.functional.pad(t, (0, 0) * (t.dim() - 1) + (0, n - t.shape[0]))

    @staticmethod
    def backward(ctx, g):
        return g[: ctx.rows], None


def _pad_rows(t, n):
    return t if t.shape[0] == n else _PadRows.apply(t, n)


# ── conv_tp ────────────────────────────────────────────────────────────────


class _ConvTPKernel(torch.autograd.Function):
    """XLA / simulator path: kernel forward and kernel backward."""

    @staticmethod
    def forward(ctx, x, y, w, cg, cgT, plan, zero_fill):
        ctx.save_for_backward(x, y, w, cg, cgT)
        ctx.plan = plan
        (out,) = _run(
            _ek.conv_tp_fwd_kernel, "conv_tp", (x, y, w, cg), {"plan": plan, "zero_fill": zero_fill}
        )
        return out

    @staticmethod
    def backward(ctx, g):
        x, y, w, cg, cgT = ctx.saved_tensors
        nx, ny, nw = ctx.needs_input_grad[:3]
        if torch.is_grad_enabled():  # create_graph: the kernel backward is first order
            dx, dy, dw = conv_tp_bwd_torch(g, x, y, w, cg, ctx.plan, nx, ny, nw)
            return dx, dy, dw, None, None, None, None
        (grads,) = _run(
            _ek.conv_tp_bwd_kernel,
            "conv_tp_bwd",
            (g.contiguous(), x, y, w, cg, cgT),
            {"plan": ctx.plan, "need_dx": int(nx), "need_dy": int(ny), "need_dw": int(nw)},
        )
        d1, d2 = x.shape[1], y.shape[1]
        dx, dy, dw = grads[:, :d1], grads[:, d1 : d1 + d2], grads[:, d1 + d2 :]
        return (dx if nx else None, dy if ny else None, dw if nw else None, None, None, None, None)


def conv_tp_nki(x, y, w, spec) -> torch.Tensor:
    """``conv_tp`` on the NKI kernels: fp32 compute, padded edges, autograd."""
    lay = spec.layout
    out_dtype = x.dtype
    E = x.shape[0]
    Ep = edge_bucket(E)
    native = _native()
    dev = torch.device(_d._NEURON_DEVICE) if native else x.device
    c = spec.constants(dev, torch.float32)

    def prep(t):
        return _pad_rows(t.to(dev, torch.float32), Ep).contiguous()

    xs, ys, ws = prep(x), prep(y), prep(w)
    cg, cgT = c["cg_kernel"], c["cg_kernel_T"]
    if native:
        check_operands("conv_tp", (xs, ys, ws, cg))
        out = _native_ops()["conv_tp"](xs, ys, ws, cg, cgT, list(lay.plan), lay.zero_fill)
    else:
        out = _ConvTPKernel.apply(xs, ys, ws, cg, cgT, lay.plan, lay.zero_fill)
    out = out[:E]
    return out.to(x.device, out_dtype)


# ── symmetric contraction ──────────────────────────────────────────────────


class _SymConKernel(torch.autograd.Function):
    """XLA / simulator path: kernel forward and kernel backward (∂x only)."""

    @staticmethod
    def forward(ctx, x, wr, sel, selT, usym, usymT, s, sT):
        ctx.save_for_backward(x, wr, sel, selT, usymT, sT)
        (out,) = _run(_ek.symcon_fwd_kernel, "symcon", (x, wr, sel, usym, s), {})
        return out

    @staticmethod
    def backward(ctx, g):
        x, wr, sel, selT, usymT, sT = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return (None,) * 8
        if torch.is_grad_enabled():  # create_graph: the kernel backward is first order
            return (symcon_bwd_torch(g, x, wr, sel, usymT, sT),) + (None,) * 7
        (dx,) = _run(
            _ek.symcon_bwd_kernel, "symcon_bwd", (g.contiguous(), x, wr, sel, selT, usymT, sT), {}
        )
        return (dx,) + (None,) * 7


def symmetric_contraction_nki(x, species, spec) -> torch.Tensor:
    """``symmetric_contraction`` on the NKI kernels: fp32, padded nodes, autograd."""
    lay = spec.layout
    out_dtype = x.dtype
    N, C, L = x.shape
    Np = node_pad(N, C)
    native = _native()
    dev = torch.device(_d._NEURON_DEVICE) if native else x.device
    c = spec.constants(dev, torch.float32)
    xs = x.to(dev, torch.float32)
    z = species.to(dev)
    if Np != N:
        xs = _pad_rows(xs, Np)
        z = torch.nn.functional.pad(z, (0, Np - N))
    xs = xs.contiguous()
    # Species enter as one row gather of the folded weight table.
    wr = c["W_zmc"].index_select(0, z).contiguous()
    consts = (c["sel"], c["selT"], c["usym"], c["usymT"], c["S"], c["ST"])
    if native:
        check_operands("symcon", (xs, wr, c["sel"], c["usym"], c["S"]))
        om = _native_ops()["symcon"](xs, wr, *consts)
    else:
        om = _SymConKernel.apply(xs, wr, *consts)
    # [D, Np*C] (m-major) -> [Np, C*D] -> MACE layout; differentiable aten.
    out = om.T.reshape(Np, C * lay.Dsum).index_select(1, c["out_gather"])[:N]
    return out.to(x.device, out_dtype)
