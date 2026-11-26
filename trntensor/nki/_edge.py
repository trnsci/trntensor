"""Host side of the per-edge MLIP kernels (trnsci/trnsci#178).

Routing, padding, constant packing and autograd for
``trntensor.spherical_harmonics`` and ``trntensor.edge_chain`` on the NKI
kernels in ``_edge_kernels.py``, following ``_equivariant.py``:

- native TorchNeuron path: ``trntensor::sh`` / ``trntensor::edge_chain`` are
  ``nki_op`` custom ops whose backward (``*_bwd``) is attached with
  ``register_autograd`` (``_native_edge.py``);
- XLA path and the CPU simulator: ``torch.autograd.Function`` wrappers here.

When autograd records the backward (``create_graph=True``) the backward runs
the PyTorch path's analytic formulas on the same padded operands, so double
backward works; first-order backward (forces) stays on the kernels.

Edges are padded to :func:`trntensor.nki._equivariant.edge_bucket` (a
multiple of 256, so the partition-major view has an even column count for
LNC=2) and viewed as ``[128, M·F]``. Padded vectors are zero (their spherical
harmonics are constants and their cotangent is zero); padded lengths are
``3·r_max`` with ``r0 = 1``, past the cutoff.

``"auto"`` does not route here yet: the kernels are verified on CPU (MLIR
verification, ``nki.simulate``) but not yet on a device. ``set_backend("nki")``
or ``TRNTENSOR_EDGE_NKI=1`` (with ``"auto"``, fp32 on the ``neuron`` device)
selects them.
"""

from __future__ import annotations

import functools
import math
import os

import numpy as np
import torch

from . import _edge_kernels as _xk
from . import dispatch as _d
from ._equivariant import edge_bucket

EDGE_NKI_ENV = "TRNTENSOR_EDGE_NKI"

_F32 = frozenset({torch.float32})
EDGE_OPERAND_DTYPES: dict[str, tuple[frozenset, ...]] = {
    "sh": (_F32,),  # v
    "sh_bwd": (_F32,) * 2,  # g, v
    "edge_chain": (_F32,) * 3,  # r, r0, wpack
    "edge_chain_bwd": (_F32,) * 5,  # g, r, r0, wpack, wtpack
}


def _auto_uses_kernels() -> bool:
    return os.environ.get(EDGE_NKI_ENV, "").strip().lower() in ("1", "true", "yes")


def _route(what: str, fits: bool, tensors) -> bool:
    if not (_d._use_nki() and _d.HAS_NKI):
        return False
    if not _d._nki_forced():
        if not (_auto_uses_kernels() and _d._on_native_path()):
            return False
        if not all(_d._on_neuron_device(t) and t.dtype == torch.float32 for t in tensors):
            return False
    if not fits:
        if _d._require_nki_now():
            raise NotImplementedError(
                f"trntensor {what}: this spec is outside the NKI kernel's coverage; "
                "unset TRNTENSOR_REQUIRE_NKI to run the PyTorch path"
            )
        return False
    if not all(t.dtype.is_floating_point for t in tensors):
        return False
    if _d._has_fp64(*tensors):
        return _d._fp64_to_kernel(what, *tensors)
    return all(t.dtype in (torch.float32, torch.bfloat16, torch.float16) for t in tensors)


def sh_kernel_route(v, lmax, normalize, normalization) -> bool:
    if v.shape[0] == 0:
        return False
    return _route("spherical_harmonics", 0 <= int(lmax) <= 3, (v,))


def edge_chain_fits(spec) -> bool:
    """The kernel covers: Agnesi transform with ``max(w) ≤ 2π`` (so every sine
    argument stays in ``[−π, π]``), an integer cutoff order, hidden widths and
    basis ≤ 128, heads of ≥ 2 layers or one layer with one output, and the
    SBUF budget."""
    if spec.agnesi is None or max(spec.bessel_weights) > 2 * math.pi + 1e-12:
        return False
    p = float(spec.cutoff_p)
    if p != round(p) or p < 1:
        return False
    if spec.num_basis > 128:
        return False
    for h in spec.heads:
        dims = [h.dim_in] + [int(w.shape[1]) for w in h.weights]
        if len(h.weights) == 1 and dims[-1] != 1:
            return False
        if any(d > 128 for d in dims[:-1]):
            return False
    dims = kernel_dims(spec)
    return _xk.edge_chain_sbuf_bytes(dims, spec.num_basis) <= 160 * 1024


def edge_chain_kernel_route(r, r0, spec) -> bool:
    if r.shape[0] == 0:
        return False
    return _route("edge_chain", edge_chain_fits(spec), (r, r0))


# ── constants ──────────────────────────────────────────────────────────────


@functools.cache
def sh_tables(lmax: int, normalization: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Static coefficient tables of the SH kernels (tuples: NKI static arguments).

    Cached: built per call, the per-element ``float()`` conversions were ~800
    host ``aten::item`` dispatches per spherical-harmonics call (Trn2, window 8).
    """
    from ..edge import sh_coefficients

    t = sh_coefficients(lmax, normalization)
    table = t["c0"].tolist() + t["C"].reshape(-1).tolist()
    if lmax == 0:
        table = table[:1] + [0.0]
    return tuple(float(v) for v in table), tuple(float(v) for v in t["dC"].reshape(-1).tolist())


def kernel_dims(spec) -> tuple[int, ...]:
    dims = [len(spec.heads)]
    for h in spec.heads:
        hs = [h.dim_in] + [int(w.shape[1]) for w in h.weights]
        dims += [len(h.weights), _xk.POST_CODES[h.post]] + hs
    return tuple(dims)


def kernel_consts(spec) -> tuple[float, ...]:
    a, q, pp = spec.agnesi if spec.agnesi is not None else (0.0, 0.0, 0.0)
    return (
        float(spec.r_max),
        float(spec.prefactor),
        float(spec.cutoff_p),
        float(a),
        float(q),
        float(pp),
        float(int(spec.apply_cutoff)),
    )


def pack_weights(spec) -> tuple[torch.Tensor, torch.Tensor]:
    """``wpack [128, WF]`` and ``wtpack [128, WB]`` (float64, host) per
    ``_edge_kernels.edge_pack_layout``."""
    P = _xk.EDGE_PART
    dims = kernel_dims(spec)
    fwd, WF, bwd, WB = _xk.edge_pack_layout(dims)
    wp = torch.zeros(P, WF, dtype=torch.float64)
    wt = torch.zeros(P, WB, dtype=torch.float64)
    for h, head in enumerate(spec.heads):
        ws = head.weights
        for ell, W in enumerate(ws):
            hi, ho = W.shape
            wp[:hi, fwd[h][ell] : fwd[h][ell] + ho] = W
            if len(ws) == 1 and ho == 1:
                wt[:, bwd[h][0] : bwd[h][0] + hi] = W[:, 0][None, :]
                continue
            WT = W.T
            for c in range(-(-ho // P)):
                rows = WT[c * P : (c + 1) * P]
                o = bwd[h][ell] + c * hi
                wt[: rows.shape[0], o : o + hi] = rows
    return wp, wt


def _edge_kernel_constants(spec, dev) -> dict:
    key = ("kernel", torch.device(dev))
    with spec._lock:
        c = spec._consts.get(key)
    if c is None:
        wp, wt = pack_weights(spec)
        c = {
            "wpack": wp.to(dev, torch.float32).contiguous(),
            "wtpack": wt.to(dev, torch.float32).contiguous(),
            "dims": kernel_dims(spec),
            "consts": kernel_consts(spec),
            "bessel": tuple(float(w) for w in spec.bessel_weights),
        }
        with spec._lock:
            spec._consts[key] = c
    return c


# ── launching ──────────────────────────────────────────────────────────────


def check_operands(op_name: str, tensors) -> None:
    allowed = EDGE_OPERAND_DTYPES.get(op_name)
    if allowed is None:
        raise TypeError(f"trntensor kernel {op_name!r} has no operand dtype entry")
    dts = [t.dtype for t in tensors]
    if len(dts) != len(allowed) or not all(d in a for d, a in zip(dts, allowed, strict=True)):
        raise TypeError(
            f"trntensor kernel {op_name!r} takes {len(allowed)} float32 operands; got "
            f"({', '.join(str(d) for d in dts)}). This is a trntensor bug: the host "
            "wrapper should have cast the operands."
        )


def _run(kernel, op_name: str, tensors, consts) -> torch.Tensor:
    """One kernel launch (simulator or XLA); output on ``tensors[0]``'s device."""
    check_operands(op_name, tensors)
    dev = tensors[0].device
    if _d._use_simulator():
        out = _d.nki.simulate(kernel)(*(t.detach().cpu().numpy() for t in tensors), **consts)
        return torch.from_numpy(np.asarray(out)).to(dev)
    xs, orig = _d._to_xla(*tensors)
    return kernel(*xs, **consts).to(orig)


_native_ops_table = None


def _native_ops() -> dict:
    global _native_ops_table
    if _native_ops_table is None:
        from . import _native_edge

        missing = sorted(set(_native_edge.OPS) - set(EDGE_OPERAND_DTYPES))
        if missing:
            raise TypeError(f"edge native ops without an operand dtype entry: {missing}")
        _native_ops_table = _native_edge.OPS
    return _native_ops_table


def _native() -> bool:
    return _d.HAS_NKI and not _d._use_simulator() and _d._get_dispatch_path() == "native"


class _PadRowsValue(torch.autograd.Function):
    """Pad dim 0 to ``n`` rows with ``value``; the backward is the row-prefix view."""

    @staticmethod
    def forward(ctx, t, n, value):
        ctx.rows = t.shape[0]
        return torch.nn.functional.pad(t, (0, 0, 0, n - t.shape[0]), value=value)

    @staticmethod
    def backward(ctx, g):
        return g[: ctx.rows], None, None


def _pad(t, n, value=0.0):
    return t if t.shape[0] == n else _PadRowsValue.apply(t, n, value)


# ── differentiable statements of the kernel contracts ─────────────────────
#
# On the kernels' own operands (partition-major 2-D views, packed weights,
# static lists). The backward ``*_bwd_torch`` run when autograd records the
# backward (create_graph=True); the forward ones are the emulators the CPU
# tests use.


def sh_fwd_torch(v2, lmax, normalize, eps, table):
    """``sh_fwd_kernel`` contract: ``v2 [128, M·3]`` → ``[128, M·D]``."""
    from ..edge import _sh_forward_torch

    D = (lmax + 1) ** 2
    c0 = torch.tensor(table[:D], dtype=v2.dtype, device=v2.device)
    C = torch.tensor(table[D:], dtype=v2.dtype, device=v2.device).reshape(-1, D)
    v = v2.reshape(-1, 3)
    return _sh_forward_torch(v, c0, C, lmax, bool(normalize), eps).reshape(v2.shape[0], -1)


def sh_bwd_torch(g2, v2, lmax, normalize, eps, dtable):
    """``sh_bwd_kernel`` contract: ``∂v2 [128, M·3]`` for ``g2 [128, M·D]``."""
    from ..edge import _sh_backward_torch

    D = (lmax + 1) ** 2
    dC = torch.tensor(dtable, dtype=g2.dtype, device=g2.device).reshape(D, -1)
    v = v2.reshape(-1, 3)
    g = g2.reshape(v.shape[0], D)
    return _sh_backward_torch(g, v, dC, lmax, bool(normalize), eps).reshape(v2.shape)


def _static(dims, consts):
    r_max, pf, p, a, q, pp, apply = consts
    posts = {v: k for k, v in _xk.POST_CODES.items()}
    heads = tuple((len(hs) - 1, posts[post]) for post, hs in _xk.edge_heads(dims))
    return (
        float(r_max),
        float(pf),
        float(p),
        (float(a), float(q), float(pp)),
        bool(int(apply)),
        heads,
    )


def _unpack(pack, dims):
    fwd, _wf, _b, _wb = _xk.edge_pack_layout(dims)
    Ws = []
    for h, (_post, hs) in enumerate(_xk.edge_heads(dims)):
        for ell in range(1, len(hs)):
            o = fwd[h][ell - 1]
            Ws.append(pack[: hs[ell - 1], o : o + hs[ell]])
    return Ws


def edge_chain_fwd_torch(r2, r02, wpack, dims, consts, bessel):
    """``edge_chain_fwd_kernel`` contract: ``[128, M·Σd]``."""
    from ..edge import _edge_chain_forward_torch

    r, r0 = r2.reshape(-1, 1), r02.reshape(-1, 1)
    w = torch.tensor(bessel, dtype=r2.dtype, device=r2.device)
    outs = _edge_chain_forward_torch(r, r0, w, _unpack(wpack, dims), _static(dims, consts))
    return torch.cat(outs, 1).reshape(r2.shape[0], -1)


def edge_chain_bwd_torch(g2, r2, r02, wpack, dims, consts, bessel):
    """``edge_chain_bwd_kernel`` contract: ``∂r2 [128, M]`` for ``g2 [128, M·Σd]``."""
    from ..edge import _edge_chain_backward_torch

    r, r0 = r2.reshape(-1, 1), r02.reshape(-1, 1)
    w = torch.tensor(bessel, dtype=r2.dtype, device=r2.device)
    Ws = _unpack(wpack, dims)
    _offs, DO = _xk.edge_out_offsets(dims)
    g = g2.reshape(r.shape[0], DO)
    gs = list(torch.split(g, [int(W.shape[1]) for W in _last_layers(Ws, dims)], dim=1))
    WTs = [W.T for W in Ws]
    return _edge_chain_backward_torch(gs, r, r0, w, Ws, WTs, _static(dims, consts)).reshape(
        r2.shape
    )


def _last_layers(Ws, dims):
    out, i = [], 0
    for _post, hs in _xk.edge_heads(dims):
        i += len(hs) - 1
        out.append(Ws[i - 1])
    return out


# ── spherical harmonics ────────────────────────────────────────────────────


class _SHKernel(torch.autograd.Function):
    """XLA / simulator path: kernel forward and kernel backward."""

    @staticmethod
    def forward(ctx, v2, lmax, normalize, eps, table, dtable):
        ctx.args = (lmax, normalize, eps, dtable)
        ctx.save_for_backward(v2)
        return _run(
            _xk.sh_fwd_kernel,
            "sh",
            (v2,),
            {"lmax": lmax, "normalize": normalize, "eps": eps, "table": table},
        )

    @staticmethod
    def backward(ctx, g2):
        (v2,) = ctx.saved_tensors
        lmax, normalize, eps, dtable = ctx.args
        if torch.is_grad_enabled():  # create_graph: the kernel backward is first order
            return (sh_bwd_torch(g2, v2, lmax, normalize, eps, dtable),) + (None,) * 5
        dv = _run(
            _xk.sh_bwd_kernel,
            "sh_bwd",
            (g2.contiguous(), v2),
            {"lmax": lmax, "normalize": normalize, "eps": eps, "dtable": dtable},
        )
        return (dv,) + (None,) * 5


def spherical_harmonics_nki(v, lmax, normalize, normalization, eps) -> torch.Tensor:
    """``spherical_harmonics`` on the NKI kernels: fp32, padded edges, autograd."""
    out_dtype = v.dtype
    E = v.shape[0]
    Ep = edge_bucket(E)
    native = _native()
    dev = torch.device(_d._NEURON_DEVICE) if native else v.device
    v2 = _pad(v.to(dev, torch.float32), Ep).contiguous().view(_xk.EDGE_PART, -1)
    table, dtable = sh_tables(lmax, normalization)
    args = (int(lmax), int(normalize), float(eps), table, dtable)
    if native:
        check_operands("sh", (v2,))
        out2 = _native_ops()["sh"](v2, *args[:3], list(table), list(dtable))
    else:
        out2 = _SHKernel.apply(v2, *args)
    return out2.view(Ep, -1)[:E].to(v.device, out_dtype)


# ── edge chain ─────────────────────────────────────────────────────────────


class _EdgeChainKernel(torch.autograd.Function):
    """XLA / simulator path: kernel forward and kernel backward (∂r only)."""

    @staticmethod
    def forward(ctx, r2, r02, wpack, wtpack, dims, consts, bessel):
        ctx.args = (dims, consts, bessel)
        ctx.save_for_backward(r2, r02, wpack, wtpack)
        return _run(
            _xk.edge_chain_fwd_kernel,
            "edge_chain",
            (r2, r02, wpack),
            {"dims": dims, "consts": consts, "bessel": bessel},
        )

    @staticmethod
    def backward(ctx, g2):
        r2, r02, wpack, wtpack = ctx.saved_tensors
        dims, consts, bessel = ctx.args
        if not ctx.needs_input_grad[0]:
            return (None,) * 7
        if torch.is_grad_enabled():  # create_graph: the kernel backward is first order
            return (edge_chain_bwd_torch(g2, r2, r02, wpack, dims, consts, bessel),) + (None,) * 6
        dr = _run(
            _xk.edge_chain_bwd_kernel,
            "edge_chain_bwd",
            (g2.contiguous(), r2, r02, wpack, wtpack),
            {"dims": dims, "consts": consts, "bessel": bessel},
        )
        return (dr,) + (None,) * 6


def edge_chain_nki(r, r0, spec) -> tuple:
    """``edge_chain`` on the NKI kernels: fp32, padded edges, autograd."""
    out_dtype = r.dtype
    E = r.shape[0]
    Ep = edge_bucket(E)
    native = _native()
    dev = torch.device(_d._NEURON_DEVICE) if native else r.device
    k = _edge_kernel_constants(spec, dev)
    P = _xk.EDGE_PART
    r2 = _pad(r.to(dev, torch.float32), Ep, 3.0 * float(spec.r_max)).contiguous().view(P, -1)
    r02 = _pad(r0.detach().to(dev, torch.float32), Ep, 1.0).contiguous().view(P, -1)
    args = (k["wpack"], k["wtpack"], k["dims"], k["consts"], k["bessel"])
    if native:
        check_operands("edge_chain", (r2, r02, k["wpack"]))
        out2 = _native_ops()["edge_chain"](
            r2, r02, k["wpack"], k["wtpack"], list(k["dims"]), list(k["consts"]), list(k["bessel"])
        )
    else:
        out2 = _EdgeChainKernel.apply(r2, r02, *args)
    flat = out2.view(Ep, -1)[:E]
    return tuple(o.to(r.device, out_dtype) for o in torch.split(flat, list(spec.dims_out), dim=1))
