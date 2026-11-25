"""Equivariant linear maps as one dense matmul per irrep (trnsci/trnsci#178).

An e3nn ``o3.Linear`` mixes channels within each irrep type and never mixes
the ``2l+1`` components: on the mul-major layout (block ``[mul, 2l+1]``
row-major) its matrix is block-diagonal, ``W_l ⊗ I_{2l+1}`` for each irrep
type. e3nn evaluates it as a generated chain of per-path einsums (62 aten
ops for one 128x0e+…+128x3o linear on the MACE-MP-0b2 step); here it is one
matmul per irrep type against the folded matrix ``K_l = W_l ⊗ I_{2l+1}``
(every path normalisation folded in), plus a concatenation.

The same shape covers MACE's skip tensor product, an e3nn
``FullyConnectedTensorProduct`` of the node features with the one-hot
species: for one-hot attributes it is a per-species linear map, so the
spec keeps one ``K_l`` per species present (``elements``) and the op
evaluates ``x_l @ [K_l,s1 | K_l,s2 | …]`` and keeps each node's own species
block (a mask and a sum).

``equivariant_linear(x, spec, species=None)`` is differentiable in ``x``
(analytic backward ``g @ K_lᵀ``); the weights are constants of the spec
(forces need input gradients only). Under a region backend
(:func:`trntensor.set_region_backend`) the forward and the backward each
run as one compiled graph.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ._region import region
from .nki.dispatch import _fp64_host_guard

__all__ = ["EquivariantLinearSpec", "equivariant_linear", "equivariant_linear_spec"]

IrrepsP = tuple[tuple[int, int, int], ...]  # ((mul, l, parity), ...)


def _parse_irreps_p(irreps) -> IrrepsP:
    """``"128x0e+128x1o"``, an e3nn ``Irreps`` or ``[(mul, l, p)]`` → ``((mul, l, p), ...)``."""
    if isinstance(irreps, str):
        out = []
        for part in irreps.replace(" ", "").split("+"):
            if not part:
                continue
            mul, _, ir = part.rpartition("x")
            p = {"e": 1, "o": -1}.get(ir[-1], 1)
            out.append((int(mul) if mul else 1, int(ir.rstrip("eoy")), p))
        return tuple(out)
    out = []
    for item in irreps:
        if hasattr(item, "mul") and hasattr(item, "ir"):
            out.append((int(item.mul), int(item.ir.l), int(item.ir.p)))
        else:
            mul, ell, *p = item
            out.append((int(mul), int(ell), int(p[0]) if p else (-1) ** int(ell)))
    return tuple(out)


def _blocks(irreps: IrrepsP):
    offs, o = [], 0
    for mul, ell, _p in irreps:
        offs.append(o)
        o += mul * (2 * ell + 1)
    return offs, o


@dataclass(frozen=True, eq=False)
class EquivariantLinearSpec:
    """Static description of an equivariant linear map (one per species when ``elements``).

    ``matrix [S, d_in, d_out]`` (float64): the dense map per species (``S = 1``
    without species); ``bias [S, d_out]`` or None; ``elements``: the species
    indices (the one-hot columns) the ``S`` maps belong to, or None.
    Derived once (``layout``): per irrep type ``(l, p)`` the input and
    output column ranges and ``W [S, mul_in, mul_out]``, checked to
    reproduce ``matrix`` exactly; ``mode`` is ``"irrep"`` (one matmul per irrep
    type) when every type's blocks are contiguous on both sides, else
    ``"dense"`` (one matmul against ``matrix``).
    """

    irreps_in: IrrepsP
    irreps_out: IrrepsP
    matrix: torch.Tensor = field(repr=False)
    bias: torch.Tensor | None = field(default=None, repr=False)
    elements: tuple[int, ...] | None = None
    mode: str = "auto"

    def __post_init__(self):
        M = torch.as_tensor(self.matrix, dtype=torch.float64).detach().cpu()
        if M.dim() == 2:
            M = M.unsqueeze(0)
        object.__setattr__(self, "matrix", M)
        if self.bias is not None:
            b = torch.as_tensor(self.bias, dtype=torch.float64).detach().cpu()
            object.__setattr__(self, "bias", b.reshape(M.shape[0], -1))
        _o, din = _blocks(self.irreps_in)
        _o, dout = _blocks(self.irreps_out)
        if M.shape[1:] != (din, dout):
            raise ValueError(
                f"equivariant_linear: matrix {tuple(M.shape)} vs irreps dims {din}, {dout}"
            )
        if self.elements is not None and len(self.elements) != M.shape[0]:
            raise ValueError("equivariant_linear: one matrix per element")
        object.__setattr__(self, "groups", self._groups())
        object.__setattr__(self, "_consts", {})
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def dim_in(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def dim_out(self) -> int:
        return int(self.matrix.shape[2])

    @property
    def num_species(self) -> int:
        return int(self.matrix.shape[0])

    def _groups(self):
        """``[(l, ilo, ihi, olo, ohi, W [S, mi, mo])]`` per irrep type, or None (dense)."""
        M = self.matrix
        offi, _ = _blocks(self.irreps_in)
        offo, _ = _blocks(self.irreps_out)
        types = []
        for _m, ell, p in self.irreps_out:
            if (ell, p) not in types:
                types.append((ell, p))
        groups, covered = [], torch.zeros_like(M, dtype=torch.bool)
        contiguous = True
        for ell, p in types:
            d = 2 * ell + 1
            ib = [
                (o, m)
                for o, (m, l2, p2) in zip(offi, self.irreps_in, strict=True)
                if (l2, p2) == (ell, p)
            ]
            ob = [
                (o, m)
                for o, (m, l2, p2) in zip(offo, self.irreps_out, strict=True)
                if (l2, p2) == (ell, p)
            ]
            if not ib:
                continue
            for bl in (ib, ob):
                for (o1, m1), (o2, _m2) in zip(bl, bl[1:], strict=False):
                    if o1 + m1 * d != o2:
                        contiguous = False
            ilo, ihi = ib[0][0], ib[-1][0] + ib[-1][1] * d
            olo, ohi = ob[0][0], ob[-1][0] + ob[-1][1] * d
            if not contiguous:
                break
            blk = M[:, ilo:ihi, olo:ohi]
            W = blk[:, 0::d, 0::d].contiguous()
            K = torch.stack(
                [torch.kron(W[s], torch.eye(d, dtype=torch.float64)) for s in range(M.shape[0])]
            )
            scale = max(float(M.abs().max()), 1e-300)
            if float((K - blk).abs().max()) > 1e-10 * scale:
                raise ValueError(
                    f"equivariant_linear: the ({ell}, {p}) block is not W ⊗ I_{d}: not an equivariant linear map"
                )
            covered[:, ilo:ihi, olo:ohi] = True
            groups.append((ell, ilo, ihi, olo, ohi, W))
        if not contiguous:
            return None
        if float(M[~covered].abs().max()) if (~covered).any() else 0.0:
            raise ValueError("equivariant_linear: the matrix couples different irrep types")
        return tuple(sorted(groups, key=lambda g: g[3]))

    @property
    def irrep_mode(self) -> bool:
        if self.mode == "dense":
            return False
        if self.groups is None:
            if self.mode == "irrep":
                raise ValueError("equivariant_linear: blocks of an irrep type are not contiguous")
            return False
        return True

    def constants(self, device, dtype) -> dict:
        key = (torch.device(device), dtype)
        with self._lock:
            c = self._consts.get(key)
        if c is not None:
            return c
        dev = torch.device(device)
        S = self.num_species
        c = {}
        if self.irrep_mode:
            Ks, KTs = [], []
            for ell, _ilo, _ihi, _olo, _ohi, W in self.groups:
                d = 2 * ell + 1
                ks = [torch.kron(W[s], torch.eye(d, dtype=torch.float64)) for s in range(S)]
                K = torch.cat(ks, dim=1)  # [Din_l, S·Dout_l]
                Ks.append(K.to(dev, dtype).contiguous())
                KTs.append(K.T.contiguous().to(dev, dtype))
            c["K"], c["KT"] = Ks, KTs
        else:
            K = torch.cat(list(self.matrix), dim=1)
            c["K"], c["KT"] = [K.to(dev, dtype).contiguous()], [K.T.contiguous().to(dev, dtype)]
        c["bias"] = None if self.bias is None else self.bias.to(dev, dtype)
        c["one"] = torch.ones(1, 1, dtype=dtype, device=dev)
        c["elements"] = (
            None
            if self.elements is None
            else torch.tensor(self.elements, dtype=torch.long, device=dev)
        )
        with self._lock:
            self._consts[key] = c
        return c

    @property
    def static(self) -> tuple:
        """(segments of the output, segments of the input, S, dims, irrep_mode)."""
        S = self.num_species
        if not self.irrep_mode:
            return (
                (("k", 0, 0, self.dim_in, 0, self.dim_out),),
                (("k", 0, 0, self.dim_out, 0, self.dim_in),),
                S,
                self.dim_in,
                self.dim_out,
            )
        outs, o = [], 0
        for k, (_l, ilo, ihi, olo, ohi, _W) in enumerate(self.groups):
            if olo > o:
                outs.append(("z", -1, 0, 0, o, olo))
            outs.append(("k", k, ilo, ihi, olo, ohi))
            o = ohi
        if o < self.dim_out:
            outs.append(("z", -1, 0, 0, o, self.dim_out))
        ins, i = [], 0
        for k, (_l, ilo, ihi, olo, ohi, _W) in sorted(
            enumerate(self.groups), key=lambda kv: kv[1][1]
        ):
            if ilo > i:
                ins.append(("z", -1, 0, 0, i, ilo))
            ins.append(("k", k, olo, ohi, ilo, ihi))
            i = ihi
        if i < self.dim_in:
            ins.append(("z", -1, 0, 0, i, self.dim_in))
        return (tuple(outs), tuple(ins), S, self.dim_in, self.dim_out)


def _probe(mod, d_in, call):
    """Dense ``M [d_in, d_out]`` and ``b`` (fp64) of a linear ``call(mod64, x)``."""
    try:
        m64 = copy.deepcopy(mod).to(device="cpu", dtype=torch.float64)
    except Exception:  # some scripted modules do not deep-copy: probe in place
        m64 = mod.to(torch.float64)
    try:
        with torch.no_grad():
            z = call(m64, torch.zeros(1, d_in, dtype=torch.float64))
            M = call(m64, torch.eye(d_in, dtype=torch.float64)) - z
    finally:
        if m64 is mod:
            mod.to(torch.float32)
    return M, z[0]


def equivariant_linear_spec(
    module=None,
    *,
    elements: Sequence[int] | None = None,
    irreps_in=None,
    irreps_out=None,
    matrix=None,
    bias=None,
    mode: str = "auto",
) -> EquivariantLinearSpec:
    """Build an :class:`EquivariantLinearSpec`.

    - ``module`` an e3nn ``o3.Linear``: its dense map, probed once in float64
      (``M = f(I) − f(0)``, bias ``f(0)``), split per irrep type and checked.
    - ``module`` an e3nn ``FullyConnectedTensorProduct`` whose second input is
      scalars (MACE's ``skip_tp`` against the one-hot species), with
      ``elements``: one map per listed one-hot column.
    - explicit: ``irreps_in``, ``irreps_out`` and ``matrix [d_in, d_out]`` (or
      ``[S, d_in, d_out]`` with ``elements``), optional ``bias``.

    ``mode``: ``"auto"`` (per irrep type when the layout allows), ``"irrep"`` or
    ``"dense"`` (one matmul against the full matrix).
    """
    if module is not None:
        if hasattr(module, "irreps_in2"):
            in1, in2 = _parse_irreps_p(module.irreps_in1), _parse_irreps_p(module.irreps_in2)
            if any(ell != 0 for _m, ell, _p in in2):
                raise NotImplementedError("equivariant_linear: the second input must be scalars")
            if elements is None:
                raise ValueError("equivariant_linear_spec: pass elements= for a tensor product")
            n_attr = sum(m for m, _l, _p in in2)
            d_in = _blocks(in1)[1]
            Ms, bs = [], []
            for z in elements:
                oh = torch.zeros(1, n_attr, dtype=torch.float64)
                oh[0, int(z)] = 1.0

                def call(m, x, oh=oh):
                    return m(x, oh.expand(x.shape[0], n_attr))

                M, b = _probe(module, d_in, call)
                Ms.append(M)
                bs.append(b)
            b = torch.stack(bs)
            return EquivariantLinearSpec(
                in1,
                _parse_irreps_p(module.irreps_out),
                torch.stack(Ms),
                b if bool(b.abs().max() > 0) else None,
                tuple(int(z) for z in elements),
                mode,
            )
        ii, io = _parse_irreps_p(module.irreps_in), _parse_irreps_p(module.irreps_out)
        M, b = _probe(module, _blocks(ii)[1], lambda m, x: m(x))
        return EquivariantLinearSpec(ii, io, M, b if bool(b.abs().max() > 0) else None, None, mode)
    if irreps_in is None or irreps_out is None or matrix is None:
        raise ValueError(
            "equivariant_linear_spec: pass module= or irreps_in, irreps_out and matrix"
        )
    return EquivariantLinearSpec(
        _parse_irreps_p(irreps_in),
        _parse_irreps_p(irreps_out),
        matrix,
        bias,
        None if elements is None else tuple(int(z) for z in elements),
        mode,
    )


# ── PyTorch path ───────────────────────────────────────────────────────────
#
# ``st`` is EquivariantLinearSpec.static: segments ("k", group, src_lo, src_hi,
# dst_lo, dst_hi) or ("z", …) for zero columns; K[group] maps the source
# columns to S·(dst width) columns.


def _species_mask(species, elements, dtype):
    return (species[:, None] == elements[None, :]).to(dtype)


def _linear_forward_torch(x, Ks, bias, mask, st):
    outs, _ins, S, _din, _dout = st
    N = x.shape[0]
    parts = []
    for kind, k, lo, hi, olo, ohi in outs:
        if kind == "z":
            parts.append(x.new_zeros(N, ohi - olo))
            continue
        src = x if (lo, hi) == (0, x.shape[1]) else x[:, lo:hi]
        y = src @ Ks[k]
        if S > 1:
            y = (y.view(N, S, ohi - olo) * mask[:, :, None]).sum(1)
        parts.append(y)
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    if bias is not None:
        out = out + (bias[0] if S == 1 else mask @ bias)
    return out


def _linear_backward_torch(g, KTs, mask, st):
    _outs, ins, S, _din, _dout = st
    N = g.shape[0]
    parts = []
    for kind, k, lo, hi, ilo, ihi in ins:
        if kind == "z":
            parts.append(g.new_zeros(N, ihi - ilo))
            continue
        src = g if (lo, hi) == (0, g.shape[1]) else g[:, lo:hi]
        if S > 1:
            src = (src[:, None, :] * mask[:, :, None]).reshape(N, S * (hi - lo))
        parts.append(src @ KTs[k])
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


class _LinearTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, spec):
        c = spec.constants(x.device, x.dtype)
        st = spec.static
        ctx.spec = spec
        ctx.save_for_backward(mask)
        return region(_linear_forward_torch, id(spec))(x, c["K"], c["bias"], mask, st)

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        spec = ctx.spec
        c = spec.constants(g.device, g.dtype)
        dx = region(_linear_backward_torch, id(spec))(g.contiguous(), c["KT"], mask, spec.static)
        return dx, None, None


@_fp64_host_guard("equivariant_linear")
def equivariant_linear(
    x: torch.Tensor, spec: EquivariantLinearSpec, species: torch.Tensor | None = None
) -> torch.Tensor:
    """``x [N, dim(irreps_in)]`` → ``[N, dim(irreps_out)]``, one matmul per irrep type.

    Equals the e3nn module the spec was read from (``o3.Linear``, or the
    one-hot ``FullyConnectedTensorProduct`` with ``species [N]`` the one-hot
    column of each node; every node's species must be in ``spec.elements``).
    Differentiable in ``x``.
    """
    if not isinstance(spec, EquivariantLinearSpec):
        raise TypeError(
            f"equivariant_linear: expected an EquivariantLinearSpec, got {type(spec).__name__}"
        )
    if x.dim() != 2 or x.shape[1] != spec.dim_in:
        raise ValueError(f"equivariant_linear: x must be [N, {spec.dim_in}], got {tuple(x.shape)}")
    if not x.dtype.is_floating_point:
        raise TypeError(f"equivariant_linear: x must be floating, got {x.dtype}")
    c = spec.constants(x.device, x.dtype)
    if spec.num_species > 1:
        if species is None:
            raise ValueError("equivariant_linear: this spec is per species; pass species [N]")
        mask = _species_mask(species.to(x.device), c["elements"], x.dtype)
    else:
        mask = c["one"]  # unused placeholder (one species: no mask)
    return _LinearTorch.apply(x, mask, spec)
