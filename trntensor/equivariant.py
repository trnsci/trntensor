"""Fixed-shape equivariant contractions for MACE-style interatomic potentials.

Two contractions dominate a MACE energy + forces step (trnsci/trntensor#138,
the MLIP exemplar trnsci/trnsci#170):

``conv_tp(x, y, w, spec)``
    The interaction tensor product: an e3nn ``TensorProduct`` with "uvu"
    instructions and per-edge weights. For edge ``e``, channel ``u`` and path
    ``p = (l1 ⊗ l2 → l3, α_p)``::

        out_p[e, u, k] = α_p · w[e, p, u] · Σ_ij C^{l1 l2 l3}_ijk · x_{l1}[e, u, i] · y_{l2}[e, j]

    with ``C`` the real Wigner-3j symbols in the e3nn basis.

``symmetric_contraction(x, species, spec)``
    MACE's product basis (``SymmetricContraction``), correlation ν ≤ 3::

        out[n, c, m] = Σ_ν Σ_k W_ν[z_n, k, c] · Σ_{a_1..a_ν} U_ν[m, a_1..a_ν, k] · Π_j x[n, c, a_j]

Both are differentiable (``torch.autograd``) with analytic backward passes:
``∂x``, ``∂y`` and ``∂w`` for ``conv_tp``; ``∂x`` for the symmetric
contraction (the species tables are constants of the spec).

A *spec* is a static description built once from the model
(:func:`conv_tp_spec`, :func:`symmetric_contraction_spec`). It owns every
constant (Clebsch-Gordan blocks, coupling tensors, per-element weights) in
the precomputed layouts the PyTorch path and the NKI kernels use; edge and
node counts stay runtime shapes, and no index is ever compiled in.

Backends follow ``trntensor.set_backend``:

- ``"pytorch"``: the fused PyTorch formulation on the operands' device. It is
  a handful of large ops per call instead of e3nn's per-path einsum chains,
  and it matches e3nn / MACE numerically (fp32 accumulation).
- ``"nki"``: the NKI kernels (``trntensor/nki/_equivariant_kernels.py``), one
  launch per direction (forward, backward) per layer. float64 follows the
  suite policy (computed in fp32 with a one-time warning).
- ``"auto"``: the PyTorch path. The NKI kernels are opt-in until they are
  validated on device; set ``TRNTENSOR_EQUIVARIANT_NKI=1`` to let ``"auto"``
  use them for fp32 operands on the ``neuron`` device. float64 on the
  ``neuron`` device is computed on the host (suite fp64 policy).

MACE and e3nn are optional: :func:`conv_tp_spec` and
:func:`symmetric_contraction_spec` read a module when one is given, and
otherwise build every constant from explicit irreps (``conv_tp_spec``) or
explicit coupling tensors and weights.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import torch

from .nki.dispatch import _fp64_host_guard

__all__ = [
    "ConvTPSpec",
    "SymmetricContractionSpec",
    "conv_tp",
    "conv_tp_spec",
    "symmetric_contraction",
    "symmetric_contraction_spec",
    "wigner_3j",
    "wigner_D",
]

# Largest l with built-in Wigner-3j constants. The e3nn sign convention is
# checked against e3nn for every triple up to this l (tests/test_equivariant.py).
L_MAX = 3

Irreps = tuple[tuple[int, int], ...]  # ((mul, l), ...); parity is not needed here


# ─────────────────────────────────────────────────────────────── SO(3) constants


def _change_basis_real_to_complex(ell: int) -> torch.Tensor:
    """e3nn's real ↔ complex spherical-harmonic basis change (with the ``(-i)^l``
    factor that makes the real Clebsch-Gordan coefficients real)."""
    q = torch.zeros((2 * ell + 1, 2 * ell + 1), dtype=torch.complex128)
    for m in range(-ell, 0):
        q[ell + m, ell + abs(m)] = 1 / 2**0.5
        q[ell + m, ell - abs(m)] = -1j / 2**0.5
    q[ell, ell] = 1
    for m in range(1, ell + 1):
        q[ell + m, ell + abs(m)] = (-1) ** m / 2**0.5
        q[ell + m, ell - abs(m)] = 1j * (-1) ** m / 2**0.5
    return (-1j) ** ell * q


def _su2_cg(j1: int, m1: int, j2: int, m2: int, j3: int, m3: int) -> float:
    """Clebsch-Gordan coefficient <j1 m1 j2 m2 | j3 m3> (Racah formula, exact)."""
    if m3 != m1 + m2:
        return 0.0
    vmin = max(-j1 + j2 + m3, -j1 + m1, 0)
    vmax = min(j2 + j3 + m1, j3 - j1 + j2, j3 + m3)
    f = math.factorial
    c = (2 * j3 + 1) * Fraction(
        f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3) * f(j3 + m3) * f(j3 - m3),
        f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
    )
    s = Fraction(0)
    for v in range(vmin, vmax + 1):
        s += (-1) ** (v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3),
        )
    return float(c) ** 0.5 * float(s)


# e3nn 0.4.x ships its Wigner-3j symbols as precomputed constants. In the same
# real basis they equal the normalised Racah construction up to one sign per
# sorted triple (l1 ≤ l2 ≤ l3); these sorted triples (l ≤ L_MAX) carry the
# opposite sign. Permuted triples follow from the sorted one in both
# conventions by the same (-1)^(l1+l2+l3) transposition rule.
_E3NN_FLIPPED_SORTED = frozenset({(1, 2, 2), (1, 3, 3)})

_w3j_cache: dict[tuple[int, int, int], torch.Tensor] = {}
_cache_lock = threading.Lock()


def wigner_3j(l1: int, l2: int, l3: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """Real Wigner-3j symbols ``C[i, j, k]`` of shape ``(2l1+1, 2l2+1, 2l3+1)``.

    Same basis, normalisation (``‖C‖ = 1``) and sign convention as e3nn's
    ``o3.wigner_3j`` (0.4.x) for ``l ≤ 3``, so a spec built from explicit
    irreps matches one read from an e3nn module. ``C`` is invariant under
    :func:`wigner_D` acting on all three indices:
    ``einsum("ijk,ia,jb,kc->abc", C, D1, D2, D3) == C``.
    """
    l1, l2, l3 = int(l1), int(l2), int(l3)
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError(f"wigner_3j({l1}, {l2}, {l3}): triangle condition fails")
    if max(l1, l2, l3) > L_MAX:
        raise ValueError(
            f"wigner_3j: built-in constants cover l <= {L_MAX}; pass w3j= explicitly "
            "(e.g. from e3nn.o3.wigner_3j) for higher l"
        )
    key = (l1, l2, l3)
    with _cache_lock:
        c = _w3j_cache.get(key)
    if c is None:
        cg = torch.zeros(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1, dtype=torch.complex128)
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                if abs(m1 + m2) <= l3:
                    cg[l1 + m1, l2 + m2, l3 + m1 + m2] = _su2_cg(l1, m1, l2, m2, l3, m1 + m2)
        q1, q2, q3 = (_change_basis_real_to_complex(v) for v in key)
        c = torch.einsum("ij,kl,mn,ikn->jlm", q1, q2, torch.conj(q3.T), cg)
        assert c.imag.abs().max() < 1e-10
        c = c.real
        c = c / c.norm()
        if tuple(sorted(key)) in _E3NN_FLIPPED_SORTED:
            c = -c
        with _cache_lock:
            _w3j_cache[key] = c
    return c.to(dtype=dtype, device=device, copy=True)


def _so3_generators(ell: int) -> torch.Tensor:
    """Real so(3) generators ``(3, 2l+1, 2l+1)`` in the e3nn basis (x, y, z order)."""
    j = ell
    m = torch.arange(-j, j, dtype=torch.float64)
    raising = torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)
    m = torch.arange(-j + 1, j + 1, dtype=torch.float64)
    lowering = torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)
    m = torch.arange(-j, j + 1, dtype=torch.float64)
    x = torch.stack(
        [
            0.5 * (raising + lowering),
            torch.diag(1j * m),
            -0.5j * (raising - lowering),
        ]
    ).to(torch.complex128)
    q = _change_basis_real_to_complex(ell)
    x = torch.conj(q.T) @ x @ q
    assert x.imag.abs().max() < 1e-10
    return x.real


def wigner_D(ell: int, alpha, beta, gamma) -> torch.Tensor:
    """Real Wigner D matrix of irrep ``l`` for the rotation ``Y(α) X(β) Y(γ)``.

    Same basis and Euler convention as e3nn's ``o3.wigner_D``. Angles are
    floats or 0-d tensors; returns ``(2l+1, 2l+1)`` float64. Used for
    equivariance checks: rotating every input irrep by ``D`` rotates the
    outputs of :func:`conv_tp` and :func:`symmetric_contraction` by ``D``.
    """
    X = _so3_generators(int(ell))
    a, b, c = (torch.as_tensor(v, dtype=torch.float64) for v in (alpha, beta, gamma))
    return (
        torch.linalg.matrix_exp(a * X[1])
        @ torch.linalg.matrix_exp(b * X[0])
        @ torch.linalg.matrix_exp(c * X[1])
    )


# ─────────────────────────────────────────────────────────────── irreps helpers


def _parse_irreps(irreps) -> Irreps:
    """``"128x0e+128x1o"``, an e3nn ``Irreps`` or ``[(mul, l), ...]`` → ``((mul, l), ...)``."""
    if irreps is None:
        raise ValueError("irreps must be given")
    if isinstance(irreps, str):
        out = []
        for part in irreps.replace(" ", "").split("+"):
            if not part:
                continue
            mul, _, ir = part.rpartition("x")
            out.append((int(mul) if mul else 1, int(ir.rstrip("eoy"))))
        return tuple(out)
    out = []
    for item in irreps:
        if hasattr(item, "mul") and hasattr(item, "ir"):  # e3nn _MulIr
            out.append((int(item.mul), int(item.ir.l)))
        else:
            mul, ir = item
            out.append((int(mul), int(getattr(ir, "l", ir))))
    return tuple(out)


def _offsets(irreps: Irreps) -> tuple[list[int], int]:
    offs, o = [], 0
    for mul, ell in irreps:
        offs.append(o)
        o += mul * (2 * ell + 1)
    return offs, o


# ─────────────────────────────────────────────────────────────── A1: conv_tp spec


@dataclass(frozen=True, eq=False)
class ConvTPSpec:
    """Static description of a "uvu" tensor product with per-edge weights.

    Fields mirror e3nn: ``irreps_in1`` (sender features, every block with the
    same multiplicity ``U``), ``irreps_in2`` (spherical harmonics, multiplicity
    1), ``irreps_out``, and ``instructions`` ``(i_in1, i_in2, i_out, α)`` in
    e3nn order (which is also the weight order: path ``p`` owns
    ``w[:, p·U:(p+1)·U]``). ``w3j`` maps ``"l1,l2,l3"`` to the real Wigner-3j
    tensor. Build with :func:`conv_tp_spec`.

    The derived constants (``layout``) are computed once:

    - ``cg``: the dense block-CG matrix ``[dim(in2), D1·K]`` with α folded in,
      where ``D1 = Σ (2 l1 + 1)`` over the in1 blocks and ``K = Σ_p (2 l3 + 1)``.
      One matmul ``y @ cg`` gives every per-edge coupling block
      ``G[e] ∈ R^{D1×K}``, and then ``out[e, u, :] = (x_cat[e, u, :] @ G[e]) ⊙ w``.
    - ``cg_kernel``: the compact layout ``[dim(in2), Σ_p d1·d3]`` the NKI kernels
      keep resident in SBUF, and ``plan``, the static per-path tuple they unroll.
    """

    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    instructions: tuple[tuple[int, int, int, float], ...]
    w3j: Mapping[str, torch.Tensor] = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_layout", _ConvTPLayout(self))
        object.__setattr__(self, "_consts", {})
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def layout(self) -> _ConvTPLayout:
        return self._layout

    @property
    def weight_numel(self) -> int:
        return self._layout.P * self._layout.U

    @property
    def dim_in1(self) -> int:
        return _offsets(self.irreps_in1)[1]

    @property
    def dim_in2(self) -> int:
        return _offsets(self.irreps_in2)[1]

    @property
    def dim_out(self) -> int:
        return _offsets(self.irreps_out)[1]

    def as_dict(self) -> dict:
        """The capture / ``check.py`` dict format of the MLIP exemplar."""
        return {
            "irreps_in1": [list(t) for t in self.irreps_in1],
            "irreps_in2": [list(t) for t in self.irreps_in2],
            "irreps_out": [list(t) for t in self.irreps_out],
            "instructions": [tuple(i) for i in self.instructions],
            "w3j": dict(self.w3j),
            "weight_numel": self.weight_numel,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> ConvTPSpec:
        """Spec from the exemplar's dict format (``irreps_in1``, ``irreps_in2``,
        ``irreps_out``, ``instructions`` ``(i1, i2, io, α)``, ``w3j``)."""
        return conv_tp_spec(
            d["irreps_in1"],
            d["irreps_in2"],
            d["irreps_out"],
            [tuple(i) for i in d["instructions"]],
            w3j=d.get("w3j"),
        )

    def constants(self, device, dtype) -> dict:
        """The derived constants on ``device`` in ``dtype`` (cached)."""
        key = (torch.device(device), dtype)
        with self._lock:
            c = self._consts.get(key)
        if c is None:
            c = self._layout.materialise(device, dtype)
            with self._lock:
                self._consts[key] = c
        return c


class _ConvTPLayout:
    """Index maps and CG blocks of a :class:`ConvTPSpec` (host, float64)."""

    def __init__(self, spec: ConvTPSpec):
        in1, in2, out = spec.irreps_in1, spec.irreps_in2, spec.irreps_out
        if not spec.instructions:
            raise ValueError("conv_tp spec has no instructions")
        muls1 = {m for m, _ in in1}
        if len(muls1) != 1:
            raise NotImplementedError(
                f"conv_tp: every irreps_in1 block must share one multiplicity, got {in1}"
            )
        (self.U,) = muls1
        U = self.U
        if any(m != 1 for m, _ in in2):
            raise NotImplementedError("conv_tp: irreps_in2 multiplicities must be 1 ('uvu')")
        off1, self.D1 = _offsets(in1)
        off2, self.D2 = _offsets(in2)
        offo, self.Dout = _offsets(out)
        # x_cat[e, u, c]: the components of channel u from every in1 block.
        self.cat_off, c = [], 0
        for _, ell in in1:
            self.cat_off.append(c)
            c += 2 * ell + 1
        self.Dcat = c
        x_perm = torch.empty(U * self.Dcat, dtype=torch.long)
        for b, (_, ell) in enumerate(in1):
            d = 2 * ell + 1
            for u in range(U):
                for i in range(d):
                    x_perm[u * self.Dcat + self.cat_off[b] + i] = off1[b] + u * d + i
        self.x_perm = x_perm
        self.x_perm_inv = torch.argsort(x_perm)

        self.P = len(spec.instructions)
        self.paths = []  # (i1, i2, io, alpha, l1, l2, l3)
        k_off, self.k_off = 0, []
        for i1, i2, io, alpha in spec.instructions:
            i1, i2, io = int(i1), int(i2), int(io)
            (m1, l1), (_, l2), (mo, l3) = in1[i1], in2[i2], out[io]
            if mo != m1:
                raise NotImplementedError(
                    f"conv_tp: 'uvu' needs mul_out == mul_in1, got {mo} != {m1} (path {i1},{i2},{io})"
                )
            if not abs(l1 - l2) <= l3 <= l1 + l2:
                raise ValueError(f"conv_tp: path ({l1} x {l2} -> {l3}) breaks the triangle rule")
            self.paths.append((i1, i2, io, float(alpha), l1, l2, l3))
            self.k_off.append(k_off)
            k_off += 2 * l3 + 1
        self.K = k_off

        # Dense block-CG with alpha: cg[y_off + j, c * K + k_off_p + k] = alpha_p C_p[i, j, k].
        cg = torch.zeros(self.D2, self.Dcat * self.K, dtype=torch.float64)
        # Compact kernel layout: columns (p, i, k), Q = sum_p d1 d3.
        q_cols = sum((2 * p[4] + 1) * (2 * p[6] + 1) for p in self.paths)
        cgk = torch.zeros(self.D2, q_cols, dtype=torch.float64)
        self.q_off = []
        q = 0
        for p, (i1, i2, _io, alpha, l1, l2, l3) in enumerate(self.paths):
            C = _w3j_lookup(spec.w3j, l1, l2, l3) * alpha
            d1, d2, d3 = 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
            for i in range(d1):
                col = (self.cat_off[i1] + i) * self.K + self.k_off[p]
                cg[off2[i2] : off2[i2] + d2, col : col + d3] = C[i]
                cgk[off2[i2] : off2[i2] + d2, q + i * d3 : q + (i + 1) * d3] = C[i]
            self.q_off.append(q)
            q += d1 * d3
        self.cg = cg
        self.cg_kernel = cgk
        self.Q = q_cols

        # w expansion: column (p, k) of the K axis reads path p's weights.
        self.w_kmap = torch.empty(self.K, dtype=torch.long)
        for p, (*_, l3) in enumerate(self.paths):
            self.w_kmap[self.k_off[p] : self.k_off[p] + 2 * l3 + 1] = p
        # out[e, offo[io] + u * d3 + k]  <-  r[e, u * K + k_off_p + k]
        dst = torch.empty(U * self.K, dtype=torch.long)
        covered = torch.zeros(self.Dout, dtype=torch.long)
        for p, (_i1, _i2, io, _alpha, _l1, _l2, l3) in enumerate(self.paths):
            d3 = 2 * l3 + 1
            for u in range(U):
                for k in range(d3):
                    o = offo[io] + u * d3 + k
                    dst[u * self.K + self.k_off[p] + k] = o
                    covered[o] += 1
        self.out_dst = dst
        # Single path per output block and every output column written: a gather.
        self.out_bijective = bool((covered == 1).all())
        self.out_gather = torch.argsort(dst) if self.out_bijective else None

        # Runs: maximal groups of consecutive paths with one 2l3+1 whose output
        # blocks are consecutive in out. A run is a [E, n, U, d3] view of out and
        # a [E, U, n, d3] view of the K axis, so w scaling and the layout change
        # are one broadcast multiply per run (bijective specs only).
        self.runs = []  # (p0, n, d3, k0, o0)
        if self.out_bijective:
            for p, (_i1, _i2, io, _alpha, _l1, _l2, l3) in enumerate(self.paths):
                d3 = 2 * l3 + 1
                if self.runs:
                    p0, n, rd3, k0, o0 = self.runs[-1]
                    if rd3 == d3 and offo[io] == o0 + n * U * d3:
                        self.runs[-1] = (p0, n + 1, d3, k0, o0)
                        continue
                self.runs.append((p, 1, d3, self.k_off[p], offo[io]))
        # x blocks as views: x_cat[:, :, cat_off_b : +d_b] = x[:, off1_b : +U d_b].view(E, U, d_b).
        self.x_blocks = [(off1[b], self.cat_off[b], 2 * ell + 1) for b, (_, ell) in enumerate(in1)]

        # NKI plan: (Dout, then per path x_off, d1, q_off, d3, out_off, accumulate).
        plan = [self.Dout]
        seen_io = set()
        for p, (i1, _i2, io, _alpha, l1, _l2, l3) in enumerate(self.paths):
            plan += [
                off1[i1],
                2 * l1 + 1,
                self.q_off[p],
                2 * l3 + 1,
                offo[io],
                1 if io in seen_io else 0,
            ]
            seen_io.add(io)
        self.plan = tuple(plan)
        self.zero_fill = int(not bool((covered > 0).all()))

    def materialise(self, device, dtype) -> dict:
        dev = torch.device(device)
        cg = self.cg.to(dev, dtype)
        cgk = self.cg_kernel.to(dev, dtype)
        sp = torch.zeros(self.K, self.P, dtype=torch.float64)
        sp[torch.arange(self.K), self.w_kmap] = 1.0
        return {
            "cg": cg,
            "cgT": cg.T.contiguous(),
            "cg_kernel": cgk,
            "cg_kernel_T": cgk.T.contiguous(),
            "x_perm": self.x_perm.to(dev),
            "x_perm_inv": self.x_perm_inv.to(dev),
            "w_kmap": self.w_kmap.to(dev),
            "w_reduce": sp.to(dev, dtype),
            "out_dst": self.out_dst.to(dev),
            "out_gather": None if self.out_gather is None else self.out_gather.to(dev),
        }


def _w3j_lookup(w3j, l1, l2, l3) -> torch.Tensor:
    if w3j is not None:
        c = w3j.get(f"{l1},{l2},{l3}") if isinstance(w3j, Mapping) else None
        if c is None and isinstance(w3j, Mapping):
            c = w3j.get((l1, l2, l3))
        if c is not None:
            return torch.as_tensor(c, dtype=torch.float64).cpu()
    return wigner_3j(l1, l2, l3)


def _e3nn_alpha(in1, in2, out, ins_all, ins, irrep_normalization, path_normalization, pw=1.0):
    """e3nn's per-path normalisation for 'uvu' (unit input/output variances)."""
    (_, l1), (m2, l2), (_, l3) = in1[ins[0]], in2[ins[1]], out[ins[2]]
    a = {"component": 2 * l3 + 1, "norm": (2 * l1 + 1) * (2 * l2 + 1), "none": 1}[
        irrep_normalization
    ]
    if path_normalization == "element":
        x = sum(in2[i[1]][0] for i in ins_all if i[2] == ins[2])
    elif path_normalization == "path":
        x = m2 * len([i for i in ins_all if i[2] == ins[2]])
    elif path_normalization == "none":
        x = 1
    else:
        raise ValueError(f"path_normalization {path_normalization!r}")
    if x > 0:
        a /= x
    return math.sqrt(a * pw)


def conv_tp_spec(
    irreps_in1=None,
    irreps_in2=None,
    irreps_out=None,
    instructions=None,
    *,
    module=None,
    w3j: Mapping | None = None,
    irrep_normalization: str = "component",
    path_normalization: str = "element",
) -> ConvTPSpec:
    """Build a :class:`ConvTPSpec`.

    From a module: ``conv_tp_spec(module=tp)`` with an e3nn ``TensorProduct``
    (or a MACE interaction block, whose ``conv_tp`` is used) reads the irreps,
    the instruction order and e3nn's normalisation (``path_weight``). e3nn is
    imported only for its ``wigner_3j`` when it is already installed.

    From explicit irreps: ``irreps_*`` as e3nn strings (``"128x0e+128x1o"``) or
    ``[(mul, l), ...]``; ``instructions`` as ``(i1, i2, io)`` (α from e3nn's
    "uvu" normalisation with ``irrep_normalization`` / ``path_normalization``,
    e3nn's defaults) or ``(i1, i2, io, α)``. 5-tuples in e3nn's
    ``(i1, i2, io, "uvu", True[, path_weight])`` form are accepted too.

    ``w3j`` overrides the built-in Wigner-3j constants (keys ``"l1,l2,l3"``).
    """
    if module is not None:
        tp = getattr(module, "conv_tp", module)
        in1, in2, out = (
            _parse_irreps(tp.irreps_in1),
            _parse_irreps(tp.irreps_in2),
            _parse_irreps(tp.irreps_out),
        )
        ins = []
        for i in tp.instructions:
            if i.connection_mode != "uvu" or not i.has_weight:
                raise NotImplementedError(
                    f"conv_tp: only weighted 'uvu' instructions, got {i.connection_mode!r}"
                )
            ins.append((int(i.i_in1), int(i.i_in2), int(i.i_out), float(i.path_weight)))
        if getattr(tp, "shared_weights", False):
            raise NotImplementedError("conv_tp: per-edge weights only (shared_weights=False)")
        if w3j is None:
            w3j = _e3nn_w3j(in1, in2, out, ins)
        return ConvTPSpec(in1, in2, out, tuple(ins), _freeze_w3j(w3j, in1, in2, out, ins))
    in1, in2, out = (_parse_irreps(v) for v in (irreps_in1, irreps_in2, irreps_out))
    if instructions is None:
        raise ValueError("conv_tp_spec: pass instructions (or module=)")
    raw = []
    for ins in instructions:
        ins = tuple(ins)
        if len(ins) >= 5:  # e3nn (i1, i2, io, mode, has_weight[, path_weight])
            if ins[3] != "uvu" or not ins[4]:
                raise NotImplementedError(f"conv_tp: only weighted 'uvu' instructions: {ins}")
            raw.append(
                (
                    int(ins[0]),
                    int(ins[1]),
                    int(ins[2]),
                    None,
                    float(ins[5]) if len(ins) > 5 else 1.0,
                )
            )
        elif len(ins) == 4:
            raw.append((int(ins[0]), int(ins[1]), int(ins[2]), float(ins[3]), None))
        elif len(ins) == 3:
            raw.append((int(ins[0]), int(ins[1]), int(ins[2]), None, 1.0))
        else:
            raise ValueError(f"conv_tp_spec: bad instruction {ins}")
    tri = [r[:3] for r in raw]
    ins = []
    for r in raw:
        alpha = r[3]
        if alpha is None:
            alpha = _e3nn_alpha(
                in1, in2, out, tri, r[:3], irrep_normalization, path_normalization, r[4]
            )
        ins.append((r[0], r[1], r[2], alpha))
    return ConvTPSpec(in1, in2, out, tuple(ins), _freeze_w3j(w3j, in1, in2, out, ins))


def _e3nn_w3j(in1, in2, out, ins):
    try:
        from e3nn import o3
    except Exception:  # not installed, or not importable in this torch
        return None
    w = {}
    for i1, i2, io, _ in ins:
        l1, l2, l3 = in1[i1][1], in2[i2][1], out[io][1]
        w[f"{l1},{l2},{l3}"] = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)
    return w


def _freeze_w3j(w3j, in1, in2, out, ins) -> dict:
    frozen = {}
    for i1, i2, io, _ in ins:
        l1, l2, l3 = in1[i1][1], in2[i2][1], out[io][1]
        frozen[f"{l1},{l2},{l3}"] = _w3j_lookup(w3j, l1, l2, l3)
    return frozen


# ─────────────────────────────────────────────── A2: symmetric contraction spec


@dataclass(frozen=True, eq=False)
class SymmetricContractionSpec:
    """Static description of a MACE ``SymmetricContraction``.

    ``contractions`` holds one entry per output irrep, in output order: a dict
    with ``irrep_out`` (e.g. ``"0e"``), ``d_out`` (``2l+1``), ``correlation``,
    ``U`` (``{ν: tensor [d_out, L, ..., L, K_ν]}``, dense, ν copies of L) and
    ``W`` (``{ν: tensor [num_elements, K_ν, C]}``). Build with
    :func:`symmetric_contraction_spec`.

    Derived constants (``layout``), built once in float64:

    - the *symmetric monomial basis*: every product ``x_{a1} … x_{aν}`` with
      ``a1 ≤ … ≤ aν`` that has a non-zero coupling, across all ν and output
      irreps (``A`` monomials; 410 of the 968 at MACE-MP layer 0, 94 at layer 1);
    - ``usym [A, MK]``: the coupling tensors folded onto that basis (summed over
      index orderings), columns ``(irrep, m, ν, k)``;
    - ``W_mk [Z, C, MK]`` (and ``[Z, MK, C]`` for the kernels): the per-element
      weights on the same columns, replicated over ``m``;
    - ``S [MK, Σ d_out]``: the 0/1 reduction over ``(ν, k)``.

    Per row ``r = (n, c)``: ``out_r = ((mono_r @ usym) ⊙ W_mk[z_n, c]) @ S``.
    """

    contractions: tuple[dict, ...] = field(repr=False)
    channels: int = 0
    num_features: int = 0
    num_elements: int = 0

    def __post_init__(self):
        object.__setattr__(self, "_layout", _SymConLayout(self))
        object.__setattr__(self, "_consts", {})
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def layout(self) -> _SymConLayout:
        return self._layout

    @property
    def dim_out(self) -> int:
        return self._layout.Dout

    @classmethod
    def from_dict(cls, d: Mapping) -> SymmetricContractionSpec:
        """Spec from the exemplar's dict format (``contractions`` with
        ``d_out``, ``correlation``, ``U`` (dense or COO) and ``W``)."""
        cons = []
        for con in d["contractions"]:
            cons.append(
                {
                    "irrep_out": con.get("irrep_out", ""),
                    "d_out": int(con["d_out"]),
                    "correlation": int(con["correlation"]),
                    "U": {int(k): _dense_u(v) for k, v in con["U"].items()},
                    "W": {int(k): torch.as_tensor(v) for k, v in con["W"].items()},
                }
            )
        return symmetric_contraction_spec(contractions=cons)

    def constants(self, device, dtype) -> dict:
        key = (torch.device(device), dtype)
        with self._lock:
            c = self._consts.get(key)
        if c is None:
            c = self._layout.materialise(device, dtype)
            with self._lock:
                self._consts[key] = c
        return c


def _dense_u(u) -> torch.Tensor:
    """U stored dense, or as COO ``{"shape", "indices", "values"}``."""
    if torch.is_tensor(u):
        return u
    dense = torch.zeros(list(u["shape"]), dtype=u["values"].dtype)
    dense[tuple(u["indices"].long())] = u["values"]
    return dense


class _SymConLayout:
    """Monomial basis and folded constants of a :class:`SymmetricContractionSpec`."""

    def __init__(self, spec: SymmetricContractionSpec):
        cons = spec.contractions
        if not cons:
            raise ValueError("symmetric contraction spec has no contractions")
        L, C, Z = spec.num_features, spec.channels, spec.num_elements
        self.L, self.C, self.Z = L, C, Z
        # Columns (irrep, m, nu, k) and the monomials each one touches.
        cols = []  # (con index, m, nu, k)
        self.d_off, d = [], 0
        for ci, con in enumerate(cons):
            self.d_off.append(d)
            d += con["d_out"]
            for m in range(con["d_out"]):
                for nu in range(1, con["correlation"] + 1):
                    K = con["U"][nu].shape[-1]
                    cols.extend((ci, m, nu, k) for k in range(K))
        self.Dsum = d
        self.MK = len(cols)
        # Fold each U_nu onto sorted monomials: Usym[alpha, col] = sum over the
        # distinct orderings of alpha of U[m, ordering, k].
        entries: dict[tuple[int, ...], dict[int, float]] = {}
        col_index = {c: j for j, c in enumerate(cols)}
        for ci, con in enumerate(cons):
            dd = con["d_out"]
            for nu in range(1, con["correlation"] + 1):
                U = con["U"][nu].to(torch.float64)
                U = U.reshape(dd, *([L] * nu), U.shape[-1])
                nz = U.nonzero(as_tuple=False)
                vals = U[tuple(nz.T)]
                for idx, v in zip(nz.tolist(), vals.tolist(), strict=True):
                    m, a, k = idx[0], tuple(sorted(idx[1:-1])), idx[-1]
                    row = entries.setdefault(a, {})
                    j = col_index[(ci, m, nu, k)]
                    row[j] = row.get(j, 0.0) + v
        monos = sorted(
            (a for a, row in entries.items() if any(abs(v) > 0 for v in row.values())),
            key=lambda a: (len(a), a),
        )
        self.monomials = monos
        self.A = len(monos)
        usym = torch.zeros(self.A, self.MK, dtype=torch.float64)
        for r, a in enumerate(monos):
            for j, v in entries[a].items():
                usym[r, j] = v
        self.usym = usym
        # Factor j of monomial alpha as a column of x_aug = [x, 1] (index L = the ones column).
        fac = torch.full((3, self.A), L, dtype=torch.long)
        for r, a in enumerate(monos):
            for j, v in enumerate(a):
                fac[j, r] = v
        if max(len(a) for a in monos) > 3:
            raise NotImplementedError("symmetric_contraction: correlation order > 3")
        self.factors = fac
        S = torch.zeros(self.MK, self.Dsum, dtype=torch.float64)
        for j, (ci, m, _nu, _k) in enumerate(cols):
            S[j, self.d_off[ci] + m] = 1.0
        self.S = S
        W = torch.zeros(Z, C, self.MK, dtype=torch.float64)
        for j, (ci, _m, nu, k) in enumerate(cols):
            W[:, :, j] = cons[ci]["W"][nu][:, k, :].to(torch.float64)
        self.W_zcm = W
        # out[n, off_c + ch * d_c + m]  <-  o[n, ch * Dsum + d_off_c + m]
        offs, o = [], 0
        for con in cons:
            offs.append(o)
            o += C * con["d_out"]
        self.Dout = o
        perm = torch.empty(self.Dout, dtype=torch.long)
        for ci, con in enumerate(cons):
            dc = con["d_out"]
            for ch in range(C):
                for m in range(dc):
                    perm[offs[ci] + ch * dc + m] = ch * self.Dsum + self.d_off[ci] + m
        self.out_gather = perm
        self.out_gather_inv = torch.argsort(perm)
        # Selection matrices for the kernels: sel[a, j, alpha] = [factor j of alpha == a].
        sel = torch.zeros(L + 1, 3, self.A, dtype=torch.float64)
        for j in range(3):
            sel[fac[j], j, torch.arange(self.A)] = 1.0
        self.sel = sel

    def materialise(self, device, dtype) -> dict:
        dev = torch.device(device)
        usym = self.usym.to(dev, dtype)
        S = self.S.to(dev, dtype)
        W = self.W_zcm.to(dev, dtype)
        sel = self.sel.to(dev, dtype)
        return {
            "usym": usym,
            "usymT": usym.T.contiguous(),
            "S": S,
            "ST": S.T.contiguous(),
            "W_zcm": W,
            "W_zmc": W.transpose(1, 2).contiguous(),
            "factors": self.factors.to(dev),
            "out_gather": self.out_gather.to(dev),
            "out_gather_inv": self.out_gather_inv.to(dev),
            "sel": sel,  # [L+1, 3, A]
            "selT": sel.permute(2, 1, 0).contiguous(),  # [A, 3, L+1]
        }


def symmetric_contraction_spec(
    *, module=None, contractions: Sequence[Mapping] | None = None
) -> SymmetricContractionSpec:
    """Build a :class:`SymmetricContractionSpec`.

    From a module: ``symmetric_contraction_spec(module=sc)`` with a MACE
    ``SymmetricContraction`` (or a product block, whose
    ``symmetric_contractions`` is used) reads each ``Contraction``'s coupling
    buffers ``U_matrix_ν`` and weights (``weights_max`` is ν = correlation,
    ``weights[j]`` is ν = correlation − 1 − j). MACE itself is not imported.

    Explicit: ``contractions`` as a list, one per output irrep, of mappings
    with ``d_out``, ``correlation``, ``U`` ``{ν: [d_out (dropped when 1), L..., K]}``
    and ``W`` ``{ν: [num_elements, K, C]}``; ``irrep_out`` is optional.
    """
    if module is not None:
        sc = getattr(module, "symmetric_contractions", module)
        cons = []
        for con in sc.contractions:
            corr = int(con.correlation)
            U = {nu: con.U_tensors(nu).detach().clone() for nu in range(1, corr + 1)}
            W = {corr: con.weights_max.detach().clone()}
            for j, w in enumerate(con.weights):
                W[corr - 1 - j] = w.detach().clone()
            irr = None
            d = U[1].shape[0] if U[1].dim() == 3 else 1
            cons.append({"irrep_out": irr or "", "d_out": d, "correlation": corr, "U": U, "W": W})
        irreps_out = getattr(sc, "irreps_out", None)
        if irreps_out is not None:
            for con, ir in zip(cons, irreps_out, strict=False):
                con["irrep_out"] = str(getattr(ir, "ir", ir))
        contractions = cons
    if contractions is None:
        raise ValueError("symmetric_contraction_spec: pass contractions= or module=")
    cons = []
    L = C = Z = None
    for con in contractions:
        d = int(con["d_out"])
        corr = int(con["correlation"])
        Us, Ws = {}, {}
        for nu in range(1, corr + 1):
            u = con["U"].get(nu, con["U"].get(str(nu)))
            w = con["W"].get(nu, con["W"].get(str(nu)))
            if u is None or w is None:
                raise ValueError(f"symmetric_contraction_spec: missing U/W for nu={nu}")
            u = _dense_u(u).detach().to("cpu", torch.float64)
            if d == 1 and u.dim() == nu + 1:
                u = u.unsqueeze(0)
            if u.dim() != nu + 2 or u.shape[0] != d:
                raise ValueError(f"U_{nu}: shape {tuple(u.shape)} is not [d_out, L x {nu}, K]")
            w = torch.as_tensor(w).detach().to("cpu", torch.float64)
            if w.dim() != 3 or w.shape[1] != u.shape[-1]:
                raise ValueError(f"W_{nu}: shape {tuple(w.shape)} is not [Z, K={u.shape[-1]}, C]")
            L = u.shape[1] if L is None else L
            Z = w.shape[0] if Z is None else Z
            C = w.shape[2] if C is None else C
            if u.shape[1] != L or w.shape[0] != Z or w.shape[2] != C:
                raise ValueError("symmetric_contraction_spec: inconsistent L / Z / C")
            Us[nu], Ws[nu] = u, w
        cons.append(
            {
                "irrep_out": str(con.get("irrep_out", "")),
                "d_out": d,
                "correlation": corr,
                "U": Us,
                "W": Ws,
            }
        )
    return SymmetricContractionSpec(tuple(cons), channels=C, num_features=L, num_elements=Z)


# ─────────────────────────────────────────────────────────────── spec coercion

_dict_specs: dict[int, tuple[Mapping, object]] = {}


def _as_spec(spec, kind):
    """A spec object, converting (and caching) the exemplar's dict format.

    Dict specs are cached by identity; the cache holds a reference to the
    dict, so its id cannot be reused while the entry lives.
    """
    if isinstance(spec, kind):
        return spec
    if isinstance(spec, Mapping):
        hit = _dict_specs.get(id(spec))
        if hit is not None and hit[0] is spec:
            return hit[1]
        built = kind.from_dict(spec)
        with _cache_lock:
            if len(_dict_specs) > 64:
                _dict_specs.clear()
            _dict_specs[id(spec)] = (spec, built)
        return built
    raise TypeError(f"expected a {kind.__name__} or its dict form, got {type(spec).__name__}")


# ─────────────────────────────────────────────────────────────── A1: PyTorch path


def _conv_tp_xcat(x, lay):
    """``x_cat [E, U, Dcat]``: every in1 component of channel u (one cat of views)."""
    E, U = x.shape[0], lay.U
    parts = [x[:, o : o + U * d].view(E, U, d) for o, _c, d in lay.x_blocks]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)


def _conv_tp_raw(xc, G, lay):
    """``t [E, U, K] = x_cat @ G`` (per edge); an outer product when Dcat = 1."""
    return xc * G if lay.Dcat == 1 else torch.bmm(xc, G)


def _conv_tp_forward_torch(x, y, w, spec: ConvTPSpec) -> torch.Tensor:
    lay = spec.layout
    c = spec.constants(x.device, x.dtype)
    E, U = x.shape[0], lay.U
    xc = _conv_tp_xcat(x, lay)
    G = (y @ c["cg"]).view(E, lay.Dcat, lay.K)
    t = _conv_tp_raw(xc, G, lay)
    if not lay.runs:  # general spec: shared or uncovered output blocks
        wx = w.view(E, lay.P, U).transpose(1, 2).index_select(2, c["w_kmap"])
        flat = (t * wx).reshape(E, U * lay.K)
        if lay.out_bijective:
            return flat.index_select(1, c["out_gather"])
        return flat.new_zeros(E, lay.Dout).index_add_(1, c["out_dst"], flat)
    out = x.new_empty(E, lay.Dout)
    wv = w.view(E, lay.P, U)
    for p0, n, d3, k0, o0 in lay.runs:
        tv = t[:, :, k0 : k0 + n * d3].view(E, U, n, d3).transpose(1, 2)
        ov = out[:, o0 : o0 + n * U * d3].view(E, n, U, d3)
        torch.mul(tv, wv[:, p0 : p0 + n, :, None], out=ov)
    return out


def _conv_tp_backward_torch(g, x, y, w, spec: ConvTPSpec, need=(True, True, True)):
    """``(∂x, ∂y, ∂w)`` for cotangent ``g`` (None where not needed).

    Recomputes the forward intermediates instead of saving them: the raw
    product is as large as the output (20 KiB per edge at MACE-MP layer 1).
    Every op is differentiable, so double backward works through autograd.
    """
    lay = spec.layout
    c = spec.constants(g.device, g.dtype)
    E, U = g.shape[0], lay.U
    xc = _conv_tp_xcat(x, lay)
    G = (y @ c["cg"]).view(E, lay.Dcat, lay.K)
    dx = dy = dw = None
    if lay.runs:
        wv = w.view(E, lay.P, U)
        gr = [g[:, o0 : o0 + n * U * d3].view(E, n, U, d3) for _p, n, d3, _k, o0 in lay.runs]
        if need[0] or need[1]:
            dts = [
                (gv * wv[:, p0 : p0 + n, :, None]).transpose(1, 2).reshape(E, U, n * d3)
                for gv, (p0, n, d3, _k, _o) in zip(gr, lay.runs, strict=True)
            ]
            dt = dts[0] if len(dts) == 1 else torch.cat(dts, dim=2)
        if need[2]:
            t = _conv_tp_raw(xc, G, lay)
            dws = [
                (gv * t[:, :, k0 : k0 + n * d3].view(E, U, n, d3).transpose(1, 2)).sum(-1)
                for gv, (_p, n, d3, k0, _o) in zip(gr, lay.runs, strict=True)
            ]
            dw = (dws[0] if len(dws) == 1 else torch.cat(dws, dim=1)).reshape(E, lay.P * U)
    else:
        wx = w.view(E, lay.P, U).transpose(1, 2).index_select(2, c["w_kmap"])
        gt = g.index_select(1, c["out_dst"]).view(E, U, lay.K)
        dt = gt * wx
        if need[2]:
            t = _conv_tp_raw(xc, G, lay)
            dw = ((gt * t) @ c["w_reduce"]).transpose(1, 2).reshape(E, lay.P * U)
    if need[0]:
        dxc = torch.bmm(dt, G.transpose(1, 2))
        parts = [dxc[:, :, cb : cb + d].reshape(E, U * d) for _o, cb, d in lay.x_blocks]
        dx = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    if need[1]:
        dG = torch.bmm(xc.transpose(1, 2), dt)
        dy = dG.reshape(E, lay.Dcat * lay.K) @ c["cgT"]
    return dx, dy, dw


class _ConvTPTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, w, spec):
        ctx.spec = spec
        ctx.save_for_backward(x, y, w)
        return _conv_tp_forward_torch(x, y, w, spec)

    @staticmethod
    def backward(ctx, g):
        x, y, w = ctx.saved_tensors
        need = ctx.needs_input_grad[:3]
        dx, dy, dw = _conv_tp_backward_torch(g.contiguous(), x, y, w, ctx.spec, need)
        return dx, dy, dw, None


# ─────────────────────────────────────────────────────────────── A2: PyTorch path

# Rows (node, channel) per chunk of the PyTorch path: bounds the
# [rows, monomials] intermediate (2^25 fp32 = 128 MiB) at large N.
_SYMCON_CHUNK_ELEMS = 1 << 25


def _monomials(xr, c):
    xa = torch.cat([xr, xr.new_ones(xr.shape[0], 1)], dim=1)
    f = c["factors"]
    x1, x2, x3 = (xa.index_select(1, f[j]) for j in range(3))
    return x1, x2, x3


def _symcon_rows_forward(xr, wr, c):
    x1, x2, x3 = _monomials(xr, c)
    return ((x1 * x2 * x3) @ c["usym"] * wr) @ c["S"]


def _symcon_rows_backward(go, xr, wr, c, L):
    x1, x2, x3 = _monomials(xr, c)
    gm = ((go @ c["ST"]) * wr) @ c["usymT"]
    f = c["factors"]
    dxa = xr.new_zeros(xr.shape[0], L + 1)
    dxa.index_add_(1, f[0], gm * x2 * x3)
    dxa.index_add_(1, f[1], gm * x1 * x3)
    dxa.index_add_(1, f[2], gm * x1 * x2)
    return dxa[:, :L]


def _node_chunks(N, C, width):
    step = max(1, _SYMCON_CHUNK_ELEMS // max(1, C * width))
    return [(n0, min(N, n0 + step)) for n0 in range(0, N, step)]


def _symcon_forward_torch(x, species, spec: SymmetricContractionSpec) -> torch.Tensor:
    lay = spec.layout
    c = spec.constants(x.device, x.dtype)
    N, C, L = x.shape
    parts = []
    for n0, n1 in _node_chunks(N, C, max(lay.A, lay.MK)):
        xr = x[n0:n1].reshape(-1, L)
        wr = c["W_zcm"].index_select(0, species[n0:n1]).reshape(-1, lay.MK)
        o = _symcon_rows_forward(xr, wr, c)
        parts.append(o.reshape(n1 - n0, C * lay.Dsum).index_select(1, c["out_gather"]))
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def _symcon_backward_torch(g, x, species, spec: SymmetricContractionSpec) -> torch.Tensor:
    lay = spec.layout
    c = spec.constants(g.device, g.dtype)
    N, C, L = x.shape
    parts = []
    for n0, n1 in _node_chunks(N, C, max(lay.A, lay.MK)):
        go = g[n0:n1].index_select(1, c["out_gather_inv"]).reshape(-1, lay.Dsum)
        xr = x[n0:n1].reshape(-1, L)
        wr = c["W_zcm"].index_select(0, species[n0:n1]).reshape(-1, lay.MK)
        parts.append(_symcon_rows_backward(go, xr, wr, c, L).reshape(n1 - n0, C, L))
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


class _SymConTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, species, spec):
        ctx.spec = spec
        ctx.save_for_backward(x, species)
        return _symcon_forward_torch(x, species, spec)

    @staticmethod
    def backward(ctx, g):
        x, species = ctx.saved_tensors
        dx = _symcon_backward_torch(g.contiguous(), x, species, ctx.spec)
        return dx, None, None


# ─────────────────────────────────────────────────────────────── public entry points

EQUIVARIANT_NKI_ENV = "TRNTENSOR_EQUIVARIANT_NKI"


def _auto_uses_kernels() -> bool:
    return os.environ.get(EQUIVARIANT_NKI_ENV, "").strip().lower() in ("1", "true", "yes")


def _check_conv_tp_args(x, y, w, spec: ConvTPSpec):
    if x.dim() != 2 or y.dim() != 2 or w.dim() != 2:
        raise ValueError("conv_tp: x, y and w must be 2-D [E, features]")
    E = x.shape[0]
    if y.shape[0] != E or w.shape[0] != E:
        raise ValueError(
            f"conv_tp: edge counts differ: x {tuple(x.shape)}, y {tuple(y.shape)}, w {tuple(w.shape)}"
        )
    for name, t, want in (
        ("x", x, spec.dim_in1),
        ("y", y, spec.dim_in2),
        ("w", w, spec.weight_numel),
    ):
        if t.shape[1] != want:
            raise ValueError(f"conv_tp: {name} has {t.shape[1]} features, spec wants {want}")
    if not (x.dtype == y.dtype == w.dtype) or not x.dtype.is_floating_point:
        raise TypeError(
            f"conv_tp: x, y, w must share one floating dtype, got {x.dtype}, {y.dtype}, {w.dtype}"
        )


@_fp64_host_guard("conv_tp")
def conv_tp(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, spec) -> torch.Tensor:
    """Equivariant interaction tensor product (e3nn "uvu", per-edge weights).

    ``x [E, dim(irreps_in1)]`` gathered sender features, ``y [E, dim(irreps_in2)]``
    spherical harmonics, ``w [E, weight_numel]`` per-edge path weights; returns
    ``[E, dim(irreps_out)]`` in e3nn's mul-major layout. ``spec`` is a
    :class:`ConvTPSpec` (or the exemplar's dict form). Differentiable in
    ``x``, ``y`` and ``w``.

    MACE-MP layer 0: ``x [E, 128]``, ``y [E, 16]``, ``w [E, 512]`` → ``[E, 2048]``;
    layer 1: ``x [E, 512]``, ``w [E, 1280]`` → ``[E, 5120]``.
    """
    spec = _as_spec(spec, ConvTPSpec)
    _check_conv_tp_args(x, y, w, spec)
    from .nki import _equivariant as _eq

    if _eq.conv_tp_kernel_route(x, y, w, spec):
        return _eq.conv_tp_nki(x, y, w, spec)
    return _ConvTPTorch.apply(x, y, w, spec)


def _check_symcon_args(x, species, spec: SymmetricContractionSpec):
    if x.dim() != 3:
        raise ValueError("symmetric_contraction: x must be [N, C, L]")
    N, C, L = x.shape
    if (spec.channels, spec.num_features) != (C, L):
        raise ValueError(
            f"symmetric_contraction: x is [N, {C}, {L}], spec wants [N, {spec.channels}, {spec.num_features}]"
        )
    if species.dim() != 1 or species.shape[0] != N:
        raise ValueError("symmetric_contraction: species must be [N]")
    if species.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"symmetric_contraction: species must be int64/int32, got {species.dtype}")
    if not x.dtype.is_floating_point:
        raise TypeError(f"symmetric_contraction: x must be floating, got {x.dtype}")


@_fp64_host_guard("symmetric_contraction")
def symmetric_contraction(x: torch.Tensor, species: torch.Tensor, spec) -> torch.Tensor:
    """MACE symmetric contraction (product basis, correlation ν ≤ 3).

    ``x [N, C, L]`` node features (``L = 16`` for 0e+1o+2e+3o), ``species [N]``
    element indices into the spec's weight tables; returns
    ``[N, Σ C·d_out]``, output irreps in spec order, each block ``[C, d_out]``
    row-major (MACE's ``out.view(N, -1)``). Differentiable in ``x``.
    """
    spec = _as_spec(spec, SymmetricContractionSpec)
    _check_symcon_args(x, species, spec)
    species = species.to(torch.int64)
    from .nki import _equivariant as _eq

    if _eq.symcon_kernel_route(x, species, spec):
        return _eq.symmetric_contraction_nki(x, species, spec)
    return _SymConTorch.apply(x, species, spec)
