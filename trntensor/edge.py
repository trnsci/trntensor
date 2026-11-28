"""Per-edge ops of a MACE-style interatomic potential (trnsci/trnsci#178).

Everything MACE computes from the edge vectors before the interaction tensor
product is a short per-edge chain of small elementwise ops and tiny
matmuls. On the MACE-MP-0b2 medium step on Trn2 (216 atoms, 10,392 edges)
that chain is about 60 ms of the 118 ms step, in about 560 launches, because
each op is its own launch. This module provides the chain as two fused ops,
each with an analytic backward (forces are ``-dE/dpositions``, so the input
gradients are needed):

``spherical_harmonics(vectors, lmax=3)``
    e3nn's real spherical harmonics (``o3.SphericalHarmonics(...,
    normalize=True, normalization="component")``, the MACE edge attributes):
    the same basis, ordering and normalisation for ``l ≤ 3``. Every component
    is a polynomial of degree ``≤ l`` in the unit vector, so the forward is
    one matmul of the degree-≤3 monomials against a fixed coefficient matrix,
    and the backward one matmul against its derivative tables.

``edge_chain(lengths, spec, r0=None)``
    The radial basis (optional Agnesi distance transform, Bessel basis,
    polynomial cutoff) and every radial MLP that reads it, fused: the
    ``conv_tp`` weights of each interaction and the edge densities
    ``tanh(density_fn(·)²)``. A spec is read once from the model
    (:func:`edge_chain_spec`); the per-edge species constant of the Agnesi
    transform (``r0``) depends only on the neighbour list and is an input.

Backends follow ``trntensor.set_backend``: the PyTorch path (the default
everywhere; one autograd node per op, its backward a fixed handful of aten
ops) or, with ``set_backend("nki")``, the NKI kernels in
``trntensor/nki/_edge_kernels.py`` (one launch per direction). ``"auto"``
keeps the PyTorch path until the kernels are validated on a device, unless
``TRNTENSOR_EDGE_NKI=1``. Under a region backend
(:func:`trntensor.set_region_backend`) the PyTorch bodies each run as one
compiled graph per direction.

MACE and e3nn are optional: specs can be built from explicit constants.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ._region import region
from .nki.dispatch import _fp64_host_guard

__all__ = [
    "EdgeChainSpec",
    "EdgeMLP",
    "edge_chain",
    "edge_chain_spec",
    "spherical_harmonics",
    "sh_coefficients",
]

SH_LMAX = 3

# ─────────────────────────────────────────────────────── spherical harmonics


class _Poly:
    """A polynomial in (x, y, z) as {(a, b, c): coefficient} (float64)."""

    __slots__ = ("t",)

    def __init__(self, t=None):
        self.t = dict(t or {})

    @classmethod
    def var(cls, i):
        e = [0, 0, 0]
        e[i] = 1
        return cls({tuple(e): 1.0})

    def __add__(self, o):
        t = dict(self.t)
        for k, v in o.t.items():
            t[k] = t.get(k, 0.0) + v
        return _Poly(t)

    def __sub__(self, o):
        return self + o * -1.0

    def __mul__(self, o):
        if isinstance(o, (int, float)):
            return _Poly({k: v * o for k, v in self.t.items()})
        t = {}
        for k1, v1 in self.t.items():
            for k2, v2 in o.t.items():
                k = (k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2])
                t[k] = t.get(k, 0.0) + v1 * v2
        return _Poly(t)

    __rmul__ = __mul__

    def d(self, i):
        t = {}
        for k, v in self.t.items():
            if k[i]:
                kk = list(k)
                kk[i] -= 1
                t[tuple(kk)] = t.get(tuple(kk), 0.0) + v * k[i]
        return _Poly(t)


def _e3nn_sh_polys(lmax: int) -> list[_Poly]:
    """e3nn 0.4's ``_spherical_harmonics`` (before normalisation), as polynomials."""
    x, y, z = (_Poly.var(i) for i in range(3))
    one = _Poly({(0, 0, 0): 1.0})
    out = [one]
    if lmax >= 1:
        out += [x, y, z]
    if lmax >= 2:
        s3 = math.sqrt(3.0)
        sh20, sh21 = s3 * (x * z), s3 * (x * y)
        y2, x2z2 = y * y, x * x + z * z
        sh22 = y2 - 0.5 * x2z2
        sh23, sh24 = s3 * (y * z), (s3 / 2.0) * (z * z - x * x)
        out += [sh20, sh21, sh22, sh23, sh24]
    if lmax >= 3:
        out += [
            math.sqrt(5.0 / 6.0) * (sh20 * z + sh24 * x),
            math.sqrt(5.0) * (sh20 * y),
            math.sqrt(3.0 / 8.0) * ((4.0 * y2 - x2z2) * x),
            0.5 * (y * (2.0 * y2 - 3.0 * x2z2)),
            math.sqrt(3.0 / 8.0) * (z * (4.0 * y2 - x2z2)),
            math.sqrt(5.0) * (sh24 * y),
            math.sqrt(5.0 / 6.0) * (sh24 * z - sh20 * x),
        ]
    return out


def _monomial_exps(deg: int) -> list[tuple[int, int, int]]:
    """Monomials of degree 1..deg in the order :func:`_monomials` builds them."""
    exps = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if deg >= 2:
        exps += [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    if deg >= 3:
        exps += [
            (3, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (1, 1, 1),
            (1, 0, 2),
            (0, 3, 0),
            (0, 2, 1),
            (0, 1, 2),
            (0, 0, 3),
        ]
    return exps[: {1: 3, 2: 9, 3: 19}[deg]]


# Degree-2 monomials from the outer product n ⊗ n (row-major 3x3): xx xy xz yy yz zz.
_I2 = (0, 1, 2, 4, 5, 8)
# Degree-3 monomials from (degree-2) ⊗ n (row-major 6x3).
_I3 = (0, 1, 2, 4, 5, 8, 10, 11, 14, 17)

_NORMS = ("component", "integral", "norm")


def _norm_scale(ell: int, normalization: str) -> float:
    if normalization == "component":
        return math.sqrt(2 * ell + 1)
    if normalization == "integral":
        return math.sqrt(2 * ell + 1) / math.sqrt(4 * math.pi)
    return 1.0


_sh_cache: dict = {}
_sh_lock = threading.Lock()


def sh_coefficients(lmax: int = 3, normalization: str = "component") -> dict:
    """The fixed tables of :func:`spherical_harmonics` (float64, host).

    ``c0 [D]`` and ``C [M, D]``: ``Y(n) = c0 + mono(n) @ C`` with ``mono`` the
    ``M`` monomials of degree 1..lmax (``M = 19`` at ``l ≤ 3``; order in
    ``exps``) and ``D = (lmax+1)²``. ``dC [D, 3, 1 + M']``: for each Cartesian
    direction ``c``, ``∂Y/∂n_c`` in the basis ``[1, monomials of degree
    1..lmax-1]`` (``M' = 9`` at ``l ≤ 3``).
    """
    lmax, key = int(lmax), (int(lmax), normalization)
    if not 0 <= lmax <= SH_LMAX:
        raise ValueError(f"spherical_harmonics: lmax must be in 0..{SH_LMAX}, got {lmax}")
    if normalization not in _NORMS:
        raise ValueError(f"spherical_harmonics: normalization must be one of {_NORMS}")
    with _sh_lock:
        hit = _sh_cache.get(key)
    if hit is not None:
        return hit
    polys = _e3nn_sh_polys(lmax)
    scale = [_norm_scale(ell, normalization) for ell in range(lmax + 1) for _ in range(2 * ell + 1)]
    polys = [p * s for p, s in zip(polys, scale, strict=True)]
    D = len(polys)
    exps = _monomial_exps(lmax) if lmax else []
    M = len(exps)
    col = {e: i for i, e in enumerate(exps)}
    c0 = torch.zeros(D, dtype=torch.float64)
    C = torch.zeros(max(M, 1), D, dtype=torch.float64)
    for k, p in enumerate(polys):
        for e, v in p.t.items():
            if e == (0, 0, 0):
                c0[k] += v
            else:
                C[col[e], k] += v
    low = [(0, 0, 0)] + (_monomial_exps(lmax - 1) if lmax >= 2 else [])
    lcol = {e: i for i, e in enumerate(low)}
    dC = torch.zeros(D, 3, len(low), dtype=torch.float64)
    for k, p in enumerate(polys):
        for c in range(3):
            for e, v in p.d(c).t.items():
                dC[k, c, lcol[e]] += v
    out = {"c0": c0, "C": C, "dC": dC, "exps": exps, "D": D, "M": M, "Mlow": len(low) - 1}
    with _sh_lock:
        _sh_cache[key] = out
    return out


_sh_dev_cache: dict = {}


def _sh_consts(lmax: int, normalization: str, device, dtype) -> dict:
    """``c0``, ``C`` and ``dC [D, 3·(1+M')]`` on ``device`` in ``dtype`` (cached)."""
    key = (int(lmax), normalization, torch.device(device), dtype)
    with _sh_lock:
        hit = _sh_dev_cache.get(key)
    if hit is None:
        t = sh_coefficients(lmax, normalization)
        hit = {
            "c0": t["c0"].to(device, dtype),
            "C": t["C"].to(device, dtype),
            "dC": t["dC"].reshape(t["D"], -1).to(device, dtype),
        }
        with _sh_lock:
            _sh_dev_cache[key] = hit
    return hit


def _monomials(n: torch.Tensor, deg: int) -> torch.Tensor:
    """``[E, M]``: the monomials of degree 1..deg of the rows of ``n [E, 3]``."""
    E = n.shape[0]
    parts = [n]
    if deg >= 2:
        m2 = (n[:, :, None] * n[:, None, :]).reshape(E, 9)[:, list(_I2)]
        parts.append(m2)
        if deg >= 3:
            m3 = (m2[:, :, None] * n[:, None, :]).reshape(E, 18)[:, list(_I3)]
            parts.append(m3)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def _unit(v, normalize: bool, eps: float):
    if not normalize:
        return v, None, None
    r = torch.linalg.vector_norm(v, dim=1, keepdim=True)
    rc = r.clamp_min(eps)
    return v / rc, r, rc


def _sh_forward_torch(v, c0, C, lmax: int, normalize: bool, eps: float):
    n, _r, _rc = _unit(v, normalize, eps)
    if lmax == 0:
        return c0.expand(v.shape[0], 1).clone()
    return torch.addmm(c0, _monomials(n, lmax), C)


def _sh_backward_torch(g, v, dC, lmax: int, normalize: bool, eps: float):
    """``∂/∂v`` for cotangent ``g [E, D]`` (differentiable aten)."""
    E = v.shape[0]
    n, r, rc = _unit(v, normalize, eps)
    if lmax == 0:
        return torch.zeros_like(v)
    h = (g @ dC).view(E, 3, -1)  # dC: [D, 3·(1 + M')]
    gn = h[:, :, 0]
    if lmax >= 2:
        low = _monomials(n, lmax - 1)
        gn = gn + (h[:, :, 1:] * low[:, None, :]).sum(-1)
    if not normalize:
        return gn
    # n = v / max(|v|, eps): (I - n nᵀ) / |v| above eps, I / eps below
    # (torch.nn.functional.normalize, which e3nn uses).
    proj = (gn * n).sum(1, keepdim=True) * n
    proj = torch.where(r >= eps, proj, torch.zeros_like(proj))
    return (gn - proj) / rc


class _SHTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, lmax, normalize, eps, normalization):
        c = _sh_consts(lmax, normalization, v.device, v.dtype)
        ctx.args = (lmax, normalize, eps, normalization)
        ctx.save_for_backward(v)
        site = (lmax, normalize, eps, normalization)
        return region(_sh_forward_torch, site)(v, c["c0"], c["C"], lmax, normalize, eps)

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        lmax, normalize, eps, normalization = ctx.args
        dC = _sh_consts(lmax, normalization, g.device, g.dtype)["dC"]
        site = (lmax, normalize, eps, normalization)
        dv = region(_sh_backward_torch, site)(g.contiguous(), v, dC, lmax, normalize, eps)
        return dv, None, None, None, None


@_fp64_host_guard("spherical_harmonics")
def spherical_harmonics(
    vectors: torch.Tensor,
    lmax: int = 3,
    normalize: bool = True,
    normalization: str = "component",
    eps: float = 1e-12,
) -> torch.Tensor:
    """e3nn real spherical harmonics ``l = 0..lmax`` (``lmax ≤ 3``) of ``vectors [..., 3]``.

    Matches ``e3nn.o3.SphericalHarmonics(range(lmax+1), normalize,
    normalization)`` (e3nn 0.4): the same component order (``l`` blocks of
    ``2l+1``, e3nn's real basis), the same normalisation, and
    ``normalize=True`` divides by ``max(|v|, eps)`` as
    ``torch.nn.functional.normalize`` does. Returns ``[..., (lmax+1)²]``.
    Differentiable in ``vectors`` (analytic backward).

    MACE edge attributes: ``spherical_harmonics(vectors)`` (``l ≤ 3``,
    component normalisation) → ``[E, 16]``.
    """
    if vectors.shape[-1] != 3:
        raise ValueError(
            f"spherical_harmonics: vectors must be [..., 3], got {tuple(vectors.shape)}"
        )
    if not vectors.dtype.is_floating_point:
        raise TypeError(f"spherical_harmonics: vectors must be floating, got {vectors.dtype}")
    sh_coefficients(lmax, normalization)  # validates
    lead = vectors.shape[:-1]
    v = vectors.reshape(-1, 3)
    from .nki import _edge as _ek

    if _ek.sh_kernel_route(v, lmax, normalize, normalization):
        out = _ek.spherical_harmonics_nki(v, lmax, normalize, normalization, eps)
    else:
        out = _SHTorch.apply(v, int(lmax), bool(normalize), float(eps), normalization)
    return out.reshape(*lead, out.shape[-1])


# ─────────────────────────────────────────────────────────────── edge chain

POSTS = (None, "tanh_sq")


@dataclass(frozen=True, eq=False)
class EdgeMLP:
    """One radial MLP head of an :class:`EdgeChainSpec`.

    ``weights``: the effective matrices ``W_1 … W_L`` (float64) with
    ``h_{i+1} = silu(h_i @ W_i)`` for the hidden layers and ``out = h_L @ W_L``;
    every normalisation (e3nn's ``1/√fan_in``, ``normalize2mom``'s activation
    constant) is already folded in. ``post``: ``None`` or ``"tanh_sq"``
    (``tanh(out²)``, MACE's edge density). ``name``: a label.
    """

    weights: tuple[torch.Tensor, ...]
    post: str | None = None
    name: str = ""

    def __post_init__(self):
        if self.post not in POSTS:
            raise ValueError(f"EdgeMLP: post must be one of {POSTS}, got {self.post!r}")
        ws = tuple(torch.as_tensor(w, dtype=torch.float64).detach().cpu() for w in self.weights)
        if not ws:
            raise ValueError("EdgeMLP: no weights")
        for a, b in zip(ws, ws[1:], strict=False):
            if a.shape[1] != b.shape[0]:
                raise ValueError(
                    f"EdgeMLP: shapes {tuple(a.shape)} and {tuple(b.shape)} do not chain"
                )
        object.__setattr__(self, "weights", ws)

    @property
    def dim_in(self) -> int:
        return int(self.weights[0].shape[0])

    @property
    def dim_out(self) -> int:
        return int(self.weights[-1].shape[1])


@dataclass(frozen=True, eq=False)
class EdgeChainSpec:
    """Static description of the fused per-edge chain.

    - radial basis ``B_n(x) = prefactor · sin(w_n x) / x`` (MACE ``BesselBasis``)
      of ``x = agnesi(r, r0)`` when ``agnesi = (a, q, p)`` (MACE
      ``AgnesiTransform``: ``x = 1 / (1 + a t^q / (1 + t^(q−p)))``,
      ``t = r / r0``), else of ``x = r``;
    - polynomial cutoff ``c(r)`` of order ``cutoff_p`` at ``r_max`` (MACE
      ``PolynomialCutoff``, zero at and beyond ``r_max``);
    - ``apply_cutoff``: the heads read ``B · c`` (MACE's default) or ``B``,
      with each head's output multiplied by ``c`` instead;
    - ``heads``: the radial MLPs (:class:`EdgeMLP`) that read the basis.

    ``covalent_radii`` (optional) is the Agnesi transform's table, for
    :meth:`agnesi_r0`. Build with :func:`edge_chain_spec`.
    """

    r_max: float
    bessel_weights: tuple[float, ...]
    prefactor: float
    cutoff_p: float
    heads: tuple[EdgeMLP, ...]
    agnesi: tuple[float, float, float] | None = None
    apply_cutoff: bool = True
    covalent_radii: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self):
        nb = len(self.bessel_weights)
        if nb == 0 or not self.heads:
            raise ValueError("EdgeChainSpec: needs a basis and at least one head")
        for h in self.heads:
            if h.dim_in != nb:
                raise ValueError(
                    f"EdgeChainSpec: head {h.name!r} reads {h.dim_in} features, basis has {nb}"
                )
        object.__setattr__(self, "_consts", {})
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def num_basis(self) -> int:
        return len(self.bessel_weights)

    @property
    def dims_out(self) -> tuple[int, ...]:
        return tuple(h.dim_out for h in self.heads)

    @property
    def static(self) -> tuple:
        """Hashable static description (what a compiled graph or kernel specialises on)."""
        return (
            float(self.r_max),
            float(self.prefactor),
            float(self.cutoff_p),
            None if self.agnesi is None else tuple(float(v) for v in self.agnesi),
            bool(self.apply_cutoff),
            tuple((len(h.weights), h.post) for h in self.heads),
        )

    def agnesi_r0(self, z_sender: torch.Tensor, z_receiver: torch.Tensor) -> torch.Tensor:
        """Per-edge ``r0 [E, 1] = (R[z_s] + R[z_r]) / 2`` (float64) from atomic numbers."""
        if self.covalent_radii is None:
            raise ValueError("EdgeChainSpec: no covalent_radii table (build from the MACE module)")
        cr = self.covalent_radii.to(torch.float64)
        zs, zr = (z.to(torch.int64).reshape(-1).cpu() for z in (z_sender, z_receiver))
        return (0.5 * (cr[zs] + cr[zr])).unsqueeze(-1)

    def constants(self, device, dtype) -> dict:
        """``w`` (Bessel weights) and per head the matrices and their transposes (cached)."""
        key = (torch.device(device), dtype)
        with self._lock:
            c = self._consts.get(key)
        if c is None:
            dev = torch.device(device)
            c = {
                "w": torch.tensor(self.bessel_weights, dtype=torch.float64).to(dev, dtype),
                "W": [[w.to(dev, dtype) for w in h.weights] for h in self.heads],
                "WT": [[w.T.contiguous().to(dev, dtype) for w in h.weights] for h in self.heads],
            }
            with self._lock:
                self._consts[key] = c
        return c


def _e3nn_fc_weights(net) -> tuple[torch.Tensor, ...]:
    """Effective matrices of an e3nn ``FullyConnectedNet`` (silu hidden layers)."""
    layers = [m for _n, m in sorted(net.named_children(), key=lambda kv: int(kv[0][5:]))]
    out, carry = [], 1.0
    for i, layer in enumerate(layers):
        W = layer.weight.detach().to(torch.float64).cpu()
        last = i == len(layers) - 1
        act = layer.act
        if act is None:
            if not last:
                raise NotImplementedError(
                    "edge_chain: a hidden FullyConnectedNet layer without act"
                )
            out.append(W * (carry / math.sqrt(layer.h_in * layer.var_in / layer.var_out)))
            break
        if last:
            raise NotImplementedError("edge_chain: FullyConnectedNet with out_act")
        f = getattr(act, "f", act)
        if f not in (torch.nn.functional.silu,) and not isinstance(f, torch.nn.SiLU):
            raise NotImplementedError(f"edge_chain: hidden activation must be silu, got {f!r}")
        cst = 1.0 if getattr(act, "_is_id", False) else float(getattr(act, "cst", 1.0))
        out.append(W * (carry / math.sqrt(layer.h_in * layer.var_in)))
        carry = cst * math.sqrt(layer.var_out)
    return tuple(out)


def _as_head(m, post, name) -> EdgeMLP:
    if isinstance(m, EdgeMLP):
        return m
    if isinstance(m, torch.nn.Module):
        if type(m).__name__ != "FullyConnectedNet":
            raise NotImplementedError(
                f"edge_chain: radial MLP {type(m).__name__} not covered (e3nn FullyConnectedNet only)"
            )
        return EdgeMLP(_e3nn_fc_weights(m), post, name)
    return EdgeMLP(tuple(m), post, name)


def edge_chain_spec(
    radial=None,
    mlps: Sequence | None = None,
    posts: Sequence | None = None,
    *,
    module=None,
    r_max: float | None = None,
    num_basis: int = 8,
    cutoff_p: float = 6,
    agnesi: tuple[float, float, float] | None = None,
    apply_cutoff: bool = True,
    covalent_radii=None,
) -> EdgeChainSpec:
    """Build an :class:`EdgeChainSpec`.

    - ``edge_chain_spec(module=mace_model)``: the model's ``radial_embedding``
      and, per interaction, ``conv_tp_weights`` (head ``tp_w{i}``) and, when
      present, ``density_fn`` with ``post="tanh_sq"`` (head ``density{i}``).
      Heads come in that order: ``tp_w0, density0, tp_w1, density1``.
    - ``edge_chain_spec(radial_embedding, [mlp, ...], posts)``: a MACE
      ``RadialEmbeddingBlock`` and e3nn ``FullyConnectedNet`` heads.
    - explicit: ``r_max``, ``num_basis``, ``cutoff_p``, ``agnesi=(a, q, p)``,
      ``apply_cutoff`` and ``mlps`` as :class:`EdgeMLP` (or lists of
      effective matrices).
    """
    if module is not None:
        heads, names, pp = [], [], []
        for i, it in enumerate(module.interactions):
            heads.append(it.conv_tp_weights)
            names.append(f"tp_w{i}")
            pp.append(None)
            if hasattr(it, "density_fn"):
                heads.append(it.density_fn)
                names.append(f"density{i}")
                pp.append("tanh_sq")
        spec = edge_chain_spec(module.radial_embedding, heads, pp)
        hs = tuple(EdgeMLP(h.weights, h.post, n) for h, n in zip(spec.heads, names, strict=True))
        return EdgeChainSpec(
            spec.r_max,
            spec.bessel_weights,
            spec.prefactor,
            spec.cutoff_p,
            hs,
            spec.agnesi,
            spec.apply_cutoff,
            spec.covalent_radii,
        )
    mlps = list(mlps or [])
    posts = list(posts) if posts is not None else [None] * len(mlps)
    if len(posts) != len(mlps):
        raise ValueError("edge_chain_spec: one post per MLP")
    heads = tuple(
        _as_head(m, p, f"head{i}") for i, (m, p) in enumerate(zip(mlps, posts, strict=True))
    )
    if radial is not None:
        bf, cf = radial.bessel_fn, radial.cutoff_fn
        if type(bf).__name__ != "BesselBasis":
            raise NotImplementedError(f"edge_chain: radial basis {type(bf).__name__} not covered")
        if type(cf).__name__ != "PolynomialCutoff":
            raise NotImplementedError(f"edge_chain: cutoff {type(cf).__name__} not covered")
        ag, cr = None, None
        dt = getattr(radial, "distance_transform", None)
        if dt is not None:
            if type(dt).__name__ != "AgnesiTransform":
                raise NotImplementedError(f"edge_chain: transform {type(dt).__name__} not covered")
            ag = (float(dt.a), float(dt.q), float(dt.p))
            cr = dt.covalent_radii.detach().to(torch.float64).cpu()
        if float(bf.r_max) != float(cf.r_max):
            raise NotImplementedError("edge_chain: Bessel and cutoff r_max differ")
        return EdgeChainSpec(
            float(cf.r_max),
            tuple(float(v) for v in bf.bessel_weights.detach().to(torch.float64).cpu()),
            float(bf.prefactor),
            float(cf.p),
            heads,
            ag,
            bool(getattr(radial, "apply_cutoff", True)),
            cr,
        )
    if r_max is None:
        raise ValueError("edge_chain_spec: pass radial=, module= or r_max=")
    w = tuple(math.pi / r_max * (n + 1) for n in range(int(num_basis)))
    return EdgeChainSpec(
        float(r_max),
        w,
        math.sqrt(2.0 / r_max),
        float(cutoff_p),
        heads,
        None if agnesi is None else tuple(float(v) for v in agnesi),
        bool(apply_cutoff),
        None if covalent_radii is None else torch.as_tensor(covalent_radii, dtype=torch.float64),
    )


# ── PyTorch path ───────────────────────────────────────────────────────────
#
# ``st`` is EdgeChainSpec.static: (r_max, prefactor, p, agnesi, apply_cutoff,
# heads) with heads ((n_layers, post), ...). ``Ws`` is the flat list of every
# head's matrices in head order.


def _cutoff_coeffs(p: float):
    return (p + 1.0) * (p + 2.0) / 2.0, p * (p + 2.0), p * (p + 1.0) / 2.0


def _radial(r, r0, w, st):
    """``x``, ``c`` and the basis ``B [E, nb]`` (MACE's formulas and op order)."""
    r_max, pf, p, ag, _apply, _heads = st
    A, Bc, Cc = _cutoff_coeffs(p)
    u = r / r_max
    c = (1.0 - A * torch.pow(u, p) + Bc * torch.pow(u, p + 1) - Cc * torch.pow(u, p + 2)) * (
        r < r_max
    )
    if ag is None:
        x = r
    else:
        a, q, pp = ag
        t = r / r0
        x = (1 + (a * torch.pow(t, q) / (1 + torch.pow(t, q - pp)))).reciprocal()
    B = pf * (torch.sin(w * x) / x)
    return x, c, B


def _split_heads(Ws, st):
    out, i = [], 0
    for n, _post in st[5]:
        out.append(Ws[i : i + n])
        i += n
    return out


def _edge_chain_forward_torch(r, r0, w, Ws, st):
    _x, c, B = _radial(r, r0, w, st)
    apply = st[4]
    f = B * c if apply else B
    outs = []
    for (_n, post), hw in zip(st[5], _split_heads(Ws, st), strict=True):
        h = f
        for W in hw[:-1]:
            h = torch.nn.functional.silu(h @ W)
        z = h @ hw[-1]
        y = torch.tanh(z * z) if post == "tanh_sq" else z
        outs.append(y if apply else y * c)
    return tuple(outs)


def _edge_chain_backward_torch(gs, r, r0, w, Ws, WTs, st):
    """``∂/∂r [E, 1]`` for the head cotangents ``gs`` (None: no gradient)."""
    r_max, pf, p, ag, apply, heads = st
    x, c, B = _radial(r, r0, w, st)
    f = B * c if apply else B
    df = None
    dc = None
    for (_n, post), hw, hwt, g in zip(
        heads, _split_heads(Ws, st), _split_heads(WTs, st), gs, strict=True
    ):
        if g is None:
            continue
        zs, h = [], f
        for W in hw[:-1]:
            z = h @ W
            zs.append(z)
            h = torch.nn.functional.silu(z)
        z = h @ hw[-1]
        if not apply:
            y = torch.tanh(z * z) if post == "tanh_sq" else z
            t = (g * y).sum(1, keepdim=True)
            dc = t if dc is None else dc + t
            g = g * c
        if post == "tanh_sq":
            y = torch.tanh(z * z)
            g = g * (1 - y * y) * (2 * z)
        dh = g @ hwt[-1]
        for zl, WT in zip(reversed(zs), reversed(hwt[:-1]), strict=True):
            s = torch.sigmoid(zl)
            dh = (dh * (s * (1 + zl * (1 - s)))) @ WT
        df = dh if df is None else df + dh
    if df is None:
        return torch.zeros_like(r)
    if apply:
        t = (df * B).sum(1, keepdim=True)
        dc = t if dc is None else dc + t
        dB = df * c
    else:
        dB = df
    wx = w * x
    dBdx = pf * (w * torch.cos(wx) * x - torch.sin(wx)) / (x * x)
    dx = (dB * dBdx).sum(1, keepdim=True)
    if ag is not None:
        a, q, pp = ag
        t = r / r0
        s1, s2 = torch.pow(t, q), torch.pow(t, q - pp)
        D = 1 + s2
        N = a * s1 / D
        dx = dx * (-(x * x) * (N / r) * (q - (q - pp) * s2 / D))
    A, Bc, Cc = _cutoff_coeffs(p)
    u = r / r_max
    dcdr = (
        (
            -A * p * torch.pow(u, p - 1)
            + Bc * (p + 1) * torch.pow(u, p)
            - Cc * (p + 2) * torch.pow(u, p + 1)
        )
        / r_max
        * (r < r_max)
    )
    return dx + dc * dcdr


class _EdgeChainTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, r0, spec):
        c = spec.constants(r.device, r.dtype)
        st = spec.static
        Ws = [w for hw in c["W"] for w in hw]
        ctx.spec = spec
        ctx.save_for_backward(r, r0)
        return region(_edge_chain_forward_torch, id(spec))(r, r0, c["w"], Ws, st)

    @staticmethod
    def backward(ctx, *gs):
        r, r0 = ctx.saved_tensors
        spec = ctx.spec
        c = spec.constants(r.device, r.dtype)
        Ws = [w for hw in c["W"] for w in hw]
        WTs = [w for hw in c["WT"] for w in hw]
        gs = [None if g is None else g.contiguous() for g in gs]
        dr = region(_edge_chain_backward_torch, id(spec))(gs, r, r0, c["w"], Ws, WTs, spec.static)
        return dr, None, None


def _check_edge_chain_args(r, r0, spec):
    if r.dim() != 2 or r.shape[1] != 1:
        raise ValueError(f"edge_chain: lengths must be [E, 1] or [E], got {tuple(r.shape)}")
    if not r.dtype.is_floating_point:
        raise TypeError(f"edge_chain: lengths must be floating, got {r.dtype}")
    if spec.agnesi is not None:
        if r0 is None:
            raise ValueError("edge_chain: the spec has an Agnesi transform; pass r0 [E, 1]")
        if r0.shape != r.shape:
            raise ValueError(
                f"edge_chain: r0 {tuple(r0.shape)} must match lengths {tuple(r.shape)}"
            )


@_fp64_host_guard("edge_chain")
def edge_chain(lengths: torch.Tensor, spec: EdgeChainSpec, r0: torch.Tensor | None = None):
    """The fused radial chain: one output ``[E, d_h]`` per head of ``spec``.

    ``lengths [E, 1]`` (or ``[E]``) edge lengths; ``r0 [E, 1]`` the Agnesi
    species constant (``spec.agnesi_r0``), required when the spec has the
    transform and ignored otherwise. Per head ``h``:
    ``out_h = post_h(MLP_h(B(x(r)) · c(r)))`` (``apply_cutoff``), or
    ``post_h(MLP_h(B)) · c`` otherwise. Differentiable in ``lengths``
    (``r0`` is a constant of the neighbour list).

    MACE-MP-0b2 medium (``edge_chain_spec(module=model)``): ``tp_w0 [E, 512]``,
    ``density0 [E, 1]``, ``tp_w1 [E, 1280]``, ``density1 [E, 1]``, which are
    ``conv_tp_weights(edge_feats)`` and ``tanh(density_fn(edge_feats)²)`` of
    both interactions.
    """
    if not isinstance(spec, EdgeChainSpec):
        raise TypeError(f"edge_chain: expected an EdgeChainSpec, got {type(spec).__name__}")
    r = lengths.unsqueeze(-1) if lengths.dim() == 1 else lengths
    if r0 is not None and r0.dim() == 1:
        r0 = r0.unsqueeze(-1)
    _check_edge_chain_args(r, r0, spec)
    r0 = r if spec.agnesi is None else r0.to(r.device, r.dtype)
    from .nki import _edge as _ek

    if _ek.edge_chain_kernel_route(r, r0, spec):
        return _ek.edge_chain_nki(r, r0, spec)
    return _EdgeChainTorch.apply(r, r0, spec)
