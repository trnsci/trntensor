"""CPU tests of the per-edge and node MLIP ops (trnsci/trnsci#178).

``spherical_harmonics``, ``edge_chain`` and ``equivariant_linear`` on the
PyTorch path:

- against MACE-MP-0b2 medium through the committed fixture
  ``fixtures/mace_mp0b2_edge.pt`` (written by the exemplar's
  ``examples/mlip/check_fused.py --trntensor-fixture``: the 16-atom box, all
  150 edges, the real force-step cotangents; the two ``conv_tp_weights``
  heads sliced to 16 outputs, which is exact because output columns are
  independent) and two e3nn-built ``o3.Linear`` maps;
- against e3nn / MACE modules directly when they are installed;
- fp64 ``gradcheck`` / ``gradgradcheck`` of every analytic backward;
- equivariance (spherical harmonics under ``wigner_D``), the padding contract
  (edges at or past ``r_max`` contribute exactly zero), and the compiled
  region path (``set_region_backend``) against eager.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import trntensor
from trntensor import edge, linear

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mace_mp0b2_edge.pt"


@pytest.fixture(scope="module")
def fx():
    return torch.load(FIXTURE, map_location="cpu", weights_only=True)


@pytest.fixture(autouse=True)
def _pytorch_backend():
    prev = trntensor.get_backend()
    trntensor.set_backend("pytorch")
    yield
    trntensor.set_backend(prev)


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm())


def _e3nn():
    try:
        torch.serialization.add_safe_globals([slice])
        from e3nn import o3

        return o3
    except Exception:
        return None


def fixture_spec(fx) -> edge.EdgeChainSpec:
    d = fx["edge_chain"]
    heads = [edge.EdgeMLP(tuple(h["weights"]), h["post"], h["name"]) for h in d["heads"]]
    return edge.EdgeChainSpec(
        d["r_max"],
        tuple(d["bessel_weights"]),
        d["prefactor"],
        d["cutoff_p"],
        tuple(heads),
        tuple(d["agnesi"]),
        d["apply_cutoff"],
    )


def random_spec(agnesi=True, apply_cutoff=True, p=5, seed=0, widths=(8, 16, 16, 24)):
    g = torch.Generator().manual_seed(seed)

    def mlp(hs):
        return [
            torch.randn(a, b, generator=g, dtype=torch.float64) / math.sqrt(a)
            for a, b in zip(hs, hs[1:], strict=False)
        ]

    heads = [
        edge.EdgeMLP(tuple(mlp(list(widths))), None, "tp_w0"),
        edge.EdgeMLP(tuple(mlp([widths[0], 1])), "tanh_sq", "density0"),
        edge.EdgeMLP(tuple(mlp([widths[0], 12, 7])), None, "tp_w1"),
    ]
    return edge.edge_chain_spec(
        r_max=5.0,
        num_basis=widths[0],
        cutoff_p=p,
        agnesi=(1.0805, 0.9183, 4.5791) if agnesi else None,
        mlps=heads,
        apply_cutoff=apply_cutoff,
    )


def _lengths(E, seed=0, dtype=torch.float64, lo=0.7, hi=5.6):
    g = torch.Generator().manual_seed(seed)
    r = lo + (hi - lo) * torch.rand(E, 1, generator=g, dtype=torch.float64)
    r0 = 0.6 + 0.8 * torch.rand(E, 1, generator=g, dtype=torch.float64)
    return r.to(dtype), r0.to(dtype)


# ─────────────────────────────────────────────────────── spherical harmonics


class TestSphericalHarmonics:
    def test_matches_mace_fixture(self, fx):
        s = fx["sh"]
        v = s["vectors"].clone().requires_grad_(True)
        y = trntensor.spherical_harmonics(v)
        assert y.shape == (150, 16)
        assert _rel(y, s["out"]) < 1e-6
        (dv,) = torch.autograd.grad(y, v, s["grad_out"])
        assert _rel(dv, s["grad_vectors"]) < 1e-6

    @pytest.mark.parametrize("lmax", [0, 1, 2, 3])
    @pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_matches_e3nn(self, lmax, normalization, normalize):
        o3 = _e3nn()
        if o3 is None:
            pytest.skip("e3nn not importable")
        v = torch.randn(64, 3, dtype=torch.float64)
        sh = o3.SphericalHarmonics(list(range(lmax + 1)), normalize, normalization)
        got = trntensor.spherical_harmonics(v, lmax, normalize, normalization)
        torch.testing.assert_close(got, sh(v), rtol=1e-12, atol=1e-12)
        if lmax:
            g = torch.randn(64, (lmax + 1) ** 2, dtype=torch.float64)
            v1, v2 = v.clone().requires_grad_(), v.clone().requires_grad_()
            (d1,) = torch.autograd.grad(sh(v1), v1, g)
            (d2,) = torch.autograd.grad(
                trntensor.spherical_harmonics(v2, lmax, normalize, normalization), v2, g
            )
            torch.testing.assert_close(d2, d1, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("lmax", [1, 2, 3])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_gradcheck_fp64(self, lmax, normalize):
        v = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)

        def f(t):
            return trntensor.spherical_harmonics(t, lmax, normalize)

        assert torch.autograd.gradcheck(f, (v,))
        assert torch.autograd.gradgradcheck(f, (v,))

    @pytest.mark.parametrize("lmax", [1, 2, 3])
    def test_equivariant_under_wigner_D(self, lmax):
        v = torch.randn(20, 3, dtype=torch.float64)
        a, b, c = 0.3, -1.1, 2.0
        R = trntensor.wigner_D(1, a, b, c)
        y, yr = trntensor.spherical_harmonics(v, lmax), trntensor.spherical_harmonics(v @ R.T, lmax)
        for ell in range(lmax + 1):
            s = slice(ell * ell, (ell + 1) * (ell + 1))
            D = trntensor.wigner_D(ell, a, b, c)
            torch.testing.assert_close(yr[:, s], y[:, s] @ D.T, rtol=1e-10, atol=1e-10)

    def test_zero_vector_is_finite(self):
        v = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        y = trntensor.spherical_harmonics(v)
        (g,) = torch.autograd.grad(y.sum(), v)
        assert torch.isfinite(y).all() and torch.isfinite(g).all()

    def test_leading_dims_and_validation(self):
        v = torch.randn(2, 5, 3)
        assert trntensor.spherical_harmonics(v).shape == (2, 5, 16)
        with pytest.raises(ValueError):
            trntensor.spherical_harmonics(torch.randn(4, 2))
        with pytest.raises(ValueError):
            trntensor.spherical_harmonics(torch.randn(4, 3), lmax=4)

    def test_coefficients_reproduce_polynomials(self):
        t = trntensor.sh_coefficients(3)
        assert t["C"].shape == (19, 16) and t["dC"].shape == (16, 3, 10)
        n = torch.nn.functional.normalize(torch.randn(10, 3, dtype=torch.float64), dim=1)
        y = t["c0"] + edge._monomials(n, 3) @ t["C"]
        torch.testing.assert_close(y, trntensor.spherical_harmonics(n), rtol=1e-12, atol=1e-12)


# ─────────────────────────────────────────────────────────────── edge chain


class TestEdgeChain:
    def test_matches_mace_fixture(self, fx):
        d = fx["edge_chain"]
        spec = fixture_spec(fx)
        L = d["lengths"].clone().requires_grad_(True)
        outs = trntensor.edge_chain(L, spec, d["r0"])
        assert [o.shape[1] for o in outs] == [16, 1, 16, 1]
        for o, ref in zip(outs, d["out"], strict=True):
            assert _rel(o, ref) < 1e-6
        (dl,) = torch.autograd.grad(outs, L, d["grad_out"])
        # The fp32 VJP sits at the chain's fp32 conditioning floor (MACE's own
        # fp32 VJP is ~1e-6 from fp64 here, ~1e-5 at 216 atoms): bound it
        # against fp64, at the scale of MACE's own fp32 error.
        err_ours = _rel(dl, d["grad_lengths_fp64"])
        err_mace = _rel(d["grad_lengths"], d["grad_lengths_fp64"])
        assert err_ours < max(3 * err_mace, 2e-6), (err_ours, err_mace)
        assert _rel(dl, d["grad_lengths"]) < 2e-5

    def test_fixture_spec_matches_mace_spec_fields(self, fx):
        spec = fixture_spec(fx)
        assert spec.num_basis == 8 and spec.r_max == 5.0 and spec.cutoff_p == 5.0
        assert spec.agnesi is not None and spec.apply_cutoff
        assert [h.post for h in spec.heads] == [None, "tanh_sq", None, "tanh_sq"]

    @pytest.mark.parametrize("agnesi", [True, False])
    @pytest.mark.parametrize("apply_cutoff", [True, False])
    @pytest.mark.parametrize("p", [5, 6])
    def test_gradcheck_fp64(self, agnesi, apply_cutoff, p):
        spec = random_spec(agnesi, apply_cutoff, p, widths=(4, 6, 5))
        r, r0 = _lengths(9, seed=p)
        r.requires_grad_(True)

        def f(t):
            return trntensor.edge_chain(t, spec, r0 if agnesi else None)

        assert torch.autograd.gradcheck(f, (r,))
        assert torch.autograd.gradgradcheck(f, (r,))

    def test_partial_cotangents(self):
        spec = random_spec()
        r, r0 = _lengths(11)
        r.requires_grad_(True)
        outs = trntensor.edge_chain(r, spec, r0)
        (g1,) = torch.autograd.grad(outs[1].sum(), r)
        r2 = r.detach().clone().requires_grad_(True)
        ref = edge._edge_chain_forward_torch(
            r2,
            r0,
            spec.constants("cpu", torch.float64)["w"],
            [w for hw in spec.constants("cpu", torch.float64)["W"] for w in hw],
            spec.static,
        )
        (g2,) = torch.autograd.grad(ref[1].sum(), r2)
        torch.testing.assert_close(g1, g2, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("apply_cutoff", [True, False])
    def test_edges_past_r_max_contribute_exactly_zero(self, apply_cutoff):
        spec = random_spec(apply_cutoff=apply_cutoff)
        r = torch.tensor([[5.0], [7.3], [15.0]], dtype=torch.float32, requires_grad=True)
        r0 = torch.ones_like(r)
        outs = trntensor.edge_chain(r, spec, r0)
        for o, h in zip(outs, spec.heads, strict=True):
            if apply_cutoff and h.post is None:
                # B·c = 0 feeds the MLP; with no bias its output is exactly 0
                assert bool((o == 0).all())
            if not apply_cutoff:
                assert bool((o == 0).all())
        (g,) = torch.autograd.grad(outs, r, [torch.ones_like(o) for o in outs])
        assert bool((g == 0).all())

    def test_r0_required_with_agnesi(self):
        spec = random_spec()
        with pytest.raises(ValueError):
            trntensor.edge_chain(torch.rand(4, 1), spec)

    def test_flat_lengths(self):
        spec = random_spec(agnesi=False)
        r, _ = _lengths(6, dtype=torch.float32)
        a = trntensor.edge_chain(r, spec)
        b = trntensor.edge_chain(r.reshape(-1), spec)
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y)

    def test_spec_from_mace_modules(self):
        o3 = _e3nn()
        if o3 is None:
            pytest.skip("e3nn not importable")
        try:
            from mace.modules.blocks import RadialEmbeddingBlock
        except Exception:
            pytest.skip("mace-torch not importable")
        from e3nn import nn

        torch.manual_seed(0)
        re = RadialEmbeddingBlock(5.0, 8, 5, "bessel", "Agnesi")
        mlp = nn.FullyConnectedNet([8, 16, 16, 24], torch.nn.functional.silu)
        dens = nn.FullyConnectedNet([8, 1])
        spec = trntensor.edge_chain_spec(re, [mlp, dens], [None, "tanh_sq"])
        r, r0 = _lengths(40, dtype=torch.float64)
        re, mlp, dens = re.double(), mlp.double(), dens.double()
        # MACE's Agnesi transform reads r0 from the species; pass it directly
        ag = re.distance_transform
        cutoff = re.cutoff_fn(r)
        t = r / r0
        x = 1 / (1 + ag.a * t**ag.q / (1 + t ** (ag.q - ag.p)))
        ef = re.bessel_fn(x) * cutoff
        got = trntensor.edge_chain(r, spec, r0)
        torch.testing.assert_close(got[0], mlp(ef), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(got[1], torch.tanh(dens(ef) ** 2), rtol=1e-10, atol=1e-12)


# ─────────────────────────────────────────────────────── equivariant linear


class TestEquivariantLinear:
    @pytest.mark.parametrize("i", [0, 1])
    def test_matches_e3nn_fixture(self, fx, i):
        d = fx["linear"][i]
        spec = trntensor.equivariant_linear_spec(
            irreps_in=d["irreps_in"], irreps_out=d["irreps_out"], matrix=d["matrix"]
        )
        assert spec.irrep_mode
        x = d["x"].clone().requires_grad_(True)
        y = trntensor.equivariant_linear(x, spec)
        assert _rel(y, d["out"]) < 1e-6
        (dx,) = torch.autograd.grad(y, x, d["grad_out"])
        assert _rel(dx, d["grad_x"]) < 1e-6

    def test_one_matmul_per_irrep_type(self, fx):
        d = fx["linear"][0]
        spec = trntensor.equivariant_linear_spec(
            irreps_in=d["irreps_in"], irreps_out=d["irreps_out"], matrix=d["matrix"]
        )
        assert [g[0] for g in spec.groups] == [0, 1, 2, 3]
        assert [tuple(g[5].shape) for g in spec.groups] == [(1, 8, 8)] * 4

    def test_dense_mode_equals_irrep_mode(self, fx):
        d = fx["linear"][1]
        kw = {"irreps_in": d["irreps_in"], "irreps_out": d["irreps_out"], "matrix": d["matrix"]}
        a = trntensor.equivariant_linear(d["x"], trntensor.equivariant_linear_spec(**kw))
        b = trntensor.equivariant_linear(
            d["x"], trntensor.equivariant_linear_spec(**kw, mode="dense")
        )
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    def _species_spec(self, S=3):
        g = torch.Generator().manual_seed(0)
        irr_in, irr_out = "4x0e+3x1o+2x2e", "5x0e+2x1o"
        Ms = []
        for _ in range(S):
            M = torch.zeros(4 + 9 + 10, 5 + 6, dtype=torch.float64)
            M[0:4, 0:5] = torch.randn(4, 5, generator=g, dtype=torch.float64)
            M[4:13, 5:11] = torch.kron(
                torch.randn(3, 2, generator=g, dtype=torch.float64), torch.eye(3)
            )
            Ms.append(M)
        return trntensor.equivariant_linear_spec(
            irreps_in=irr_in, irreps_out=irr_out, matrix=torch.stack(Ms), elements=[2, 5, 7]
        ), Ms

    def test_per_species(self):
        spec, Ms = self._species_spec()
        x = torch.randn(9, spec.dim_in, dtype=torch.float64)
        z = torch.tensor([2, 5, 7, 7, 2, 5, 2, 7, 5])
        idx = {2: 0, 5: 1, 7: 2}
        ref = torch.stack([x[n] @ Ms[idx[int(z[n])]] for n in range(9)])
        torch.testing.assert_close(trntensor.equivariant_linear(x, spec, z), ref)
        with pytest.raises(ValueError):
            trntensor.equivariant_linear(x, spec)

    def test_gradcheck_fp64(self, fx):
        spec, _ = self._species_spec()
        x = torch.randn(5, spec.dim_in, dtype=torch.float64, requires_grad=True)
        z = torch.tensor([7, 2, 5, 5, 2])
        assert torch.autograd.gradcheck(lambda t: trntensor.equivariant_linear(t, spec, z), (x,))
        assert torch.autograd.gradgradcheck(
            lambda t: trntensor.equivariant_linear(t, spec, z), (x,)
        )
        d = fx["linear"][1]
        spec1 = trntensor.equivariant_linear_spec(
            irreps_in=d["irreps_in"], irreps_out=d["irreps_out"], matrix=d["matrix"]
        )
        x1 = torch.randn(4, spec1.dim_in, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: trntensor.equivariant_linear(t, spec1), (x1,))

    def test_rejects_non_equivariant_matrix(self):
        M = torch.randn(4 + 3, 2 + 3, dtype=torch.float64)
        with pytest.raises(ValueError):
            trntensor.equivariant_linear_spec(
                irreps_in="4x0e+1x1o", irreps_out="2x0e+1x1o", matrix=M
            )

    def test_unused_inputs_and_uncovered_outputs(self):
        M = torch.zeros(2 + 3, 3 + 5, dtype=torch.float64)
        M[0:2, 0:3] = torch.randn(2, 3, dtype=torch.float64)  # 1o in, 2e out: no path
        spec = trntensor.equivariant_linear_spec(
            irreps_in="2x0e+1x1o", irreps_out="3x0e+1x2e", matrix=M
        )
        x = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
        y = trntensor.equivariant_linear(x, spec)
        torch.testing.assert_close(y, x.detach() @ M)
        (dx,) = torch.autograd.grad(y.sum(), x)
        torch.testing.assert_close(dx, torch.ones(4, 8, dtype=torch.float64) @ M.T)

    def test_spec_from_e3nn_modules(self):
        o3 = _e3nn()
        if o3 is None:
            pytest.skip("e3nn not importable")
        torch.manual_seed(0)
        lin = o3.Linear("8x0e+4x1o+4x1o+2x2e", "6x0e+3x1o+2x2e")
        spec = trntensor.equivariant_linear_spec(lin)
        assert spec.irrep_mode and len(spec.groups) == 3
        x = torch.randn(10, lin.irreps_in.dim)
        torch.testing.assert_close(
            trntensor.equivariant_linear(x, spec), lin(x), rtol=1e-5, atol=1e-6
        )
        tp = o3.FullyConnectedTensorProduct("4x0e+2x1o", "5x0e", "3x0e+2x1o")
        spec = trntensor.equivariant_linear_spec(tp, elements=[1, 3])
        z = torch.tensor([1, 3, 3, 1])
        attrs = torch.nn.functional.one_hot(z, 5).float()
        x = torch.randn(4, tp.irreps_in1.dim)
        torch.testing.assert_close(
            trntensor.equivariant_linear(x, spec, z), tp(x, attrs), rtol=1e-5, atol=1e-6
        )


# ─────────────────────────────────────────────────────────── compiled regions


@pytest.fixture
def region():
    prev = trntensor.get_region_backend()
    trntensor.set_region_backend("aot_eager")
    yield
    trntensor.set_region_backend(prev)


class TestCompiledRegion:
    def _fb(self, fn, *ts):
        ins = [t.clone().requires_grad_(True) for t in ts]
        out = fn(*ins)
        outs = out if isinstance(out, tuple) else (out,)
        gs = torch.autograd.grad(outs, ins, [torch.ones_like(o) for o in outs])
        return [o.detach() for o in outs], gs

    def test_ops_match_eager(self, region, fx):
        spec = fixture_spec(fx)
        d = fx["edge_chain"]
        cases = [
            (trntensor.spherical_harmonics, (fx["sh"]["vectors"],)),
            (lambda r: trntensor.edge_chain(r, spec, d["r0"]), (d["lengths"],)),
        ]
        lspec = trntensor.equivariant_linear_spec(
            irreps_in=fx["linear"][0]["irreps_in"],
            irreps_out=fx["linear"][0]["irreps_out"],
            matrix=fx["linear"][0]["matrix"],
        )
        cases.append((lambda x: trntensor.equivariant_linear(x, lspec), (fx["linear"][0]["x"],)))
        for fn, ts in cases:
            got = self._fb(fn, *ts)
            trntensor.set_region_backend(None)
            ref = self._fb(fn, *ts)
            trntensor.set_region_backend("aot_eager")
            for a, b in zip(got[0] + list(got[1]), ref[0] + list(ref[1]), strict=True):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_double_backward_stays_eager(self, region):
        v = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        y = trntensor.spherical_harmonics(v)
        (g,) = torch.autograd.grad(y.square().sum(), v, create_graph=True)
        (gg,) = torch.autograd.grad(g.sum(), v)
        assert torch.isfinite(gg).all()


def test_linear_module_exports():
    assert trntensor.EquivariantLinearSpec is linear.EquivariantLinearSpec
    assert trntensor.EdgeChainSpec is edge.EdgeChainSpec
