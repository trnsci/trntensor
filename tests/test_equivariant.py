"""CPU tests for the equivariant contractions (trnsci/trntensor#138).

Oracles, in order of authority:

1. the MACE-MP-0b2 medium capture slice in ``fixtures/mace_mp0b2_tiny.pt``
   (MACE's own outputs and input gradients; bar 2e-5 relative, fp32);
2. the MLIP exemplar's per-path reference blocks (``mlip_reference.py``);
3. fp64 ``torch.autograd.gradcheck`` of every required gradient;
4. equivariance: rotating every input irrep by its Wigner D matrix rotates
   the outputs the same way.

e3nn / mace-torch checks run only where those packages are installed (they
are optional, test-only dependencies).
"""

from __future__ import annotations

import math

import pytest
import torch
from mlip_reference import (
    conv_tp_reference,
    load_fixture,
    mace_conv_tp_irreps,
    symmetric_contraction_reference,
)

import trntensor
from trntensor import equivariant as eq

RTOL_MACE = 2e-5  # the contract's bar against MACE captures (fp32)


def rel(got, ref) -> float:
    got, ref = got.detach().double(), ref.detach().double()
    assert got.shape == ref.shape, (tuple(got.shape), tuple(ref.shape))
    den = ref.norm()
    return float((got - ref).norm() / den) if den > 0 else float((got - ref).norm())


@pytest.fixture(scope="module")
def cap():
    return load_fixture()


@pytest.fixture(autouse=True)
def _pytorch_backend():
    prev = trntensor.get_backend()
    trntensor.set_backend("pytorch")
    yield
    trntensor.set_backend(prev)


def _e3nn():
    """Import e3nn under torch >= 2.6 (its constants file pickles ``slice``)."""
    pytest.importorskip("e3nn")
    torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    return o3


class _f64_default:
    """Build e3nn / MACE modules with float64 constants (not fp32 upcast)."""

    def __enter__(self):
        self.prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *exc):
        torch.set_default_dtype(self.prev)


def _rotation(seed=0):
    g = torch.Generator().manual_seed(seed)
    a, b, c = (torch.rand(3, generator=g, dtype=torch.float64) * 2 * math.pi).tolist()
    return a, b, c


def _block_rotate(t, irreps, Ds):
    """Rotate a mul-major feature vector [E, Σ mul (2l+1)] block by block."""
    parts, o = [], 0
    for mul, ell in irreps:
        d = 2 * ell + 1
        blk = t[:, o : o + mul * d].reshape(-1, mul, d)
        parts.append(torch.einsum("ij,euj->eui", Ds[ell], blk).reshape(-1, mul * d))
        o += mul * d
    return torch.cat(parts, 1)


# ─────────────────────────────────────────────────────────────── constants


class TestConstants:
    def test_wigner_3j_matches_capture_constants(self, cap):
        # The capture stored e3nn's constants at fp32 precision (then cast to fp64).
        for a in cap["A1"]:
            for key, C in a["spec"]["w3j"].items():
                l1, l2, l3 = (int(v) for v in key.split(","))
                assert (trntensor.wigner_3j(l1, l2, l3) - C).abs().max() < 5e-8, key

    def test_wigner_3j_matches_e3nn_all_triples(self):
        o3 = _e3nn()
        n = 0
        for l1 in range(4):
            for l2 in range(4):
                for l3 in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
                    ours = trntensor.wigner_3j(l1, l2, l3)
                    ref = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)
                    assert (ours - ref).abs().max() < 1e-14, (l1, l2, l3)
                    n += 1
        assert n == 34

    def test_wigner_D_matches_e3nn(self):
        o3 = _e3nn()
        a, b, c = _rotation(3)
        for ell in range(4):
            ref = o3.wigner_D(ell, *(torch.tensor(v, dtype=torch.float64) for v in (a, b, c)))
            assert (trntensor.wigner_D(ell, a, b, c) - ref).abs().max() < 1e-12, ell

    def test_wigner_3j_is_rotation_invariant(self):
        """C is an intertwiner of the D matrices (no e3nn needed)."""
        R = _rotation(1)
        D = [trntensor.wigner_D(ell, *R) for ell in range(4)]
        for ell in range(4):
            assert torch.allclose(D[ell] @ D[ell].T, torch.eye(2 * ell + 1, dtype=torch.float64))
        for l1 in range(4):
            for l2 in range(4):
                for l3 in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
                    C = trntensor.wigner_3j(l1, l2, l3)
                    rot = torch.einsum("ijk,ai,bj,ck->abc", C, D[l1], D[l2], D[l3])
                    assert torch.allclose(rot, C, atol=1e-12), (l1, l2, l3)
                    assert math.isclose(C.norm().item(), 1.0)

    def test_wigner_3j_limits(self):
        with pytest.raises(ValueError, match="triangle"):
            trntensor.wigner_3j(0, 1, 2)
        with pytest.raises(ValueError, match="l <= 3"):
            trntensor.wigner_3j(4, 4, 0)


# ─────────────────────────────────────────────────────────────── A1 spec


class TestConvTPSpec:
    @pytest.mark.parametrize("layer", [0, 1])
    def test_explicit_irreps_reproduce_the_captured_spec(self, cap, layer):
        """MACE's irreps and instruction order, built without e3nn, give the
        captured α (e3nn's 'uvu' normalisation) and w3j."""
        a = next(a for a in cap["A1"] if a["layer"] == layer)
        in1, in2, out, ins = mace_conv_tp_irreps(layer, channels=8)
        assert [tuple(t) for t in a["spec"]["irreps_out"]] == out
        spec = trntensor.conv_tp_spec(in1, in2, out, ins)
        got = [(i1, i2, io) for i1, i2, io, _ in spec.instructions]
        assert got == [tuple(i[:3]) for i in a["spec"]["instructions"]]
        for (_, _, _, al), ref in zip(spec.instructions, a["spec"]["instructions"], strict=True):
            assert math.isclose(al, ref[3], rel_tol=1e-12)
        assert spec.weight_numel == a["spec"]["weight_numel"]
        torch.testing.assert_close(
            trntensor.conv_tp(a["x"], a["y"], a["w"], spec), a["out"], rtol=0, atol=1e-6
        )

    def test_string_irreps(self):
        s1 = trntensor.conv_tp_spec(
            "4x0e+4x1o", "1x0e+1x1o+1x2e+1x3o", "4x0e+4x1o", [(0, 0, 0), (1, 0, 1)]
        )
        s2 = trntensor.conv_tp_spec(
            [(4, 0), (4, 1)],
            [(1, ell) for ell in range(4)],
            [(4, 0), (4, 1)],
            [(0, 0, 0), (1, 0, 1)],
        )
        assert s1.irreps_in1 == s2.irreps_in1 and s1.instructions == s2.instructions

    def test_dict_roundtrip_and_cache(self, cap):
        d = cap["A1"][1]["spec"]
        s = eq._as_spec(d, eq.ConvTPSpec)
        assert eq._as_spec(d, eq.ConvTPSpec) is s
        s2 = eq.ConvTPSpec.from_dict(s.as_dict())
        assert s2.instructions == s.instructions and s2.irreps_out == s.irreps_out

    def test_from_e3nn_module(self):
        o3 = _e3nn()
        in1, in2, out, ins = mace_conv_tp_irreps(1, channels=3)

        def irr(lst):
            return o3.Irreps([(m, (ell, (-1) ** ell)) for m, ell in lst])

        with _f64_default():
            tp = o3.TensorProduct(
                irr(in1),
                irr(in2),
                irr(out),
                [(i1, i2, io, "uvu", True) for i1, i2, io in ins],
                shared_weights=False,
                internal_weights=False,
            )
        spec = trntensor.conv_tp_spec(module=tp)
        explicit = trntensor.conv_tp_spec(in1, in2, out, ins)
        for a, b in zip(spec.instructions, explicit.instructions, strict=True):
            assert a[:3] == b[:3] and math.isclose(a[3], b[3], rel_tol=1e-12)
        g = torch.Generator().manual_seed(0)
        E = 7
        x = torch.randn(E, tp.irreps_in1.dim, generator=g, dtype=torch.float64)
        y = torch.randn(E, tp.irreps_in2.dim, generator=g, dtype=torch.float64)
        w = torch.randn(E, tp.weight_numel, generator=g, dtype=torch.float64)
        torch.testing.assert_close(
            trntensor.conv_tp(x, y, w, spec), tp(x, y, w), rtol=1e-12, atol=1e-12
        )

    def test_from_mace_interaction_block(self):
        pytest.importorskip("mace")
        o3 = _e3nn()
        from mace.modules.irreps_tools import tp_out_irreps_with_instructions

        irreps1 = o3.Irreps("4x0e+4x1o")
        irreps2 = o3.Irreps("1x0e+1x1o+1x2e+1x3o")
        target = o3.Irreps("4x0e+4x1o+4x2e+4x3o")
        irreps_mid, instr = tp_out_irreps_with_instructions(irreps1, irreps2, target)
        with _f64_default():
            tp = o3.TensorProduct(
                irreps1,
                irreps2,
                irreps_mid,
                instructions=instr,
                shared_weights=False,
                internal_weights=False,
            )
        spec = trntensor.conv_tp_spec(module=tp)
        in1, in2, out, ins = mace_conv_tp_irreps(1, 4)
        assert list(spec.irreps_out) == out
        assert [i[:3] for i in spec.instructions] == [tuple(i) for i in ins]
        g = torch.Generator().manual_seed(1)
        x = torch.randn(5, irreps1.dim, generator=g, dtype=torch.float64)
        y = torch.randn(5, irreps2.dim, generator=g, dtype=torch.float64)
        w = torch.randn(5, tp.weight_numel, generator=g, dtype=torch.float64)
        torch.testing.assert_close(
            trntensor.conv_tp(x, y, w, spec), tp(x, y, w), rtol=1e-12, atol=1e-12
        )

    def test_argument_checks(self):
        spec = trntensor.conv_tp_spec(*mace_conv_tp_irreps(0, 2))
        x, y, w = torch.randn(3, 2), torch.randn(3, 16), torch.randn(3, 8)
        with pytest.raises(ValueError, match="features"):
            trntensor.conv_tp(x, y[:, :9], w, spec)
        with pytest.raises(ValueError, match="edge counts"):
            trntensor.conv_tp(x, y[:2], w, spec)
        with pytest.raises(TypeError, match="dtype"):
            trntensor.conv_tp(x.double(), y, w, spec)
        with pytest.raises(NotImplementedError, match="multiplicity"):
            trntensor.conv_tp_spec([(2, 0), (3, 1)], [(1, 0)], [(2, 0)], [(0, 0, 0)])


# ─────────────────────────────────────────────────────────────── A1 numerics


class TestConvTP:
    @pytest.mark.parametrize("layer", [0, 1])
    def test_matches_mace_capture(self, cap, layer):
        a = next(a for a in cap["A1"] if a["layer"] == layer)
        want = [k for k in ("x", "y", "w") if a.get("grad_" + k) is not None]
        ins = {k: a[k].clone().requires_grad_(k in want) for k in ("x", "y", "w")}
        out = trntensor.conv_tp(ins["x"], ins["y"], ins["w"], a["spec"])
        assert rel(out, a["out"]) < RTOL_MACE
        gs = torch.autograd.grad(out, [ins[k] for k in want], a["grad_out"])
        for k, gk in zip(want, gs, strict=True):
            assert rel(gk, a["grad_" + k]) < RTOL_MACE, k
        assert ("x" in want) == (layer == 1)  # layer-0 x depends on species only

    @pytest.mark.parametrize("layer", [0, 1])
    def test_matches_reference_blocks(self, cap, layer):
        a = next(a for a in cap["A1"] if a["layer"] == layer)
        x, y, w = (a[k].double().requires_grad_(True) for k in ("x", "y", "w"))
        spec = a["spec"]
        ours = trntensor.conv_tp(x, y, w, spec)
        ref = conv_tp_reference(x, y, w, spec)
        torch.testing.assert_close(ours, ref, rtol=1e-12, atol=1e-12)
        g = torch.randn_like(ours)
        for a1, a2 in zip(
            torch.autograd.grad(ours, (x, y, w), g),
            torch.autograd.grad(ref, (x, y, w), g),
            strict=True,
        ):
            torch.testing.assert_close(a1, a2, rtol=1e-11, atol=1e-11)

    @pytest.mark.parametrize("layer", [0, 1])
    def test_gradcheck_fp64(self, layer):
        spec = trntensor.conv_tp_spec(*mace_conv_tp_irreps(layer, channels=2))
        g = torch.Generator().manual_seed(layer)
        E = 3
        x, y, w = (
            torch.randn(E, n, generator=g, dtype=torch.float64, requires_grad=True)
            for n in (spec.dim_in1, spec.dim_in2, spec.weight_numel)
        )
        fn = lambda x, y, w: trntensor.conv_tp(x, y, w, spec)  # noqa: E731
        assert torch.autograd.gradcheck(fn, (x, y, w))
        # Each gradient on its own (the backward skips the others).
        for k in range(3):
            args = [t.detach().requires_grad_(i == k) for i, t in enumerate((x, y, w))]
            assert torch.autograd.gradcheck(fn, tuple(args))

    def test_gradgradcheck_fp64(self):
        """The PyTorch backward is built from differentiable ops (training
        through forces works on this path)."""
        spec = trntensor.conv_tp_spec(*mace_conv_tp_irreps(1, channels=1))
        g = torch.Generator().manual_seed(5)
        x, y, w = (
            torch.randn(2, n, generator=g, dtype=torch.float64, requires_grad=True)
            for n in (spec.dim_in1, spec.dim_in2, spec.weight_numel)
        )
        assert torch.autograd.gradgradcheck(
            lambda x, y, w: trntensor.conv_tp(x, y, w, spec), (x, y, w)
        )

    @pytest.mark.parametrize("layer", [0, 1])
    def test_equivariance(self, layer):
        in1, in2, out_irreps, ins = mace_conv_tp_irreps(layer, channels=4)
        spec = trntensor.conv_tp_spec(in1, in2, out_irreps, ins)
        g = torch.Generator().manual_seed(2)
        E = 6
        x = torch.randn(E, spec.dim_in1, generator=g, dtype=torch.float64)
        y = torch.randn(E, spec.dim_in2, generator=g, dtype=torch.float64)
        w = torch.randn(E, spec.weight_numel, generator=g, dtype=torch.float64)
        R = _rotation(7)
        D = [trntensor.wigner_D(ell, *R) for ell in range(4)]
        out = trntensor.conv_tp(x, y, w, spec)
        out_rot = trntensor.conv_tp(_block_rotate(x, in1, D), _block_rotate(y, in2, D), w, spec)
        torch.testing.assert_close(
            out_rot, _block_rotate(out, out_irreps, D), rtol=1e-10, atol=1e-10
        )
        assert rel(out_rot, out) > 1e-3  # the rotation is not trivial

    def test_shared_output_block_and_uncovered_output(self):
        """Two paths into one output block sum (index_add path), and an output
        block no path writes is zero, as in e3nn."""
        in1, in2 = [(3, 0), (3, 1)], [(1, 0), (1, 1)]
        out = [(3, 1), (3, 0)]
        spec = trntensor.conv_tp_spec(in1, in2, out, [(0, 1, 0), (1, 0, 0)])
        assert not spec.layout.out_bijective and spec.layout.zero_fill == 1
        g = torch.Generator().manual_seed(4)
        x = torch.randn(5, spec.dim_in1, generator=g, dtype=torch.float64, requires_grad=True)
        y = torch.randn(5, spec.dim_in2, generator=g, dtype=torch.float64, requires_grad=True)
        w = torch.randn(5, spec.weight_numel, generator=g, dtype=torch.float64, requires_grad=True)
        ours = trntensor.conv_tp(x, y, w, spec)
        ref = conv_tp_reference(x, y, w, spec.as_dict())
        torch.testing.assert_close(ours, ref, rtol=1e-12, atol=1e-12)
        assert (ours[:, 9:] == 0).all()
        assert torch.autograd.gradcheck(lambda x, y, w: trntensor.conv_tp(x, y, w, spec), (x, y, w))

    def test_zero_edges(self):
        spec = trntensor.conv_tp_spec(*mace_conv_tp_irreps(1, 2))
        x, y, w = (
            torch.zeros(0, spec.dim_in1),
            torch.zeros(0, 16),
            torch.zeros(0, spec.weight_numel),
        )
        assert trntensor.conv_tp(x, y, w, spec).shape == (0, spec.dim_out)

    @pytest.mark.parametrize("layer", [0, 1])
    def test_fewer_launches_than_the_per_path_reference(self, layer):
        """The fused path is a fixed handful of kernel-launching aten ops per
        call, independent of the path count; e3nn's formulation is per path."""
        from torch.utils._python_dispatch import TorchDispatchMode

        class Count(TorchDispatchMode):
            n = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if not getattr(func, "is_view", False):
                    Count.n += 1
                return func(*args, **(kwargs or {}))

        def count(fn):
            Count.n = 0
            with Count():
                fn()
            return Count.n

        spec = trntensor.conv_tp_spec(*mace_conv_tp_irreps(layer, 8))
        g = torch.Generator().manual_seed(0)
        x, y, w, go = (
            torch.randn(16, n, generator=g)
            for n in (spec.dim_in1, spec.dim_in2, spec.weight_numel, spec.dim_out)
        )
        spec.constants("cpu", torch.float32)  # constants are built once, not per call
        d = spec.as_dict()
        ours = count(lambda: eq._conv_tp_forward_torch(x, y, w, spec))
        ours_bwd = count(lambda: eq._conv_tp_backward_torch(go, x, y, w, spec))
        ref = count(lambda: conv_tp_reference(x, y, w, d))
        assert ours <= 4 + len(spec.layout.runs) < ref, (ours, ref)
        assert ours_bwd <= 32, ours_bwd  # 29 at layer 1; e3nn forward + backward: 144


# ─────────────────────────────────────────────────────────────── A2


def _symcon_inputs(cap, layer):
    a = next(a for a in cap["A2"] if a["layer"] == layer)
    return a, a["spec"]


class TestSymmetricContraction:
    @pytest.mark.parametrize("layer", [0, 1])
    def test_matches_mace_capture(self, cap, layer):
        a, spec = _symcon_inputs(cap, layer)
        x = a["x"].clone().requires_grad_(True)
        out = trntensor.symmetric_contraction(x, a["species"], spec)
        assert rel(out, a["out"]) < RTOL_MACE
        (gx,) = torch.autograd.grad(out, x, a["grad_out"])
        assert rel(gx, a["grad_x"]) < RTOL_MACE

    @pytest.mark.parametrize("layer", [0, 1])
    def test_matches_reference_blocks(self, cap, layer):
        a, spec = _symcon_inputs(cap, layer)
        x = a["x"].double().requires_grad_(True)
        ours = trntensor.symmetric_contraction(x, a["species"], spec)
        ref = symmetric_contraction_reference(x, a["species"], spec)
        torch.testing.assert_close(ours, ref, rtol=1e-11, atol=1e-11)
        g = torch.randn_like(ours)
        (g1,) = torch.autograd.grad(ours, x, g)
        (g2,) = torch.autograd.grad(ref, x, g)
        torch.testing.assert_close(g1, g2, rtol=1e-10, atol=1e-10)

    def test_monomial_basis_is_pruned(self, cap):
        sizes = {}
        for a in cap["A2"]:
            lay = eq._as_spec(a["spec"], eq.SymmetricContractionSpec).layout
            sizes[a["layer"]] = (lay.A, lay.MK, lay.Dsum)
        # 968 sorted monomials of degree <= 3 in 16 features; only these couple.
        assert sizes == {0: (410, 202, 4), 1: (94, 28, 1)}

    @pytest.mark.parametrize("layer", [0, 1])
    def test_gradcheck_fp64(self, cap, layer):
        a, spec = _symcon_inputs(cap, layer)
        x = a["x"][:3].double().requires_grad_(True)
        z = a["species"][:3]
        assert torch.autograd.gradcheck(lambda x: trntensor.symmetric_contraction(x, z, spec), (x,))

    def test_gradgradcheck_fp64(self, cap):
        a, spec = _symcon_inputs(cap, 1)
        x = a["x"][:1, :2].double()
        sp = eq.symmetric_contraction_spec(
            contractions=[
                {**c, "W": {k: v[:, :, :2] for k, v in c["W"].items()}}
                for c in spec["contractions"]
            ]
        )
        x.requires_grad_(True)
        z = a["species"][:1]
        assert torch.autograd.gradgradcheck(
            lambda x: trntensor.symmetric_contraction(x, z, sp), (x,)
        )

    @pytest.mark.parametrize("layer", [0, 1])
    def test_equivariance(self, cap, layer):
        a, spec = _symcon_inputs(cap, layer)
        x = a["x"].double()
        R = _rotation(11)
        D = [trntensor.wigner_D(ell, *R) for ell in range(4)]
        Din = torch.block_diag(*D)  # 0e+1o+2e+3o on the last axis
        out = trntensor.symmetric_contraction(x, a["species"], spec)
        out_rot = trntensor.symmetric_contraction(
            torch.einsum("ab,ncb->nca", Din, x), a["species"], spec
        )
        C = x.shape[1]
        ls = [0] if layer == 1 else [0, 1]
        expect = _block_rotate(out, [(C, ell) for ell in ls], D)
        # The captured U are fp32 values: equivariance holds to their precision.
        assert rel(out_rot, expect) < 1e-6
        if layer == 0:
            assert rel(out_rot, out) > 1e-3  # the 1o block really rotates

    def test_chunked_rows_match(self, cap, monkeypatch):
        a, spec = _symcon_inputs(cap, 0)
        x = a["x"].double().requires_grad_(True)
        full = trntensor.symmetric_contraction(x, a["species"], spec)
        (gf,) = torch.autograd.grad(full.sum(), x)
        monkeypatch.setattr(eq, "_SYMCON_CHUNK_ELEMS", 3 * 8 * 410)  # 3 nodes per chunk
        part = trntensor.symmetric_contraction(x, a["species"], spec)
        (gp,) = torch.autograd.grad(part.sum(), x)
        torch.testing.assert_close(part, full, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(gp, gf, rtol=1e-12, atol=1e-12)

    def test_from_mace_module(self):
        pytest.importorskip("mace")
        o3 = _e3nn()
        from mace.modules.symmetric_contraction import SymmetricContraction

        torch.manual_seed(0)
        with _f64_default():
            sc = SymmetricContraction(
                o3.Irreps("4x0e+4x1o+4x2e+4x3o"),
                o3.Irreps("4x0e+4x1o"),
                correlation=3,
                num_elements=3,
            )
        spec = trntensor.symmetric_contraction_spec(module=sc)
        assert [c["irrep_out"] for c in spec.contractions] == ["0e", "1o"]
        N = 5
        x = torch.randn(N, 4, 16, dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0, 2, 1, 1, 0])
        onehot = torch.nn.functional.one_hot(z, 3).double()
        ref = sc(x, onehot)
        ours = trntensor.symmetric_contraction(x, z, spec)
        torch.testing.assert_close(ours, ref, rtol=1e-11, atol=1e-11)
        g = torch.randn_like(ref)
        torch.testing.assert_close(
            torch.autograd.grad(ours, x, g)[0],
            torch.autograd.grad(ref, x, g)[0],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_argument_checks(self, cap):
        a, spec = _symcon_inputs(cap, 1)
        with pytest.raises(ValueError, match="spec wants"):
            trntensor.symmetric_contraction(a["x"][:, :4], a["species"], spec)
        with pytest.raises(ValueError, match="species"):
            trntensor.symmetric_contraction(a["x"], a["species"][:3], spec)
        with pytest.raises(TypeError, match="int64"):
            trntensor.symmetric_contraction(a["x"], a["species"].float(), spec)


# ─────────────────────────────────────────────────────────────── dispatch


class TestDispatch:
    def test_auto_equals_pytorch_on_cpu(self, cap):
        a = cap["A1"][1]
        ref = trntensor.conv_tp(a["x"], a["y"], a["w"], a["spec"])
        trntensor.set_backend("auto")
        torch.testing.assert_close(
            trntensor.conv_tp(a["x"], a["y"], a["w"], a["spec"]), ref, rtol=0, atol=0
        )
        b = cap["A2"][0]
        ref = trntensor.symmetric_contraction(b["x"], b["species"], b["spec"])
        trntensor.set_backend("pytorch")
        torch.testing.assert_close(
            trntensor.symmetric_contraction(b["x"], b["species"], b["spec"]), ref
        )

    def test_auto_keeps_host_tensors_and_opt_out_on_pytorch(self, monkeypatch):
        from trntensor.nki import _equivariant as eqk
        from trntensor.nki import dispatch

        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.delenv(eq.EQUIVARIANT_NKI_ENV, raising=False)
        assert eq._auto_uses_kernels()
        for off in ("0", "false", "no"):
            monkeypatch.setenv(eq.EQUIVARIANT_NKI_ENV, off)
            assert not eq._auto_uses_kernels()
        monkeypatch.delenv(eq.EQUIVARIANT_NKI_ENV, raising=False)
        trntensor.set_backend("auto")
        spec = trntensor.conv_tp_spec(*mace_conv_tp_irreps(1, 2))
        x, y, w = (
            torch.randn(4, spec.dim_in1),
            torch.randn(4, 16),
            torch.randn(4, spec.weight_numel),
        )
        assert not eqk.conv_tp_kernel_route(x, y, w, spec)

    def test_public_entry_points_have_the_fp64_host_guard(self):
        assert hasattr(trntensor.conv_tp, "__wrapped__")
        assert hasattr(trntensor.symmetric_contraction, "__wrapped__")

    @pytest.mark.parametrize(
        "E,bucket",
        [(1, 256), (150, 256), (2156, 2304), (10392, 11264), (10378, 11264), (96278, 98304)],
    )
    def test_edge_bucket(self, E, bucket):
        from trntensor.nki._equivariant import edge_bucket

        assert edge_bucket(E) == bucket
        assert bucket % 256 == 0 and bucket >= E and (bucket - E) <= max(255, E // 8)

    def test_node_pad(self):
        from trntensor.nki._equivariant import node_pad

        assert node_pad(16, 128) == 16 and node_pad(17, 128) == 24 and node_pad(1728, 128) == 1728
        assert node_pad(3, 8) == 128
