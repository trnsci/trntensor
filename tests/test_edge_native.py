"""Host plumbing of the per-edge NKI kernels, on CPU (trnsci/trnsci#178).

No NKI and no device. The four kernels are replaced by the differentiable
statements of their contracts in ``trntensor.nki._edge`` (partition-major
``[128, M·F]`` views, packed weights, static tables), so these tests check
everything around a launch, as ``test_equivariant_native.py`` does for the
equivariant kernels:

- the native TorchNeuron path through a stand-in ``torch_neuronx.nki_op``
  that builds real ``torch.library`` custom ops (``register_autograd``, fake
  functions, ``opcheck``);
- the XLA / simulator ``torch.autograd.Function`` path;
- routing (``"auto"`` is opt-in until device validation), padding and the
  weight packing.

The kernels themselves are checked against the same contracts in
``test_edge_nki.py`` (``nki.simulate`` and MLIR verification) and on device
in ``test_edge_neuron.py``.
"""

from __future__ import annotations

import itertools
import sys
import types

import pytest
import torch
from test_edge import FIXTURE, fixture_spec, random_spec

import trntensor
from trntensor.nki import _edge as eh
from trntensor.nki import _edge_kernels as xk
from trntensor.nki import dispatch


def emu_sh_fwd(v2, lmax, normalize, eps, table):
    return eh.sh_fwd_torch(v2, lmax, normalize, eps, table)


def emu_sh_bwd(g2, v2, lmax, normalize, eps, dtable):
    return eh.sh_bwd_torch(g2, v2, lmax, normalize, eps, dtable)


def emu_edge_fwd(r2, r02, wpack, dims, consts, bessel):
    return eh.edge_chain_fwd_torch(r2, r02, wpack, dims, consts, bessel)


def emu_edge_bwd(g2, r2, r02, wpack, wtpack, dims, consts, bessel):
    # The kernel reads the transposed weights from wtpack: check the packing
    # here, so the emulator fails if the host packed them wrong.
    _check_wtpack(wpack, wtpack, dims)
    return eh.edge_chain_bwd_torch(g2, r2, r02, wpack, dims, consts, bessel)


def _check_wtpack(wpack, wtpack, dims):
    _f, _wf, bwd, _wb = xk.edge_pack_layout(dims)
    Ws = eh._unpack(wpack, dims)
    i = 0
    for h, (_post, hs) in enumerate(xk.edge_heads(dims)):
        for ell in range(1, len(hs)):
            W = Ws[i]
            i += 1
            if len(hs) == 2 and hs[1] == 1:
                rep = wtpack[:, bwd[h][0] : bwd[h][0] + hs[0]]
                assert torch.equal(rep, W[:, 0][None, :].expand(128, -1))
                continue
            for c in range(-(-hs[ell] // 128)):
                rows = W.T[c * 128 : (c + 1) * 128]
                o = bwd[h][ell - 1] + c * hs[ell - 1]
                assert torch.equal(wtpack[: rows.shape[0], o : o + hs[ell - 1]], rows)


EMULATORS = {
    "sh_fwd_kernel": emu_sh_fwd,
    "sh_bwd_kernel": emu_sh_bwd,
    "edge_chain_fwd_kernel": emu_edge_fwd,
    "edge_chain_bwd_kernel": emu_edge_bwd,
}

_ns = itertools.count()


def _fake_torch_neuronx(registered):
    mod = types.ModuleType("torch_neuronx")
    tag = next(_ns)

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            qual = name.replace("trntensor::", f"trntensor_edgetest{tag}::")
            op = torch.library.custom_op(qual, mutates_args=())(f)
            if fake_fn is not None:
                op.register_fake(fake_fn)
            registered[name] = op
            return op

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    return mod


@pytest.fixture
def kernels(monkeypatch):
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setenv("TRNTENSOR_FORCE_BACKEND", "nki")
    monkeypatch.delenv("TRNTENSOR_USE_SIMULATOR", raising=False)
    for name, fn in EMULATORS.items():
        monkeypatch.setattr(xk, name, fn, raising=False)


@pytest.fixture
def native(kernels, monkeypatch):
    registered = {}
    monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_torch_neuronx(registered))
    monkeypatch.delitem(sys.modules, "trntensor.nki._native_edge", raising=False)
    monkeypatch.delattr(trntensor.nki, "_native_edge", raising=False)
    monkeypatch.setattr(eh, "_native_ops_table", None)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    yield registered


@pytest.fixture
def xla(kernels, monkeypatch):
    calls = []

    def run(kernel, op_name, tensors, consts):
        eh.check_operands(op_name, tensors)
        calls.append(op_name)
        return kernel(*tensors, **consts)

    monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
    monkeypatch.setattr(eh, "_run", run)
    yield calls


@pytest.fixture(scope="module")
def fx():
    return torch.load(FIXTURE, map_location="cpu", weights_only=True)


def _vjp(fn, ins, cot):
    ins = [t.clone().requires_grad_(True) for t in ins]
    out = fn(*ins)
    outs = out if isinstance(out, tuple) else (out,)
    gs = torch.autograd.grad(outs, ins, cot)
    return [o.detach() for o in outs], gs


def _reference(fn, ins, cot):
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TRNTENSOR_FORCE_BACKEND", "pytorch")
        return _vjp(fn, ins, cot)


def _check(fn, ins, cot):
    ref = _reference(fn, ins, cot)
    got = _vjp(fn, ins, cot)
    for a, b in zip(got[0] + list(got[1]), ref[0] + list(ref[1]), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def _sh_case(fx):
    s = fx["sh"]
    return trntensor.spherical_harmonics, [s["vectors"]], [s["grad_out"]]


def _edge_case(fx):
    d = fx["edge_chain"]
    spec = fixture_spec(fx)
    return (lambda r: trntensor.edge_chain(r, spec, d["r0"])), [d["lengths"]], list(d["grad_out"])


class TestNativePath:
    @pytest.mark.parametrize("case", [_sh_case, _edge_case])
    def test_autograd_through_registered_ops(self, native, fx, case):
        _check(*case(fx))
        names = {"sh", "sh_bwd", "edge_chain", "edge_chain_bwd"}
        assert {f"trntensor::{n}" for n in names} <= set(native)

    def test_ops_pass_opcheck(self, native, fx):
        _check(*_sh_case(fx))  # registers the ops
        v2 = eh._pad(fx["sh"]["vectors"], 256).contiguous().view(128, -1)
        table, dtable = eh.sh_tables(3, "component")
        torch.library.opcheck(native["trntensor::sh"], (v2, 3, 1, 1e-12, list(table), list(dtable)))
        spec = fixture_spec(fx)
        k = eh._edge_kernel_constants(spec, "cpu")
        r2 = eh._pad(fx["edge_chain"]["lengths"], 256, 15.0).contiguous().view(128, -1)
        r02 = eh._pad(fx["edge_chain"]["r0"], 256, 1.0).contiguous().view(128, -1)
        args = (
            r2,
            r02,
            k["wpack"],
            k["wtpack"],
            list(k["dims"]),
            list(k["consts"]),
            list(k["bessel"]),
        )
        torch.library.opcheck(
            native["trntensor::edge_chain"], args, test_utils=("test_schema", "test_faketensor")
        )

    def test_double_backward_falls_back_to_aten(self, native, fx):
        v = fx["sh"]["vectors"][:20].double()

        def second(backend):
            with pytest.MonkeyPatch.context() as m:
                m.setenv("TRNTENSOR_FORCE_BACKEND", backend)
                x = v.clone().requires_grad_(True)
                y = trntensor.spherical_harmonics(x)
                (g,) = torch.autograd.grad(y.square().sum(), x, create_graph=True)
                return torch.autograd.grad(g.square().sum(), x)[0]

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # fp64 under explicit nki: fp32 kernel, warns once
            got = second("nki")
        torch.testing.assert_close(got, second("pytorch"), rtol=1e-4, atol=1e-5)


class TestAutogradFunctionPath:
    @pytest.mark.parametrize("case", [_sh_case, _edge_case])
    def test_kernels_forward_and_backward(self, xla, fx, case):
        _check(*case(fx))
        assert xla[0] in ("sh", "edge_chain") and xla[1].endswith("_bwd")

    @pytest.mark.parametrize("apply_cutoff", [True, False])
    def test_padding_and_partial_heads(self, xla, apply_cutoff):
        spec = random_spec(apply_cutoff=apply_cutoff, p=6, widths=(8, 16, 16, 140))
        assert eh.edge_chain_fits(spec)
        g = torch.Generator().manual_seed(4)
        r = (0.7 + 5.0 * torch.rand(300, 1, generator=g)).float()
        r0 = (0.6 + 0.8 * torch.rand(300, 1, generator=g)).float()
        cot = [torch.randn(300, d, generator=g) for d in spec.dims_out]
        cot[1] = None  # an unused head
        fn = lambda t: trntensor.edge_chain(t, spec, r0)  # noqa: E731

        def vjp(f):
            x = r.clone().requires_grad_(True)
            outs = f(x)
            pairs = [(o, c) for o, c in zip(outs, cot, strict=True) if c is not None]
            return outs, torch.autograd.grad([o for o, _ in pairs], x, [c for _, c in pairs])[0]

        outs, gr = vjp(fn)
        with pytest.MonkeyPatch.context() as m:
            m.setenv("TRNTENSOR_FORCE_BACKEND", "pytorch")
            routs, rgr = vjp(fn)
        # fp32 on both sides, different GEMM blockings (padded rows): norm-relative
        for a, b in zip(outs, routs, strict=True):
            assert float((a - b).norm() / b.norm()) < 1e-6
        assert float((gr - rgr).norm() / rgr.norm()) < 1e-6

    def test_operand_dtype_guard(self):
        with pytest.raises(TypeError):
            eh.check_operands("sh", (torch.zeros(2, 3, dtype=torch.float64),))
        with pytest.raises(TypeError):
            eh.check_operands("nope", ())


class TestRouting:
    def test_auto_is_opt_in(self, kernels, monkeypatch):
        monkeypatch.delenv("TRNTENSOR_FORCE_BACKEND")
        monkeypatch.setattr(dispatch, "_backend", "auto")
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        v = torch.randn(4, 3)
        assert not eh.sh_kernel_route(v, 3, True, "component")  # cpu tensor, no opt-in
        monkeypatch.setenv(eh.EDGE_NKI_ENV, "1")
        assert not eh.sh_kernel_route(v, 3, True, "component")  # opted in, but not on neuron
        monkeypatch.setattr(dispatch, "_on_neuron_device", lambda t: True)
        assert eh.sh_kernel_route(v, 3, True, "component")
        assert not eh.sh_kernel_route(v.double(), 3, True, "component")

    def test_forced_nki_routes(self, kernels):
        assert eh.sh_kernel_route(torch.randn(4, 3), 3, True, "component")
        assert not eh.sh_kernel_route(torch.randn(0, 3), 3, True, "component")
        spec = random_spec()
        r = torch.rand(5, 1)
        assert eh.edge_chain_kernel_route(r, r, spec)

    def test_specs_outside_the_kernel(self, kernels, monkeypatch):
        r = torch.rand(5, 1)
        for spec in (
            random_spec(agnesi=False),  # sine argument up to num_basis·π
            random_spec(p=5.5),  # non-integer cutoff order
            random_spec(widths=(8, 160, 4)),  # hidden width > 128
        ):
            assert not eh.edge_chain_fits(spec)
            assert not eh.edge_chain_kernel_route(r, r, spec)
        monkeypatch.setenv("TRNTENSOR_REQUIRE_NKI", "1")
        with pytest.raises(NotImplementedError):
            eh.edge_chain_kernel_route(r, r, random_spec(agnesi=False))


class TestPacking:
    def test_pack_layout_roundtrip(self):
        spec = random_spec(widths=(8, 64, 64, 300))
        wp, wt = eh.pack_weights(spec)
        dims = eh.kernel_dims(spec)
        Ws = eh._unpack(wp, dims)
        flat = [w for h in spec.heads for w in h.weights]
        for a, b in zip(Ws, flat, strict=True):
            assert torch.equal(a, b)
        _check_wtpack(wp, wt, dims)

    def test_heads_and_offsets(self):
        spec = random_spec()
        dims = eh.kernel_dims(spec)
        heads = xk.edge_heads(dims)
        assert [hs for _p, hs in heads] == [(8, 16, 16, 24), (8, 1), (8, 12, 7)]
        assert [p for p, _hs in heads] == [0, 1, 0]
        assert xk.edge_out_offsets(dims) == ([0, 24, 25], 32)
