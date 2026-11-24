"""Host plumbing of the equivariant NKI kernels, on CPU (trnsci/trntensor#138).

No NKI and no device here. The four kernels are replaced by torch
*emulators* of their exact I/O contracts (padded shapes, the compact
block-CG and ``plan`` tuple, the m-major symmetric-contraction layout), so
these tests check everything around a kernel launch:

- the native TorchNeuron path: a stand-in ``torch_neuronx.nki_op`` that
  builds real ``torch.library`` custom ops, so ``register_autograd``, the
  fake (meta) functions and the op schemas are exercised (``opcheck``);
- the XLA / simulator path: the ``torch.autograd.Function`` wrappers;
- routing, edge / node padding and the fp64 policy.

The emulators are also the readable statement of the kernel contracts; the
kernels themselves are checked against the PyTorch path in
``test_equivariant_nki.py`` (``nki.simulate``, CPU) and on device in
``test_equivariant_neuron.py``.
"""

from __future__ import annotations

import itertools
import sys
import types
import warnings

import pytest
import torch
from mlip_reference import load_fixture

import trntensor
from trntensor import equivariant as eq
from trntensor.nki import _equivariant as eqk
from trntensor.nki import _equivariant_kernels as ek
from trntensor.nki import dispatch
from trntensor.nki._equivariant_kernels import conv_tp_plan_paths

# ─────────────────────────────────────────────────────────────── emulators


def emu_conv_tp_fwd(x, y, w, cg, plan, zero_fill=0):
    plan = tuple(plan)
    E, Dout = x.shape[0], plan[0]
    paths = conv_tp_plan_paths(plan)
    U = w.shape[1] // len(paths)
    G = y @ cg
    # Unwritten columns are NaN unless zero_fill: a plan that misses a column fails.
    out = x.new_zeros(E, Dout) if zero_fill else x.new_full((E, Dout), float("nan"))
    for p, (x_off, d1, q_off, d3, o_off, acc) in enumerate(paths):
        xw = x[:, x_off : x_off + U * d1].reshape(E, U, d1) * w[:, p * U : (p + 1) * U, None]
        r = torch.einsum("eui,eik->euk", xw, G[:, q_off : q_off + d1 * d3].reshape(E, d1, d3))
        r = r.reshape(E, U * d3)
        if acc:
            out[:, o_off : o_off + U * d3] += r
        else:
            out[:, o_off : o_off + U * d3] = r
    return out


def emu_conv_tp_bwd(g, x, y, w, cg, cgT, plan, need_dx=1, need_dy=1, need_dw=1):
    plan = tuple(plan)
    E, D1 = x.shape
    D2, PW, Q = y.shape[1], w.shape[1], cg.shape[1]
    paths = conv_tp_plan_paths(plan)
    U = PW // len(paths)
    G = y @ cg
    dx, dG, dw = x.new_zeros(E, D1), x.new_zeros(E, Q), x.new_zeros(E, PW)
    for p, (x_off, d1, q_off, d3, o_off, _acc) in enumerate(paths):
        xv = x[:, x_off : x_off + U * d1].reshape(E, U, d1)
        gv = g[:, o_off : o_off + U * d3].reshape(E, U, d3)
        wp = w[:, p * U : (p + 1) * U, None]
        Gp = G[:, q_off : q_off + d1 * d3].reshape(E, d1, d3)
        h = torch.einsum("euk,eik->eui", gv, Gp)
        dw[:, p * U : (p + 1) * U] = (xv * h).sum(-1)
        dx[:, x_off : x_off + U * d1] += (wp * h).reshape(E, U * d1)
        dG[:, q_off : q_off + d1 * d3] = torch.einsum("eui,euk->eik", xv * wp, gv).reshape(E, -1)
    out = x.new_full((E, D1 + D2 + PW), float("nan"))
    if need_dx:
        out[:, :D1] = dx
    if need_dy:
        out[:, D1 : D1 + D2] = dG @ cgT
    if need_dw:
        out[:, D1 + D2 :] = dw
    return out


def _factors(x, sel):
    N, C, L = x.shape
    xa = torch.cat([x.reshape(N * C, L), x.new_ones(N * C, 1)], 1)
    return [xa @ sel[:, j, :] for j in range(3)]


def emu_symcon_fwd(x, wr, sel, usym, s):
    N, C, _ = x.shape
    X0, X1, X2 = _factors(x, sel)
    wrows = wr.permute(0, 2, 1).reshape(N * C, -1)
    return (((X0 * X1 * X2) @ usym * wrows) @ s).T.contiguous()


def emu_symcon_bwd(g, x, wr, sel, selT, usymT, sT):
    N, C, L = x.shape
    X0, X1, X2 = _factors(x, sel)
    wrows = wr.permute(0, 2, 1).reshape(N * C, -1)
    R = ((g.T @ sT) * wrows) @ usymT
    dxa = sum(
        (R * a * b) @ selT[:, j, :] for j, (a, b) in enumerate(((X1, X2), (X0, X2), (X0, X1)))
    )
    return dxa[:, :L].reshape(N, C, L).contiguous()


EMULATORS = {
    "conv_tp_fwd_kernel": emu_conv_tp_fwd,
    "conv_tp_bwd_kernel": emu_conv_tp_bwd,
    "symcon_fwd_kernel": emu_symcon_fwd,
    "symcon_bwd_kernel": emu_symcon_bwd,
}

# ─────────────────────────────────────────────────────────────── fixtures

_ns = itertools.count()


def _fake_torch_neuronx(registered):
    """``nki_op`` that registers a real custom op (unique namespace per import)."""
    mod = types.ModuleType("torch_neuronx")
    tag = next(_ns)

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            qual = name.replace("trntensor::", f"trntensor_eqtest{tag}::")
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
    """NKI 'available', kernels emulated, explicit NKI backend, host stays host."""
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setenv("TRNTENSOR_FORCE_BACKEND", "nki")
    monkeypatch.delenv("TRNTENSOR_USE_SIMULATOR", raising=False)
    for name, fn in EMULATORS.items():
        monkeypatch.setattr(ek, name, fn, raising=False)


@pytest.fixture
def native(kernels, monkeypatch):
    registered = {}
    monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_torch_neuronx(registered))
    monkeypatch.delitem(sys.modules, "trntensor.nki._native_equivariant", raising=False)
    monkeypatch.delattr(trntensor.nki, "_native_equivariant", raising=False)
    monkeypatch.setattr(eqk, "_native_ops_table", None)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    yield registered


@pytest.fixture
def xla(kernels, monkeypatch):
    """The autograd.Function path, with launches running the emulators."""
    calls = []

    def run(kernel, op_name, tensors, consts):
        eqk.check_operands(op_name, tensors)
        calls.append(op_name)
        return (kernel(*tensors, **consts),)

    monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
    monkeypatch.setattr(eqk, "_run", run)
    yield calls


@pytest.fixture(scope="module")
def cap():
    return load_fixture()


def _reference(fn, inputs, want, grad_out):
    prev = trntensor.get_backend()
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TRNTENSOR_FORCE_BACKEND", "pytorch")
        ins = {
            k: v.clone().requires_grad_(k in want)
            if torch.is_tensor(v) and v.is_floating_point()
            else v
            for k, v in inputs.items()
        }
        out = fn(**ins)
        gs = torch.autograd.grad(out, [ins[k] for k in want], grad_out)
    assert trntensor.get_backend() == prev
    return out.detach(), gs


def _through(fn, inputs, want, grad_out):
    ins = {
        k: v.clone().requires_grad_(k in want)
        if torch.is_tensor(v) and v.is_floating_point()
        else v
        for k, v in inputs.items()
    }
    out = fn(**ins)
    gs = torch.autograd.grad(out, [ins[k] for k in want], grad_out)
    return out.detach(), gs


def _check_conv_tp(a):
    want = [k for k in ("x", "y", "w") if a.get("grad_" + k) is not None]
    inputs = {"x": a["x"], "y": a["y"], "w": a["w"], "spec": a["spec"]}
    ref, rg = _reference(trntensor.conv_tp, inputs, want, a["grad_out"])
    out, gs = _through(trntensor.conv_tp, inputs, want, a["grad_out"])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    for g1, g2 in zip(gs, rg, strict=True):
        torch.testing.assert_close(g1, g2, rtol=1e-5, atol=1e-6)


def _check_symcon(a):
    inputs = {"x": a["x"], "species": a["species"], "spec": a["spec"]}
    ref, rg = _reference(trntensor.symmetric_contraction, inputs, ["x"], a["grad_out"])
    out, gs = _through(trntensor.symmetric_contraction, inputs, ["x"], a["grad_out"])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gs[0], rg[0], rtol=1e-5, atol=1e-6)


def _second_order(fn, inputs, want, grad_out, backend):
    """∂/∂(inputs) of |∂out/∂x · grad_out|²: double backward (training on forces)."""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TRNTENSOR_FORCE_BACKEND", backend)
        ins = {
            k: v.clone().requires_grad_(k in want)
            if torch.is_tensor(v) and v.is_floating_point()
            else v
            for k, v in inputs.items()
        }
        out = fn(**ins)
        (gx,) = torch.autograd.grad(out, ins["x"], grad_out, create_graph=True)
        # conv_tp is linear in x, so ∂x only reaches y and w there.
        second = [k for k in want if not (k == "x" and "y" in want)]
        return torch.autograd.grad(gx.square().sum(), [ins[k] for k in second])


def _check_double_backward(a, block):
    if block == "A1":
        fn, want = trntensor.conv_tp, ["x", "y", "w"]
        inputs = {"x": a["x"], "y": a["y"], "w": a["w"], "spec": a["spec"]}
    else:
        fn, want = trntensor.symmetric_contraction, ["x"]
        inputs = {"x": a["x"], "species": a["species"], "spec": a["spec"]}
    ref = _second_order(fn, inputs, want, a["grad_out"], "pytorch")
    got = _second_order(fn, inputs, want, a["grad_out"], "nki")
    for g1, g2 in zip(got, ref, strict=True):
        torch.testing.assert_close(g1, g2, rtol=1e-4, atol=1e-5)


# ─────────────────────────────────────────────────────────────── native path


class TestNativePath:
    @pytest.mark.parametrize("layer", [0, 1])
    def test_conv_tp_autograd_through_registered_ops(self, native, cap, layer):
        _check_conv_tp(next(a for a in cap["A1"] if a["layer"] == layer))
        assert set(native) == {
            f"trntensor::{n}" for n in ("conv_tp", "conv_tp_bwd", "symcon", "symcon_bwd")
        }

    @pytest.mark.parametrize("layer", [0, 1])
    def test_symcon_autograd_through_registered_ops(self, native, cap, layer):
        _check_symcon(next(a for a in cap["A2"] if a["layer"] == layer))

    @pytest.mark.parametrize("block", ["A1", "A2"])
    def test_double_backward_falls_back_to_aten(self, native, cap, block):
        """create_graph=True: the backward is built from differentiable aten ops."""
        _check_double_backward(next(a for a in cap[block] if a["layer"] == 1), block)

    def test_ops_pass_opcheck(self, native, cap):
        """Schema, fake functions and autograd registration are consistent."""
        a = cap["A1"][1]
        spec = eq._as_spec(a["spec"], eq.ConvTPSpec)
        ops = eqk._native_ops()
        c = spec.constants("cpu", torch.float32)
        E = 256
        x = torch.randn(E, spec.dim_in1, requires_grad=True)
        y = torch.randn(E, spec.dim_in2, requires_grad=True)
        w = torch.randn(E, spec.weight_numel, requires_grad=True)
        args = (x, y, w, c["cg_kernel"], c["cg_kernel_T"], list(spec.layout.plan), 0)
        tests = ("test_schema", "test_autograd_registration", "test_faketensor")
        torch.library.opcheck(ops["conv_tp"], args, test_utils=tests)
        b = cap["A2"][0]
        sspec = eq._as_spec(b["spec"], eq.SymmetricContractionSpec)
        sc = sspec.constants("cpu", torch.float32)
        xs = torch.randn(128, 8, 16, requires_grad=True)
        wr = sc["W_zmc"].index_select(0, torch.zeros(128, dtype=torch.long))
        sargs = (xs, wr, sc["sel"], sc["selT"], sc["usym"], sc["usymT"], sc["S"], sc["ST"])
        torch.library.opcheck(ops["symcon"], sargs, test_utils=tests)

    def test_only_needed_gradients_are_computed(self, native, cap, monkeypatch):
        seen = []

        def spy(*args, **kw):
            seen.append((kw["need_dx"], kw["need_dy"], kw["need_dw"]))
            return emu_conv_tp_bwd(*args, **kw)

        monkeypatch.setattr(ek, "conv_tp_bwd_kernel", spy)
        a = cap["A1"][0]
        y = a["y"].clone().requires_grad_(True)  # x and w are constants here
        out = trntensor.conv_tp(a["x"], y, a["w"], a["spec"])
        torch.autograd.grad(out, y, a["grad_out"])
        assert seen == [(0, 1, 0)]


# ─────────────────────────────────────────────────────────────── XLA / simulator path


class TestAutogradFunctionPath:
    @pytest.mark.parametrize("layer", [0, 1])
    def test_conv_tp(self, xla, cap, layer):
        _check_conv_tp(next(a for a in cap["A1"] if a["layer"] == layer))
        assert xla == ["conv_tp", "conv_tp_bwd"]

    @pytest.mark.parametrize("layer", [0, 1])
    def test_symcon(self, xla, cap, layer):
        _check_symcon(next(a for a in cap["A2"] if a["layer"] == layer))
        assert xla == ["symcon", "symcon_bwd"]

    @pytest.mark.parametrize("block", ["A1", "A2"])
    def test_double_backward_falls_back_to_aten(self, xla, cap, block):
        _check_double_backward(next(a for a in cap[block] if a["layer"] == 1), block)
        assert not any(c.endswith("_bwd") for c in xla)

    def test_fp64_under_explicit_nki_warns_and_keeps_dtype(self, xla, cap, monkeypatch):
        monkeypatch.setattr(dispatch, "_fp64_warned", set())
        a = cap["A1"][1]
        x, y, w = (a[k].double() for k in ("x", "y", "w"))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = trntensor.conv_tp(x, y, w, a["spec"])
        assert out.dtype == torch.float64
        assert any("fp32" in str(r.message) for r in rec)
        assert xla == ["conv_tp"]

    def test_fp64_under_require_nki_raises(self, xla, cap, monkeypatch):
        monkeypatch.setenv("TRNTENSOR_REQUIRE_NKI", "1")
        a = cap["A2"][1]
        with pytest.raises(NotImplementedError, match="float64"):
            trntensor.symmetric_contraction(a["x"].double(), a["species"], a["spec"])

    def test_operand_dtype_guard(self):
        with pytest.raises(TypeError, match="float32"):
            eqk.check_operands("conv_tp", [torch.zeros(1, dtype=torch.float64)] * 4)
        with pytest.raises(TypeError, match="no operand dtype entry"):
            eqk.check_operands("nope", [])


class TestRouting:
    def test_auto_needs_native_device_fp32_and_no_opt_out(self, kernels, monkeypatch):
        monkeypatch.delenv("TRNTENSOR_FORCE_BACKEND")
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        spec = trntensor.conv_tp_spec(
            [(2, 0)], [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(0, 0, 0), (0, 1, 1)]
        )
        x, y, w = torch.randn(3, 2), torch.randn(3, 4), torch.randn(3, 4)
        prev = trntensor.get_backend()
        try:
            trntensor.set_backend("auto")
            monkeypatch.delenv(eq.EQUIVARIANT_NKI_ENV, raising=False)
            assert not eqk.conv_tp_kernel_route(x, y, w, spec)  # host tensors
            monkeypatch.setattr(dispatch, "_on_neuron_device", lambda t: True)
            assert eqk.conv_tp_kernel_route(x, y, w, spec)  # default on
            monkeypatch.setenv(eq.EQUIVARIANT_NKI_ENV, "0")
            assert not eqk.conv_tp_kernel_route(x, y, w, spec)  # opt-out
            monkeypatch.setenv(eq.EQUIVARIANT_NKI_ENV, "1")
            assert eqk.conv_tp_kernel_route(x, y, w, spec)
            assert not eqk.conv_tp_kernel_route(x.double(), y.double(), w.double(), spec)
            trntensor.set_backend("pytorch")
            assert not eqk.conv_tp_kernel_route(x, y, w, spec)
        finally:
            trntensor.set_backend(prev)

    def test_specs_beyond_the_tiling_take_the_pytorch_path(self, kernels, monkeypatch):
        U = 4096  # a 64 KiB output row per edge: more than one SBUF tile budget
        big = trntensor.conv_tp_spec(
            [(U, 0)], [(1, 0), (1, 1)], [(U, 0), (U, 1)], [(0, 0, 0), (0, 1, 1)]
        )
        x, y, w = torch.randn(3, U), torch.randn(3, 4), torch.randn(3, 2 * U)
        assert not eqk.conv_tp_kernel_route(x, y, w, big)
        monkeypatch.setenv("TRNTENSOR_REQUIRE_NKI", "1")
        with pytest.raises(NotImplementedError, match="tiling"):
            eqk.conv_tp_kernel_route(x, y, w, big)
