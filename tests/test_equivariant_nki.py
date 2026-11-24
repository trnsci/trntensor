"""Equivariant NKI kernels on CPU: MLIR verification and ``nki.simulate``.

Needs ``nki`` (x86_64 Linux; the CI ``nki-simulator`` job) but no device.

- ``test_mlir_verifies``: every kernel at the MACE-MP-0b2 shapes (and tiling
  edge cases) traces to MLIR that passes ``verify()`` for trn1 and trn2 at
  LNC 1 and 2, as ``tests/test_nki_compile.py`` does for the other kernels.
- ``TestSimulated``: the kernels run through ``nki.simulate`` behind the public
  API (``set_backend("nki")``, ``TRNTENSOR_USE_SIMULATOR=1``), forward and
  backward through ``torch.autograd``, against MACE's captured outputs and
  gradients (the committed fixture) and the fp64 PyTorch path.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from mlip_reference import load_fixture, mace_conv_tp_irreps

import trntensor
from trntensor import equivariant as eq
from trntensor.nki._equivariant_kernels import HAS_NKI

pytestmark = pytest.mark.nki_simulator

try:
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    source_to_mlir = None


@pytest.fixture(autouse=True)
def _nki_available():
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")


def _np(t):
    return t.detach().cpu().numpy()


def _conv_tp_case(layer, channels=128, E=256, spec=None):
    spec = spec or trntensor.conv_tp_spec(*mace_conv_tp_irreps(layer, channels))
    lay = spec.layout
    c = spec.constants("cpu", torch.float32)
    g = torch.Generator().manual_seed(0)
    x, y, w, go = (
        torch.randn(E, n, generator=g)
        for n in (spec.dim_in1, spec.dim_in2, spec.weight_numel, spec.dim_out)
    )
    fwd = (
        (_np(x), _np(y), _np(w), _np(c["cg_kernel"])),
        {"plan": lay.plan, "zero_fill": lay.zero_fill},
    )
    bwd = (
        (_np(go), _np(x), _np(y), _np(w), _np(c["cg_kernel"]), _np(c["cg_kernel_T"])),
        {"plan": lay.plan},
    )
    return fwd, bwd


def _symcon_case(layer, N=16):
    a = next(a for a in load_fixture()["A2"] if a["layer"] == layer)
    spec = eq._as_spec(a["spec"], eq.SymmetricContractionSpec)
    lay, c = spec.layout, spec.constants("cpu", torch.float32)
    C = lay.C
    x = torch.randn(N, C, lay.L)
    wr = c["W_zmc"].index_select(0, torch.zeros(N, dtype=torch.long))
    g = torch.randn(lay.Dsum, N * C)
    fwd = ((_np(x), _np(wr), _np(c["sel"]), _np(c["usym"]), _np(c["S"])), {})
    bwd = (
        (_np(g), _np(x), _np(wr), _np(c["sel"]), _np(c["selT"]), _np(c["usymT"]), _np(c["ST"])),
        {},
    )
    return fwd, bwd


def _mace_symcon_spec_128(layer):
    """The fixture's U (MACE-MP) with 128 channels of synthetic weights."""
    a = next(a for a in load_fixture()["A2"] if a["layer"] == layer)
    g = torch.Generator().manual_seed(1)
    cons = []
    for con in a["spec"]["contractions"]:
        W = {k: torch.randn(4, v.shape[1], 128, generator=g) for k, v in con["W"].items()}
        cons.append({**con, "W": W})
    return eq.symmetric_contraction_spec(contractions=cons)


def _symcon_case_128(layer, N=8):
    spec = _mace_symcon_spec_128(layer)
    lay, c = spec.layout, spec.constants("cpu", torch.float32)
    x = torch.randn(N, 128, 16)
    wr = c["W_zmc"].index_select(0, torch.arange(N) % 4)
    g = torch.randn(lay.Dsum, N * 128)
    fwd = ((_np(x), _np(wr), _np(c["sel"]), _np(c["usym"]), _np(c["S"])), {})
    bwd = (
        (_np(g), _np(x), _np(wr), _np(c["sel"]), _np(c["selT"]), _np(c["usymT"]), _np(c["ST"])),
        {},
    )
    return fwd, bwd


def _odd_conv_tp_spec():
    """Two paths into one output, an output no path writes, Q > 128 (two dG chunks)."""
    in1, in2 = [(16, 0), (16, 1)], [(1, ell) for ell in range(4)]
    out = [(16, 1), (16, 0), (16, 2), (16, 3)]
    ins = [(0, 1, 0), (1, 0, 0), (1, 1, 2), (1, 2, 2), (1, 2, 3), (1, 3, 2), (1, 3, 3), (1, 2, 0)]
    ins += [(1, 1, 0), (1, 1, 2)]  # Q = 132
    return trntensor.conv_tp_spec(in1, in2, out, ins)


CASES = {
    "conv_tp_L0": lambda: _conv_tp_case(0),
    "conv_tp_L1": lambda: _conv_tp_case(1),
    "conv_tp_L1_E512": lambda: _conv_tp_case(1, E=512),
    "conv_tp_odd_paths": lambda: _conv_tp_case(None, spec=_odd_conv_tp_spec()),
    "symcon_L0_C128": lambda: _symcon_case_128(0),
    "symcon_L1_C128": lambda: _symcon_case_128(1),
    "symcon_L0_C8": lambda: _symcon_case(0, N=128),
}
KERNELS = {
    "conv_tp": ("conv_tp_fwd_kernel", "conv_tp_bwd_kernel"),
    "symcon": ("symcon_fwd_kernel", "symcon_bwd_kernel"),
}


@pytest.mark.parametrize("target,lnc", [("trn1", 1), ("trn2", 1), ("trn2", 2)])
@pytest.mark.parametrize("direction", [0, 1], ids=["fwd", "bwd"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_mlir_verifies(case, direction, target, lnc):
    if source_to_mlir is None:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available")
    from trntensor.nki import _equivariant_kernels as ek

    family = "symcon" if case.startswith("symcon") else "conv_tp"
    kernel = getattr(ek, KERNELS[family][direction])
    args, kwargs = CASES[case]()[direction]
    source_to_mlir(
        kernel, *args, target=target, lnc=lnc, frontend=FrontendKind.TRACER, **kwargs
    ).verify()


def test_odd_spec_has_the_cases_it_claims():
    lay = _odd_conv_tp_spec().layout
    assert not lay.out_bijective and lay.zero_fill == 1 and lay.Q > 128
    assert any(p[5] for p in (lay.plan[1 + 6 * i : 7 + 6 * i] for i in range(lay.P)))


@pytest.fixture
def simulated(monkeypatch):
    monkeypatch.setenv("TRNTENSOR_USE_SIMULATOR", "1")
    from trntensor.nki import dispatch

    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", True)
    prev = trntensor.get_backend()
    trntensor.set_backend("nki")
    yield
    trntensor.set_backend(prev)


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm())


class TestSimulated:
    @pytest.mark.parametrize("layer", [0, 1])
    def test_conv_tp_matches_mace(self, simulated, layer):
        a = next(a for a in load_fixture()["A1"] if a["layer"] == layer)
        want = [k for k in ("x", "y", "w") if a.get("grad_" + k) is not None]
        ins = {k: a[k].clone().requires_grad_(k in want) for k in ("x", "y", "w")}
        out = trntensor.conv_tp(ins["x"], ins["y"], ins["w"], a["spec"])
        assert _rel(out, a["out"]) < 2e-5
        for k, gk in zip(
            want, torch.autograd.grad(out, [ins[k] for k in want], a["grad_out"]), strict=True
        ):
            assert _rel(gk, a["grad_" + k]) < 2e-5, k

    @pytest.mark.parametrize("layer", [0, 1])
    def test_symcon_matches_mace(self, simulated, layer):
        a = next(a for a in load_fixture()["A2"] if a["layer"] == layer)
        x = a["x"].clone().requires_grad_(True)
        out = trntensor.symmetric_contraction(x, a["species"], a["spec"])
        assert _rel(out, a["out"]) < 2e-5
        (gx,) = torch.autograd.grad(out, x, a["grad_out"])
        assert _rel(gx, a["grad_x"]) < 2e-5

    def test_conv_tp_odd_spec_matches_pytorch_fp64(self, simulated):
        spec = _odd_conv_tp_spec()
        g = torch.Generator().manual_seed(3)
        E = 300  # pads to 512: two tiles per program at LNC=2, a partial last tile of data
        x, y, w = (
            torch.randn(E, n, generator=g).requires_grad_(True)
            for n in (spec.dim_in1, spec.dim_in2, spec.weight_numel)
        )
        out = trntensor.conv_tp(x, y, w, spec)
        go = torch.randn(out.shape, generator=g)
        gs = torch.autograd.grad(out, (x, y, w), go)
        xd, yd, wd = (t.detach().double().requires_grad_(True) for t in (x, y, w))
        ref = eq._ConvTPTorch.apply(xd, yd, wd, spec)
        rgs = torch.autograd.grad(ref, (xd, yd, wd), go.double())
        assert _rel(out, ref) < 1e-6
        for a, b in zip(gs, rgs, strict=True):
            assert _rel(a, b) < 1e-6

    @pytest.mark.skipif(not os.environ.get("TRNTENSOR_SIM_SLOW"), reason="set TRNTENSOR_SIM_SLOW=1")
    def test_symcon_c128_matches_pytorch_fp64(self, simulated):
        spec = _mace_symcon_spec_128(0)
        x = torch.randn(8, 128, 16).requires_grad_(True)
        z = torch.arange(8) % 4
        out = trntensor.symmetric_contraction(x, z, spec)
        go = torch.randn(out.shape)
        (gx,) = torch.autograd.grad(out, x, go)
        xd = x.detach().double().requires_grad_(True)
        ref = eq._SymConTorch.apply(xd, z, spec)
        (rgx,) = torch.autograd.grad(ref, xd, go.double())
        assert _rel(out, ref) < 1e-6 and _rel(gx, rgx) < 1e-6
        assert np.isfinite(_np(gx)).all()
