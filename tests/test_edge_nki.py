"""Per-edge NKI kernels on CPU: MLIR verification and ``nki.simulate`` (trnsci/trnsci#178).

Needs ``nki`` (x86_64 Linux; the CI ``nki-simulator`` job) but no device.

- ``test_mlir_verifies``: the four kernels (spherical harmonics and edge
  chain, forward and backward) at the MACE-MP-0b2 shapes trace to MLIR that
  passes ``verify()`` for trn1 and trn2 at LNC 1 and 2. The edge counts
  cover one chunk, several chunks with a partial last one, and the 216-atom
  bucket (10,752 edges, 84 columns).
- ``TestSimulated``: the kernels behind the public API (``set_backend("nki")``,
  ``TRNTENSOR_USE_SIMULATOR=1``), forward and backward through autograd,
  against MACE (the committed fixture) and the fp64 PyTorch path.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
from test_edge import FIXTURE, fixture_spec

import trntensor
from trntensor import edge
from trntensor.nki import _edge as eh
from trntensor.nki._edge_kernels import HAS_NKI

pytestmark = pytest.mark.nki_simulator

try:
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    source_to_mlir = None


@pytest.fixture(autouse=True)
def _nki_available():
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")


def mace_like_spec(d0=512, d1=1280, apply_cutoff=True, p=5, seed=0):
    """MACE-MP-0b2 medium's edge chain shape (8 Bessel, Agnesi, two 8→64→64→64
    heads and two 8→1 density heads) with synthetic weights."""
    g = torch.Generator().manual_seed(seed)

    def mlp(hs):
        return [
            torch.randn(a, b, generator=g, dtype=torch.float64) / math.sqrt(a)
            for a, b in zip(hs, hs[1:], strict=False)
        ]

    heads = [
        edge.EdgeMLP(tuple(mlp([8, 64, 64, 64, d0])), None, "tp_w0"),
        edge.EdgeMLP(tuple(mlp([8, 1])), "tanh_sq", "density0"),
        edge.EdgeMLP(tuple(mlp([8, 64, 64, 64, d1])), None, "tp_w1"),
        edge.EdgeMLP(tuple(mlp([8, 1])), "tanh_sq", "density1"),
    ]
    return edge.edge_chain_spec(
        r_max=5.0,
        num_basis=8,
        cutoff_p=p,
        agnesi=(1.0805, 0.9183, 4.5791),
        mlps=heads,
        apply_cutoff=apply_cutoff,
    )


def _sh_case(E):
    g = torch.Generator().manual_seed(0)
    v2 = torch.randn(E, 3, generator=g).reshape(128, -1).numpy()
    table, dtable = eh.sh_tables(3, "component")
    g2 = torch.randn(128, (E // 128) * 16, generator=g).numpy()
    kw = {"lmax": 3, "normalize": 1, "eps": 1e-12}
    return ((v2,), {**kw, "table": table}), ((g2, v2), {**kw, "dtable": dtable})


def _edge_case(E, **spec_kw):
    spec = mace_like_spec(**spec_kw)
    k = eh._edge_kernel_constants(spec, "cpu")
    g = torch.Generator().manual_seed(1)
    r = (0.8 + 5.0 * torch.rand(E, 1, generator=g)).reshape(128, -1).numpy()
    r0 = (0.6 + 0.8 * torch.rand(E, 1, generator=g)).reshape(128, -1).numpy()
    DO = sum(spec.dims_out)
    g2 = torch.randn(128, (E // 128) * DO, generator=g).numpy()
    wp, wt = k["wpack"].numpy(), k["wtpack"].numpy()
    kw = {"dims": k["dims"], "consts": k["consts"], "bessel": k["bessel"]}
    return ((r, r0, wp), kw), ((g2, r, r0, wp, wt), kw)


CASES = {
    "sh_E512": lambda: _sh_case(512),
    "sh_E10752": lambda: _sh_case(10752),
    "edge_E512": lambda: _edge_case(512),
    "edge_E5376": lambda: _edge_case(5376),
    "edge_E10752": lambda: _edge_case(10752),
    "edge_E1024_no_cutoff_p6": lambda: _edge_case(1024, apply_cutoff=False, p=6),
}
KERNELS = {
    "sh": ("sh_fwd_kernel", "sh_bwd_kernel"),
    "edge": ("edge_chain_fwd_kernel", "edge_chain_bwd_kernel"),
}


@pytest.mark.parametrize("target,lnc", [("trn1", 1), ("trn2", 1), ("trn2", 2)])
@pytest.mark.parametrize("direction", [0, 1], ids=["fwd", "bwd"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_mlir_verifies(case, direction, target, lnc):
    if source_to_mlir is None:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available")
    from trntensor.nki import _edge_kernels as xk

    kernel = getattr(xk, KERNELS[case.split("_")[0]][direction])
    args, kwargs = CASES[case]()[direction]
    source_to_mlir(
        kernel, *args, target=target, lnc=lnc, frontend=FrontendKind.TRACER, **kwargs
    ).verify()


@pytest.mark.parametrize("case", ["sh_E512", "edge_E512", "edge_E1024_no_cutoff_p6"])
def test_simulated_kernels_match_their_contracts(case):
    """Kernel vs the differentiable statement of its contract (fp64)."""
    import nki

    from trntensor.nki import _edge_kernels as xk

    names = KERNELS[case.split("_")[0]]
    fwd, bwd = CASES[case]()
    refs = (
        (eh.sh_fwd_torch, eh.sh_bwd_torch)
        if case.startswith("sh")
        else (eh.edge_chain_fwd_torch, eh.edge_chain_bwd_torch)
    )
    for name, (args, kw), ref in zip(names, (fwd, bwd), refs, strict=True):
        got = torch.from_numpy(np.asarray(nki.simulate(getattr(xk, name))(*args, **kw))).double()
        targs = [torch.from_numpy(a).double() for a in args]
        if name == "edge_chain_bwd_kernel":
            targs = targs[:4]  # the statement reads the forward pack only
        want = ref(*targs, **kw)
        assert float((got - want).norm() / want.norm()) < 2e-6, name


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
    def test_sh_matches_mace(self, simulated):
        s = torch.load(FIXTURE, weights_only=True)["sh"]
        v = s["vectors"].clone().requires_grad_(True)
        y = trntensor.spherical_harmonics(v)
        assert _rel(y, s["out"]) < 2e-6
        (dv,) = torch.autograd.grad(y, v, s["grad_out"])
        assert _rel(dv, s["grad_vectors"]) < 2e-6

    def test_edge_chain_matches_mace(self, simulated):
        fx = torch.load(FIXTURE, weights_only=True)
        d = fx["edge_chain"]
        spec = fixture_spec(fx)
        L = d["lengths"].clone().requires_grad_(True)
        outs = trntensor.edge_chain(L, spec, d["r0"])
        # fp64 truth: the PyTorch path in fp64 on the same inputs (it matches
        # MACE in fp64 to 1e-15). The kernel's fp32 rounding differs from
        # MACE's (exp/log for the powers), so bound it at the scale of MACE's
        # own fp32 error against that truth.
        with pytest.MonkeyPatch.context() as m:
            m.setenv("TRNTENSOR_FORCE_BACKEND", "pytorch")
            truth = trntensor.edge_chain(d["lengths"].double(), spec, d["r0"].double())
        for o, ref, t in zip(outs, d["out"], truth, strict=True):
            assert _rel(o, t) < max(3 * _rel(ref, t), 2e-6)
            assert _rel(o, ref) < 2e-5
        (dl,) = torch.autograd.grad(outs, L, d["grad_out"])
        assert _rel(dl, d["grad_lengths_fp64"]) < max(
            3 * _rel(d["grad_lengths"], d["grad_lengths_fp64"]), 2e-6
        )

    @pytest.mark.skipif(not os.environ.get("TRNTENSOR_SIM_SLOW"), reason="set TRNTENSOR_SIM_SLOW=1")
    def test_edge_chain_mace_shape_matches_pytorch_fp64(self, simulated):
        spec = mace_like_spec()
        g = torch.Generator().manual_seed(3)
        E = 1000  # pads to 1024
        r = (0.8 + 5.0 * torch.rand(E, 1, generator=g)).requires_grad_(True)
        r0 = 0.6 + 0.8 * torch.rand(E, 1, generator=g)
        outs = trntensor.edge_chain(r, spec, r0)
        cot = [torch.randn(o.shape, generator=g) for o in outs]
        (dr,) = torch.autograd.grad(outs, r, cot)
        rd = r.detach().double().requires_grad_(True)
        with pytest.MonkeyPatch.context() as m:
            m.setenv("TRNTENSOR_FORCE_BACKEND", "pytorch")
            ref = trntensor.edge_chain(rd, spec, r0.double())
        (rdr,) = torch.autograd.grad(ref, rd, [c.double() for c in cot])
        for a, b in zip(outs, ref, strict=True):
            assert _rel(a, b) < 2e-6
        assert _rel(dr, rdr) < 2e-6
