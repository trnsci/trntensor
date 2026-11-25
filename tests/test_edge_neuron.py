"""Per-edge and node MLIP ops on Trainium (native TorchNeuron path): correctness and timing.

For the next hardware window (trnsci/trnsci#178):

    pytest tests/test_edge_neuron.py -m neuron -s
    TRNTENSOR_EDGE_BENCH_JSON=edge_bench.json pytest tests/test_edge_neuron.py -m neuron -s -k timing

- ``TestShapes``: ``spherical_harmonics`` and ``edge_chain`` on the NKI kernels
  (``set_backend("nki")``) at the 216-atom (E = 10,392) and 1728-atom
  (E = 96,278) shapes, forward and input gradient, against the fp64 PyTorch
  path on the host. The edge chain has MACE-MP-0b2 medium's shape (8 Bessel
  functions, Agnesi transform, 8→64→64→64→512 / 1280 heads and two density
  heads) with synthetic weights. Bars: 2e-5 relative for outputs, 5e-5 for
  the VJP (the fp32 conditioning floor of the chain is ~1e-5 at 216 atoms).
- ``test_fixture``: the committed MACE-MP-0b2 fixture through the kernels.
- ``TestRegion``: the same ops and ``equivariant_linear`` with
  ``set_region_backend("neuron")`` (one compiled graph per direction).
- ``test_timing``: warm medians (ms) of forward and forward + backward, and
  the op count per call, for each variant: ``nki`` (the kernels), ``region``
  (compiled PyTorch bodies), ``eager`` (the PyTorch path on the device) and,
  when e3nn is importable, ``e3nn`` (the unfused module, MACE's formulation).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from test_edge import FIXTURE, fixture_spec
from test_edge_nki import mace_like_spec

import trntensor
from trntensor.nki import HAS_NKI
from trntensor.nki.dispatch import _get_dispatch_path, _use_simulator


def _native() -> bool:
    if not HAS_NKI or _use_simulator():
        return False
    try:
        return _get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

DEV = "neuron"
SHAPES = {"216": 10392, "1728": 96278}
RTOL_OUT, RTOL_GRAD = 2e-5, 5e-5


@pytest.fixture
def nki_backend():
    prev = trntensor.get_backend()
    trntensor.set_backend("nki")
    yield
    trntensor.set_backend(prev)


@pytest.fixture
def region_backend():
    prev = trntensor.get_region_backend()
    trntensor.set_region_backend("neuron")
    yield
    trntensor.set_region_backend(prev)


def _sync():
    torch.neuron.synchronize()


def _rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / b.norm())


def _edge_inputs(E, spec, seed=0):
    g = torch.Generator().manual_seed(seed)
    # neighbour-list-like lengths, some past r_max (MD buckets pad with far edges)
    r = 0.9 + 4.6 * torch.rand(E, 1, generator=g)
    r0 = 0.6 + 0.8 * torch.rand(E, 1, generator=g)
    cot = [torch.randn(E, d, generator=g) for d in spec.dims_out]
    return r, r0, cot


def _vectors(E, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(E, 3, generator=g)
    return v * (0.9 + 4.0 * torch.rand(E, 1, generator=g)), torch.randn(E, 16, generator=g)


def _host_fp64(fn):
    prev = trntensor.get_backend()
    trntensor.set_backend("pytorch")
    try:
        return fn()
    finally:
        trntensor.set_backend(prev)


def _vjp(fn, x, cot, extra=()):
    x = x.detach().requires_grad_(True)
    out = fn(x, *extra)
    outs = out if isinstance(out, tuple) else (out,)
    (g,) = torch.autograd.grad(outs, x, cot)
    return [o.detach() for o in outs], g


class TestShapes:
    @pytest.mark.parametrize("atoms", sorted(SHAPES))
    def test_spherical_harmonics(self, nki_backend, atoms):
        E = SHAPES[atoms]
        v, go = _vectors(E)
        y, dv = _vjp(trntensor.spherical_harmonics, v.to(DEV), [go.to(DEV)])
        _sync()
        yr, dvr = _host_fp64(lambda: _vjp(trntensor.spherical_harmonics, v.double(), [go.double()]))
        print(f"sh {atoms}: out {_rel(y[0], yr[0]):.2e} dv {_rel(dv, dvr):.2e}")
        assert _rel(y[0], yr[0]) < RTOL_OUT and _rel(dv, dvr) < RTOL_GRAD

    @pytest.mark.parametrize("atoms", sorted(SHAPES))
    def test_edge_chain(self, nki_backend, atoms):
        E = SHAPES[atoms]
        spec = mace_like_spec()
        r, r0, cot = _edge_inputs(E, spec)
        outs, dr = _vjp(
            lambda t, z: trntensor.edge_chain(t, spec, z),
            r.to(DEV),
            [c.to(DEV) for c in cot],
            (r0.to(DEV),),
        )
        _sync()
        ro, rdr = _host_fp64(
            lambda: _vjp(
                lambda t, z: trntensor.edge_chain(t, spec, z),
                r.double(),
                [c.double() for c in cot],
                (r0.double(),),
            )
        )
        errs = [_rel(a, b) for a, b in zip(outs, ro, strict=True)]
        print(f"edge_chain {atoms}: outs {[f'{e:.2e}' for e in errs]} dr {_rel(dr, rdr):.2e}")
        assert max(errs) < RTOL_OUT and _rel(dr, rdr) < RTOL_GRAD

    def test_reruns_are_bitwise_identical(self, nki_backend):
        spec = mace_like_spec()
        r, r0, cot = _edge_inputs(SHAPES["216"], spec)
        runs = [
            _vjp(
                lambda t, z: trntensor.edge_chain(t, spec, z),
                r.to(DEV),
                [c.to(DEV) for c in cot],
                (r0.to(DEV),),
            )
            for _ in range(2)
        ]
        _sync()
        for a, b in zip(runs[0][0] + [runs[0][1]], runs[1][0] + [runs[1][1]], strict=True):
            assert torch.equal(a.cpu(), b.cpu())


def test_fixture(nki_backend):
    fx = torch.load(FIXTURE, weights_only=True)
    d, s = fx["edge_chain"], fx["sh"]
    spec = fixture_spec(fx)
    outs, dl = _vjp(
        lambda t, z: trntensor.edge_chain(t, spec, z),
        d["lengths"].to(DEV),
        [g.to(DEV) for g in d["grad_out"]],
        (d["r0"].to(DEV),),
    )
    y, dv = _vjp(trntensor.spherical_harmonics, s["vectors"].to(DEV), [s["grad_out"].to(DEV)])
    _sync()
    truth = _host_fp64(lambda: trntensor.edge_chain(d["lengths"].double(), spec, d["r0"].double()))
    errs = {
        "edge out vs fp64": max(_rel(o, t) for o, t in zip(outs, truth, strict=True)),
        "edge out MACE32 vs fp64": max(_rel(o, t) for o, t in zip(d["out"], truth, strict=True)),
        "edge dr vs fp64": _rel(dl, d["grad_lengths_fp64"]),
        "edge dr MACE32 vs fp64": _rel(d["grad_lengths"], d["grad_lengths_fp64"]),
        "sh out vs MACE": _rel(y[0], s["out"]),
        "sh dv vs MACE": _rel(dv, s["grad_vectors"]),
    }
    print("fixture: " + ", ".join(f"{k} {v:.2e}" for k, v in errs.items()))
    assert errs["edge out vs fp64"] < max(3 * errs["edge out MACE32 vs fp64"], 2e-6)
    assert errs["edge dr vs fp64"] < max(3 * errs["edge dr MACE32 vs fp64"], 2e-6)
    assert errs["sh out vs MACE"] < 2e-6 and errs["sh dv vs MACE"] < 2e-6


class TestRegion:
    def test_ops_match_host(self, region_backend):
        spec = mace_like_spec()
        E = SHAPES["216"]
        r, r0, cot = _edge_inputs(E, spec)
        v, go = _vectors(E)
        outs, dr = _vjp(
            lambda t, z: trntensor.edge_chain(t, spec, z),
            r.to(DEV),
            [c.to(DEV) for c in cot],
            (r0.to(DEV),),
        )
        y, dv = _vjp(trntensor.spherical_harmonics, v.to(DEV), [go.to(DEV)])
        _sync()
        ro, rdr = _host_fp64(
            lambda: _vjp(
                lambda t, z: trntensor.edge_chain(t, spec, z),
                r.double(),
                [c.double() for c in cot],
                (r0.double(),),
            )
        )
        yr, dvr = _host_fp64(lambda: _vjp(trntensor.spherical_harmonics, v.double(), [go.double()]))
        assert max(_rel(a, b) for a, b in zip(outs, ro, strict=True)) < RTOL_OUT
        assert (
            _rel(dr, rdr) < RTOL_GRAD and _rel(y[0], yr[0]) < RTOL_OUT and _rel(dv, dvr) < RTOL_GRAD
        )

    def test_equivariant_linear(self, region_backend):
        g = torch.Generator().manual_seed(5)
        mul, d = 128, [1, 3, 5, 7]
        M = torch.zeros(sum(mul * k for k in d), sum(mul * k for k in d), dtype=torch.float64)
        o = 0
        for k in d:
            W = torch.randn(mul, mul, generator=g, dtype=torch.float64) / mul**0.5
            M[o : o + mul * k, o : o + mul * k] = torch.kron(W, torch.eye(k, dtype=torch.float64))
            o += mul * k
        spec = trntensor.equivariant_linear_spec(
            irreps_in="128x0e+128x1o+128x2e+128x3o",
            irreps_out="128x0e+128x1o+128x2e+128x3o",
            matrix=M,
        )
        x = torch.randn(216, spec.dim_in, generator=g)
        go = torch.randn(216, spec.dim_out, generator=g)
        y, dx = _vjp(lambda t: trntensor.equivariant_linear(t, spec), x.to(DEV), [go.to(DEV)])
        _sync()
        yr, dxr = _vjp(lambda t: trntensor.equivariant_linear(t, spec), x.double(), [go.double()])
        assert _rel(y[0], yr[0]) < RTOL_OUT and _rel(dx, dxr) < RTOL_OUT


# ─────────────────────────────────────────────────────────────── timing harness


class _Count(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not getattr(func, "is_view", False):
            self.n += 1
        return func(*args, **(kwargs or {}))


def _time(fn, warm=3, reps=10, budget_s=20.0):
    for _ in range(warm):
        fn()
    _sync()
    t0 = time.perf_counter()
    fn()
    _sync()
    one = time.perf_counter() - t0
    if one > 0.05:
        reps = max(3, min(reps, int(budget_s / one)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def _count(fn) -> int:
    with _Count() as c:
        fn()
    _sync()
    return c.n


def _variant(name):
    """Context for one timing variant: (backend, region backend)."""

    class V:
        def __enter__(self):
            self.prev = (trntensor.get_backend(), trntensor.get_region_backend())
            trntensor.set_backend("nki" if name == "nki" else "pytorch")
            trntensor.set_region_backend("neuron" if name == "region" else None)

        def __exit__(self, *a):
            trntensor.set_backend(self.prev[0])
            trntensor.set_region_backend(self.prev[1])

    return V()


def _e3nn_modules(spec):
    try:
        torch.serialization.add_safe_globals([slice])
        from e3nn import o3
    except Exception:
        return None
    return o3.SphericalHarmonics([0, 1, 2, 3], True, "component").to(DEV)


def _emit(rows, row):
    rows.append(row)
    print("BENCH " + json.dumps(row), flush=True)


@pytest.mark.parametrize("atoms", sorted(SHAPES))
def test_timing(atoms):
    """Warm medians (ms) and op counts per call; prints a table, writes JSON.

    ``TRNTENSOR_EDGE_BENCH_VARIANTS=nki,region,eager,e3nn`` and
    ``TRNTENSOR_EDGE_BENCH_OPS=sh,edge_chain,linear`` select subsets."""
    E = SHAPES[atoms]
    N = int(atoms)
    variants = set(
        os.environ.get("TRNTENSOR_EDGE_BENCH_VARIANTS", "nki,region,eager,e3nn").split(",")
    )
    ops = set(os.environ.get("TRNTENSOR_EDGE_BENCH_OPS", "sh,edge_chain,linear").split(","))
    spec = mace_like_spec()
    r, r0, cot = _edge_inputs(E, spec)
    rd, r0d, cotd = r.to(DEV), r0.to(DEV), [c.to(DEV) for c in cot]
    v, go = _vectors(E)
    vd, god = v.to(DEV), go.to(DEV)
    cases = {
        "sh": (lambda t: trntensor.spherical_harmonics(t), vd, [god]),
        "edge_chain": (lambda t: trntensor.edge_chain(t, spec, r0d), rd, cotd),
    }
    g = torch.Generator().manual_seed(7)
    lspec = trntensor.equivariant_linear_spec(
        irreps_in="128x0e+128x1o+128x2e+128x3o",
        irreps_out="128x0e+128x1o+128x2e+128x3o",
        matrix=torch.block_diag(
            *[
                torch.kron(
                    torch.randn(128, 128, generator=g, dtype=torch.float64) / 128**0.5,
                    torch.eye(k, dtype=torch.float64),
                )
                for k in (1, 3, 5, 7)
            ]
        ),
    )
    xl = torch.randn(N, lspec.dim_in, generator=g).to(DEV)
    gl = torch.randn(N, lspec.dim_out, generator=g).to(DEV)
    cases["linear"] = (lambda t: trntensor.equivariant_linear(t, lspec), xl, [gl])
    sh_e3nn = _e3nn_modules(spec) if "e3nn" in variants else None
    rows = []
    for op, (fn, x, ct) in cases.items():
        if op not in ops:
            continue
        names = [n for n in ("nki", "region", "eager") if n in variants]
        if op == "linear" and "nki" in names:
            names.remove("nki")  # no kernel: a matmul per irrep type, compiled or eager
        if op == "sh" and sh_e3nn is not None:
            names.append("e3nn")
        for name in names:
            f = sh_e3nn if name == "e3nn" else fn

            def fwd(f=f, x=x):
                with torch.no_grad():
                    return f(x)

            def fwdbwd(f=f, x=x, ct=ct):
                xs = x.detach().requires_grad_(True)
                out = f(xs)
                outs = out if isinstance(out, tuple) else (out,)
                return torch.autograd.grad(outs, xs, ct)

            try:
                with _variant(name if name != "e3nn" else "eager"):
                    _emit(
                        rows,
                        {
                            "op": op,
                            "atoms": N,
                            "E": E,
                            "variant": name,
                            "fwd_ms": _time(fwd),
                            "fwd_bwd_ms": _time(fwdbwd),
                            "fwd_ops": _count(fwd),
                            "fwd_bwd_ops": _count(fwdbwd),
                        },
                    )
            except Exception as exc:  # record, do not hide: a variant may not run on device
                _emit(rows, {"op": op, "atoms": N, "variant": name, "error": repr(exc)[:300]})
    print(
        f"\n{'op':12s} {'atoms':>5s} {'variant':8s} {'fwd ms':>9s} {'f+b ms':>9s} {'ops f':>6s} {'ops f+b':>7s}"
    )
    for r_ in rows:
        if "error" in r_:
            print(f"{r_['op']:12s} {r_['atoms']:5d} {r_['variant']:8s} ERROR {r_['error']}")
        else:
            print(
                f"{r_['op']:12s} {r_['atoms']:5d} {r_['variant']:8s} {r_['fwd_ms']:9.3f} "
                f"{r_['fwd_bwd_ms']:9.3f} {r_['fwd_ops']:6d} {r_['fwd_bwd_ops']:7d}"
            )
    path = os.environ.get("TRNTENSOR_EDGE_BENCH_JSON")
    if path:
        prev = json.loads(Path(path).read_text()) if Path(path).exists() else []
        Path(path).write_text(json.dumps(prev + rows, indent=1))
    if "nki" in variants:
        assert any(r_["variant"] == "nki" and "error" not in r_ for r_ in rows)
