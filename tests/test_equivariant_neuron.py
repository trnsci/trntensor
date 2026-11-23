"""Equivariant kernels on Trainium (native TorchNeuron path): correctness and timing.

Run in a hardware window (trnsci/trntensor#138, MLIP exemplar trnsci/trnsci#170):

    pytest tests/test_equivariant_neuron.py -m neuron -s
    TRNTENSOR_MLIP_CAPTURES=~/captures pytest tests/test_equivariant_neuron.py -m neuron -s -k capture
    TRNTENSOR_EQUIVARIANT_BENCH_JSON=bench.json pytest tests/test_equivariant_neuron.py -m neuron -s -k timing

- ``TestShapes``: forward and every needed input gradient of both layers at the
  216-atom (E = 10,392, N = 216) and 1728-atom (E = 96,278, N = 1,728) shapes,
  on the NKI kernels (``set_backend("nki")``), against the fp64 PyTorch path on
  the host (computed in edge / node chunks: rows are independent). Bar: 2e-5
  relative (the contract's bar against MACE).
- ``test_capture``: every capture in ``$TRNTENSOR_MLIP_CAPTURES`` (written by the
  exemplar's ``capture.py``) against MACE's own tensors, bar 2e-5.
- ``test_timing``: warm medians of the NKI kernels against the aten paths on
  the device, forward and forward + backward, and the aten op count per call:
  ``nki`` (this kernel), ``fused`` (trntensor's PyTorch path on the device),
  ``per_path`` (the exemplar's per-path einsum reference, e3nn's formulation),
  and ``e3nn`` (the e3nn ``TensorProduct`` itself, when e3nn is importable).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

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
RTOL = 2e-5
SHAPES = {"216": (216, 10392), "1728": (1728, 96278)}
CHUNK = 8192


@pytest.fixture
def nki_backend():
    prev = trntensor.get_backend()
    trntensor.set_backend("nki")
    yield
    trntensor.set_backend(prev)


def _sync():
    torch.neuron.synchronize()


def _dev(t, grad=False):
    # Host tensors copied to the device (aws-neuron-sdk#1439: no device RNG use).
    t = t.to(DEV)
    return t.requires_grad_(True) if grad else t


def _rel_parts(got_parts, ref_parts) -> float:
    num = sum(
        float((g.double() - r).norm() ** 2) for g, r in zip(got_parts, ref_parts, strict=True)
    )
    den = sum(float(r.norm() ** 2) for r in ref_parts)
    return (num / den) ** 0.5


def mace_conv_tp_spec(layer: int, channels: int = 128):
    return trntensor.conv_tp_spec(*mace_conv_tp_irreps(layer, channels))


def mace_symcon_spec(layer: int, channels: int = 128, elements: int = 89, seed: int = 0):
    """MACE-MP coupling tensors (from the fixture) with 128-channel weights."""
    a = next(a for a in load_fixture()["A2"] if a["layer"] == layer)
    g = torch.Generator().manual_seed(seed)
    cons = []
    for con in a["spec"]["contractions"]:
        W = {
            k: torch.randn(elements, v.shape[1], channels, generator=g) / v.shape[1]
            for k, v in con["W"].items()
        }
        cons.append({**con, "W": W})
    return eq.symmetric_contraction_spec(contractions=cons)


def _conv_tp_inputs(spec, E, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(E, spec.dim_in1, generator=g)
    v = torch.nn.functional.normalize(torch.randn(E, 3, generator=g), dim=1)
    y = torch.randn(E, spec.dim_in2, generator=g)
    y[:, 1:4] = v  # l = 1 components of unit vectors (scale of real SH)
    w = torch.randn(E, spec.weight_numel, generator=g) * 0.1
    go = torch.randn(E, spec.dim_out, generator=g)
    return x, y, w, go


class TestShapes:
    @pytest.mark.parametrize("atoms", sorted(SHAPES))
    @pytest.mark.parametrize("layer", [0, 1])
    def test_conv_tp(self, nki_backend, atoms, layer):
        _, E = SHAPES[atoms]
        spec = mace_conv_tp_spec(layer)
        x, y, w, go = _conv_tp_inputs(spec, E)
        want = ["y", "w"] + (["x"] if layer == 1 else [])
        xd, yd, wd = _dev(x, "x" in want), _dev(y, True), _dev(w, True)
        out = trntensor.conv_tp(xd, yd, wd, spec)
        assert out.device.type == DEV and out.dtype == torch.float32
        grads = torch.autograd.grad(out, [{"x": xd, "y": yd, "w": wd}[k] for k in want], _dev(go))
        got = {"out": out.cpu()} | {k: g.cpu() for k, g in zip(want, grads, strict=True)}
        ref = {k: [] for k in got}
        for e0 in range(0, E, CHUNK):
            sl = slice(e0, min(E, e0 + CHUNK))
            xr, yr, wr = (t[sl].double() for t in (x, y, w))
            ref["out"].append(eq._conv_tp_forward_torch(xr, yr, wr, spec))
            dx, dy, dw = eq._conv_tp_backward_torch(go[sl].double(), xr, yr, wr, spec)
            for k, v in (("x", dx), ("y", dy), ("w", dw)):
                if k in ref:
                    ref[k].append(v)
        for k in got:
            parts = [got[k][e0 : e0 + CHUNK] for e0 in range(0, E, CHUNK)]
            err = _rel_parts(parts, ref[k])
            print(f"conv_tp L{layer} {atoms} atoms (E={E}) {k}: rel {err:.2e}")
            assert err < RTOL, (k, err)

    @pytest.mark.parametrize("atoms", sorted(SHAPES))
    @pytest.mark.parametrize("layer", [0, 1])
    def test_symmetric_contraction(self, nki_backend, atoms, layer):
        N, _ = SHAPES[atoms]
        spec = mace_symcon_spec(layer)
        g = torch.Generator().manual_seed(1)
        x = torch.randn(N, 128, 16, generator=g)
        z = torch.randint(0, 89, (N,), generator=g)
        xd = _dev(x, True)
        out = trntensor.symmetric_contraction(xd, _dev(z), spec)
        go = torch.randn(out.shape, generator=g)
        (gx,) = torch.autograd.grad(out, xd, _dev(go))
        ref = eq._symcon_forward_torch(x.double(), z, spec)
        rgx = eq._symcon_backward_torch(go.double(), x.double(), z, spec)
        for k, a, b in (("out", out.cpu(), ref), ("grad_x", gx.cpu(), rgx)):
            err = _rel_parts([a], [b])
            print(f"symmetric_contraction L{layer} {atoms} atoms (N={N}) {k}: rel {err:.2e}")
            assert err < RTOL, (k, err)


def _captures():
    root = os.environ.get("TRNTENSOR_MLIP_CAPTURES")
    return sorted(Path(root).expanduser().glob("*.pt")) if root else []


@pytest.mark.skipif(not _captures(), reason="set TRNTENSOR_MLIP_CAPTURES to a capture directory")
@pytest.mark.parametrize("path", _captures(), ids=lambda p: p.stem)
def test_capture(nki_backend, path):
    """The acceptance check of #138 (``check.py --blocks A1,A2``), in-suite."""
    cap = torch.load(path, map_location="cpu", weights_only=True)
    for a in cap["A1"]:
        want = [k for k in ("x", "y", "w") if a.get("grad_" + k) is not None]
        ins = {k: _dev(a[k], k in want) for k in ("x", "y", "w")}
        out = trntensor.conv_tp(ins["x"], ins["y"], ins["w"], a["spec"])
        gs = torch.autograd.grad(out, [ins[k] for k in want], _dev(a["grad_out"]))
        errs = {"out": _rel_parts([out.cpu()], [a["out"].double()])}
        errs |= {
            k: _rel_parts([g.cpu()], [a["grad_" + k].double()])
            for k, g in zip(want, gs, strict=True)
        }
        print(
            f"{path.stem} A1 L{a['layer']}: " + ", ".join(f"{k} {v:.2e}" for k, v in errs.items())
        )
        assert max(errs.values()) < RTOL, errs
    for a in cap["A2"]:
        x = _dev(a["x"], True)
        out = trntensor.symmetric_contraction(x, _dev(a["species"]), a["spec"])
        (gx,) = torch.autograd.grad(out, x, _dev(a["grad_out"]))
        errs = {
            "out": _rel_parts([out.cpu()], [a["out"].double()]),
            "grad_x": _rel_parts([gx.cpu()], [a["grad_x"].double()]),
        }
        print(
            f"{path.stem} A2 L{a['layer']}: " + ", ".join(f"{k} {v:.2e}" for k, v in errs.items())
        )
        assert max(errs.values()) < RTOL, errs


# ─────────────────────────────────────────────────────────────── timing harness


class _AtenCount(torch.utils._python_dispatch.TorchDispatchMode):
    """Counts the aten (and custom) ops one call dispatches, views excluded:
    on the device each of these is a launch."""

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
    with _AtenCount() as c:
        fn()
    _sync()
    return c.n


def _e3nn_tp(spec):
    try:
        torch.serialization.add_safe_globals([slice])
        from e3nn import o3
    except Exception:
        return None

    def irr(lst):
        return o3.Irreps([(m, (ell, (-1) ** ell)) for m, ell in lst])

    tp = o3.TensorProduct(
        irr(spec.irreps_in1),
        irr(spec.irreps_in2),
        irr(spec.irreps_out),
        [(i1, i2, io, "uvu", True) for i1, i2, io, _ in spec.instructions],
        shared_weights=False,
        internal_weights=False,
    )
    return tp.to(DEV)


def _backend(name):
    def run(fn):
        prev = trntensor.get_backend()
        trntensor.set_backend(name)
        try:
            return fn()
        finally:
            trntensor.set_backend(prev)

    return run


def _variants_conv_tp(spec):
    d = spec.as_dict()
    v = {
        "nki": lambda x, y, w: _backend("nki")(lambda: trntensor.conv_tp(x, y, w, spec)),
        "fused": lambda x, y, w: _backend("pytorch")(lambda: trntensor.conv_tp(x, y, w, spec)),
        "per_path": lambda x, y, w: conv_tp_reference(x, y, w, d),
    }
    tp = _e3nn_tp(spec)
    if tp is not None:
        v["e3nn"] = lambda x, y, w: tp(x, y, w)
    return v


def _variants_symcon(spec):
    def ref_con(c):
        U = {str(k): (u.squeeze(0) if c["d_out"] == 1 else u).float() for k, u in c["U"].items()}
        return {**c, "U": U, "W": {str(k): w.float() for k, w in c["W"].items()}}

    d = {"contractions": [ref_con(c) for c in spec.contractions]}
    return {
        "nki": lambda x, z: _backend("nki")(lambda: trntensor.symmetric_contraction(x, z, spec)),
        "fused": lambda x, z: _backend("pytorch")(
            lambda: trntensor.symmetric_contraction(x, z, spec)
        ),
        "per_path": lambda x, z: symmetric_contraction_reference(x, z, d),
    }


@pytest.mark.parametrize("atoms", sorted(SHAPES))
def test_timing(atoms):
    """Warm medians (ms) and aten op counts per call; prints a table, writes JSON."""
    N, E = SHAPES[atoms]
    rows = []
    for layer in (0, 1):
        spec = mace_conv_tp_spec(layer)
        x, y, w, go = _conv_tp_inputs(spec, E)
        xd, yd, wd, god = (t.to(DEV) for t in (x, y, w, go))
        need_x = layer == 1
        for name, fn in _variants_conv_tp(spec).items():

            def fwd(fn=fn, xd=xd, yd=yd, wd=wd):
                return fn(xd, yd, wd)

            def fwdbwd(fn=fn, xd=xd, yd=yd, wd=wd, god=god, need_x=need_x):
                xs = xd.detach().requires_grad_(need_x)
                ys, ws = yd.detach().requires_grad_(True), wd.detach().requires_grad_(True)
                out = fn(xs, ys, ws)
                return torch.autograd.grad(out, [t for t in (xs, ys, ws) if t.requires_grad], god)

            try:
                rows.append(
                    {
                        "block": "conv_tp",
                        "layer": layer,
                        "atoms": int(atoms),
                        "E": E,
                        "variant": name,
                        "fwd_ms": _time(fwd),
                        "fwd_bwd_ms": _time(fwdbwd),
                        "fwd_ops": _count(fwd),
                        "fwd_bwd_ops": _count(fwdbwd),
                    }
                )
            except Exception as exc:  # record, do not hide: a variant may not run on device
                rows.append(
                    {
                        "block": "conv_tp",
                        "layer": layer,
                        "atoms": int(atoms),
                        "variant": name,
                        "error": repr(exc)[:300],
                    }
                )
        sspec = mace_symcon_spec(layer)
        g = torch.Generator().manual_seed(2)
        xs0 = torch.randn(N, 128, 16, generator=g).to(DEV)
        z = torch.randint(0, 89, (N,), generator=g).to(DEV)
        sgo = torch.randn(N, sspec.dim_out, generator=g).to(DEV)
        for name, fn in _variants_symcon(sspec).items():

            def sfwd(fn=fn, xs0=xs0, z=z):
                return fn(xs0, z)

            def sfwdbwd(fn=fn, xs0=xs0, z=z, sgo=sgo):
                xs = xs0.detach().requires_grad_(True)
                return torch.autograd.grad(fn(xs, z), xs, sgo)

            try:
                rows.append(
                    {
                        "block": "symmetric_contraction",
                        "layer": layer,
                        "atoms": int(atoms),
                        "N": N,
                        "variant": name,
                        "fwd_ms": _time(sfwd),
                        "fwd_bwd_ms": _time(sfwdbwd),
                        "fwd_ops": _count(sfwd),
                        "fwd_bwd_ops": _count(sfwdbwd),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "block": "symmetric_contraction",
                        "layer": layer,
                        "atoms": int(atoms),
                        "variant": name,
                        "error": repr(exc)[:300],
                    }
                )
    print(
        f"\n{'block':24s} {'L':>2s} {'atoms':>5s} {'variant':9s} {'fwd ms':>9s} {'f+b ms':>9s} {'ops f':>6s} {'ops f+b':>7s}"
    )
    for r in rows:
        if "error" in r:
            print(
                f"{r['block']:24s} {r['layer']:2d} {r['atoms']:5d} {r['variant']:9s} ERROR {r['error']}"
            )
        else:
            print(
                f"{r['block']:24s} {r['layer']:2d} {r['atoms']:5d} {r['variant']:9s} "
                f"{r['fwd_ms']:9.3f} {r['fwd_bwd_ms']:9.3f} {r['fwd_ops']:6d} {r['fwd_bwd_ops']:7d}"
            )
    path = os.environ.get("TRNTENSOR_EQUIVARIANT_BENCH_JSON")
    if path:
        prev = json.loads(Path(path).read_text()) if Path(path).exists() else []
        Path(path).write_text(json.dumps(prev + rows, indent=1))
    assert any(r["variant"] == "nki" and "error" not in r for r in rows)
