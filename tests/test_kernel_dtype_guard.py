"""No NKI kernel is reached with an operand dtype it does not take.

Trainium has no fp64 datapath and the trntensor kernels are written for fp32
(bf16 / fp16 where listed below). Through 0.22.0 float64, integer, bool and
complex tensors could reach them unchanged and come back inf, garbage or in
the wrong dtype (#87, #90; trnsci/trnsci#35). These CPU tests stand in a fake
TorchNeuron ``torch_neuronx`` whose ``nki_op`` turns every registered
``trntensor::`` op into a stub that records any operand dtype outside the
kernel's allowed set, force the native dispatch path with CPU as the
``neuron`` device, and drive every public API that can reach a kernel with
every dtype, under ``set_backend("nki")`` and ``"auto"``. The stubs record
instead of raising, because the host wrappers catch kernel exceptions and
fall back to aten (which would hide the violation).

Results are also checked for the pytorch backend's dtype and shape (the stubs
return zeros, so values are not compared).

float64 follows the suite fp64 policy (#92, trnsci/trnsci#35): under
``"auto"`` it runs torch at float64 accuracy and reaches no kernel; under
explicit ``set_backend("nki")`` the kernel runs on fp32 operands with one
``UserWarning`` per op per process; under ``TRNTENSOR_REQUIRE_NKI=1`` it raises
``NotImplementedError``. The result dtype is always the pytorch backend's,
including mixed-dtype promotion. float64 / complex128 already on the
``neuron`` device is computed on the host under ``"auto"`` and returned on the
device (device aten computes float64 at fp32 precision), unless
``TRNTENSOR_FP64_ON_DEVICE=1``; see the last section.
"""

import sys
import types
import warnings

import pytest
import torch

import trntensor
import trntensor.nki
from trntensor import parallel
from trntensor.nki import _kernels, dispatch

F32, BF16, F16 = torch.float32, torch.bfloat16, torch.float16
_MATMUL = {F32, BF16, F16}
_CHEM = {F32, BF16}
_IDX = {torch.uint32}

# What each kernel takes, per positional tensor operand (independent of the
# library's own table, so a too-permissive table entry fails here too).
KERNEL_TAKES = {
    "matmul": (_MATMUL,) * 2,
    "batched_matmul": (_MATMUL,) * 2,
    "mp2_energy": (_CHEM,) * 4,
    "mp2_energy_row": (_CHEM,) * 6,
    "ao_to_mo_transform": (_CHEM,) * 3,
    "ao_to_mo_trn2": (_CHEM,) * 3,
    "ao_to_mo_trn2_packed": (_CHEM, _IDX, _CHEM, _CHEM),
    "df_half_packed": ({F32}, _IDX, {F32}),
    "df_exchange_syrk": ({F32},) * 2,
    "df_coulomb_d": ({F32},) * 2,
    "df_coulomb_j": ({F32},) * 2,
}
SAME_DTYPE = {"matmul", "batched_matmul"}
COLLECTIVE_TAKES = _MATMUL

DTYPES = [
    torch.float64,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]
_RAW_KERNELS = [
    "matmul_kernel",
    "batched_matmul_kernel",
    "mp2_energy_kernel",
    "ao_to_mo_transform_kernel",
    "mp2_energy_row_kernel",
    "ao_to_mo_trn2_kernel",
    "ao_to_mo_trn2_packed_kernel",
    "df_half_packed_kernel",
    "df_exchange_syrk_kernel",
    "df_coulomb_d_kernel",
    "df_coulomb_j_kernel",
]


class _Recorder:
    def __init__(self):
        self.calls: list[str] = []
        self.violations: list[str] = []
        self.float_dtypes: set[torch.dtype] = set()  # floating operand dtypes seen


def _fake_torch_neuronx(rec: _Recorder):
    mod = types.ModuleType("torch_neuronx")
    real = sys.modules.get("torch_neuronx")
    if real is not None:
        # On a Neuron host the real runtime keeps reading module state from
        # `sys.modules["torch_neuronx"]` while the stand-in is installed (e.g.
        # `torch.manual_seed` -> `torch_neuronx._is_in_bad_fork`), so the
        # stand-in starts as a copy of the real module; `nki_op` and
        # `device_count` are replaced below.
        mod.__dict__.update({k: v for k, v in vars(real).items() if not k.startswith("__")})

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        short = name.split("::", 1)[1]

        def dec(_wrapper):
            def stub(*args, **kwargs):
                rec.calls.append(short)
                ts = [a for a in args if isinstance(a, torch.Tensor)]
                takes = KERNEL_TAKES.get(short)
                dts = [t.dtype for t in ts]
                rec.float_dtypes.update(d for d in dts if d.is_floating_point)
                if takes is None:
                    rec.violations.append(f"{short}: no allowed-dtype entry in the test")
                elif len(dts) != len(takes) or any(
                    d not in a for d, a in zip(dts, takes, strict=False)
                ):
                    rec.violations.append(f"{short}: operand dtypes {dts}")
                elif short in SAME_DTYPE and len(set(dts)) > 1:
                    rec.violations.append(f"{short}: mixed operand dtypes {dts}")
                return torch.zeros_like(fake_fn(*args, **kwargs))

            return stub

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: 1
    return mod


def _set_backend(monkeypatch, backend):
    monkeypatch.setattr(dispatch, "_backend", backend)


@pytest.fixture
def guarded(monkeypatch):
    """Native path on CPU; every `trntensor::` op is a dtype-checking stub."""
    rec = _Recorder()
    monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_torch_neuronx(rec))
    # Record the real state first, so teardown removes the stub-backed `_native`
    # imported below (a later test in the session would otherwise get stub ops).
    monkeypatch.setitem(sys.modules, "trntensor.nki._native", None)
    monkeypatch.setattr(trntensor.nki, "_native", None, raising=False)
    monkeypatch.delitem(sys.modules, "trntensor.nki._native", raising=False)
    monkeypatch.delattr(trntensor.nki, "_native", raising=False)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setattr(dispatch, "_DEVICE_TYPES", ("xla", "cpu"))
    monkeypatch.setattr(dispatch, "_native_kernel_failed", set(), raising=False)
    for name in (
        "_MIN_NKI_FLOPS",
        "_MIN_NKI_FLOPS_PINNED",
        "_MIN_NKI_FLOPS_NATIVE",
        "_MIN_NKI_FLOPS_NATIVE_PINNED",
    ):
        monkeypatch.setattr(dispatch, name, 0)

    def no_xla(*tensors):
        raise AssertionError("the XLA path must not be used on the native path")

    def no_raw_kernel(*a, **k):
        raise AssertionError("raw kernel must not be called on the native path")

    monkeypatch.setattr(dispatch, "_to_xla", no_xla)
    for kname in _RAW_KERNELS:
        monkeypatch.setattr(_kernels, kname, no_raw_kernel, raising=False)
    for var in (
        "TRNTENSOR_FORCE_BACKEND",
        "TRNTENSOR_REQUIRE_NKI",
        "TRNTENSOR_FP64_ON_DEVICE",
        "TRNSCI_NKI_DISPATCH",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", False)
    monkeypatch.setattr(dispatch, "_fp64_warned", set())
    monkeypatch.setattr(dispatch, "_fp64_host_warned", set())
    monkeypatch.setenv("TRNTENSOR_HOST_ROUTE", "host")  # CPU is the "device"
    trntensor.clear_plan_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield rec
    trntensor.clear_plan_cache()


def _require_nki(monkeypatch):
    monkeypatch.setenv("TRNTENSOR_REQUIRE_NKI", "1")
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)


def _mk(shape, dtype, g):
    if dtype.is_complex:
        return torch.randn(shape, generator=g, dtype=dtype)
    if dtype.is_floating_point:
        return (0.1 * torch.randn(shape, generator=g)).to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    return torch.randint(-3, 4, shape, generator=g, dtype=dtype)


def _sym(eri):
    return eri + eri.transpose(0, 1)


# ── The public calls that can reach a kernel ─────────────────────────────────
# Each entry: (id, env overrides, fn(dtype, g) -> result).


def _einsum(sub, shapes, **kw):
    def run(dt, g):
        return trntensor.einsum(sub, *(_mk(s, dt, g) for s in shapes), **kw)

    return run


def _multi(dt, g):
    a, b, c, d = (_mk(s, dt, g) for s in [(6, 5), (5, 4), (6, 5), (5, 4)])
    return trntensor.multi_einsum(("ij,jk->ik", a, b), ("ij,jk->ik", c, d))


def _sharded(dt, g):
    a, b = _mk((8, 5), dt, g), _mk((5, 4), dt, g)
    return trntensor.einsum("ij,jk->ik", trntensor.scatter(a, 0, 2), b)


def _mp2(dt, g):
    B = _mk((3, 5, 7), dt, g)
    eo = _mk((3,), dt, g)
    ev = _mk((5,), dt, g)
    if dt.is_floating_point:
        eo = (-(eo.float().abs()) - 1).to(dt)
        ev = (ev.float().abs() + 1).to(dt)
    return trntensor.mp2_energy(B, eo, ev)


def _ao2mo(packed):
    def run(dt, g):
        eri = _sym(_mk((6, 6, 5), dt, g))
        co, cv = _mk((6, 2), dt, g), _mk((6, 3), dt, g)
        return trntensor.ao_to_mo_transform(trntensor.pack_tril(eri) if packed else eri, co, cv)

    return run


def _dfk(dt, g):
    eri = _sym(_mk((5, 5, 4), dt, g))
    return trntensor.df_exchange(trntensor.pack_tril(eri), _mk((5, 2), dt, g), alpha=2.0)


def _dfj(dt, g):
    eri = _sym(_mk((23, 23, 3), dt, g))  # 276 packed rows: the kernel route needs >= 256
    D = _mk((23, 23), dt, g)
    return trntensor.df_coulomb(trntensor.pack_tril(eri), D + D.T)


def _cp(dt, g):
    return trntensor.cp_decompose(_mk((4, 5, 6), dt, g), rank=2, max_iter=2)


def _tucker(dt, g):
    return trntensor.tucker_decompose(_mk((4, 5, 6), dt, g), ranks=(2, 2, 2))


def _tt(dt, g):
    return trntensor.tt_decompose(_mk((4, 5, 6), dt, g), max_rank=2)


CALLS = [
    ("einsum-matmul", {}, _einsum("ij,jk->ik", [(6, 5), (5, 4)])),
    ("einsum-matmul-T", {}, _einsum("ji,jk->ik", [(5, 6), (5, 4)])),
    ("einsum-sr", {}, _einsum("ij,jk->ik", [(6, 5), (5, 4)], precision="sr")),
    ("einsum-kahan", {}, _einsum("ij,jk->ik", [(6, 5), (5, 4)], precision="kahan")),
    ("einsum-dd", {}, _einsum("ij,jk->ik", [(6, 5), (5, 4)], precision="dd")),
    ("einsum-bmm", {}, _einsum("bij,bjk->bik", [(2, 6, 5), (2, 5, 4)])),
    ("einsum-path", {}, _einsum("ij,jk,kl->il", [(6, 5), (5, 4), (4, 3)])),
    ("einsum-target", {}, _einsum("ij,jk->ik", [(6, 5), (5, 4)], target_forward_error=1e-2)),
    ("multi_einsum", {}, _multi),
    ("einsum-sharded", {}, _sharded),
    ("mp2-rows", {"TRNTENSOR_MP2_ENERGY": "rows"}, _mp2),
    ("mp2-legacy", {"TRNTENSOR_MP2_ENERGY": "legacy"}, _mp2),
    ("ao2mo-trn2", {"TRNTENSOR_AO_TO_MO": "trn2"}, _ao2mo(False)),
    ("ao2mo-trn2-packed", {"TRNTENSOR_AO_TO_MO": "trn2"}, _ao2mo(True)),
    ("ao2mo-legacy", {"TRNTENSOR_AO_TO_MO": "legacy"}, _ao2mo(False)),
    ("df_exchange", {"TRNTENSOR_DF_EXCHANGE": "trn2"}, _dfk),
    ("df_coulomb", {"TRNTENSOR_DF_COULOMB": "trn2"}, _dfj),
    ("cp_decompose", {}, _cp),
    ("tucker_decompose", {}, _tucker),
    ("tt_decompose", {}, _tt),
]


def _leaves(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _leaves(v)]
    if isinstance(x, parallel.ShardedTensor):
        return _leaves(list(x.chunks))
    return []


def _run(fn, dt, seed=0):
    try:
        return fn(dt, torch.Generator().manual_seed(seed)), None
    except Exception as e:  # noqa: BLE001 - compared against the pytorch backend
        return None, e


def _reference(monkeypatch, fn, dt):
    _set_backend(monkeypatch, "pytorch")
    try:
        return _run(fn, dt)
    finally:
        trntensor.clear_plan_cache()


@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name,env,fn", CALLS, ids=[c[0] for c in CALLS])
def test_no_unsupported_dtype_reaches_a_kernel(guarded, monkeypatch, name, env, fn, dtype, backend):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ref, ref_err = _reference(monkeypatch, fn, dtype)
    _set_backend(monkeypatch, backend)
    out, err = _run(fn, dtype)
    assert guarded.violations == [], guarded.violations
    if ref_err is not None:
        # The pytorch backend rejects these operands too (e.g. complex SVD,
        # integer decompositions, precision="dd" with NKI installed).
        return
    assert err is None, f"{name} {dtype} under {backend!r}: {err!r} (pytorch backend: ok)"
    got, want = _leaves(out), _leaves(ref)
    assert [t.dtype for t in got] == [t.dtype for t in want]
    assert [t.shape for t in got] == [t.shape for t in want]


def test_every_native_op_is_driven_and_listed(guarded, monkeypatch):
    """fp32 through every call reaches every registered op, and every op has an
    allowed-dtype entry here: a new kernel without one fails this test."""
    _require_nki(monkeypatch)  # a kernel failure must not hide behind aten
    _set_backend(monkeypatch, "nki")
    for _name, env, fn in CALLS:
        if _name == "einsum-dd":
            continue  # raises NotImplementedError with NKI installed (trnblas#22)
        with monkeypatch.context() as m:
            for k, v in env.items():
                m.setenv(k, v)
            fn(F32, torch.Generator().manual_seed(0))
    from trntensor.nki import _native

    assert set(_native.OPS) == set(KERNEL_TAKES)
    assert set(guarded.calls) == set(KERNEL_TAKES)
    assert guarded.violations == []


# ── TRNTENSOR_REQUIRE_NKI=1 ──────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.bool, torch.complex64])
@pytest.mark.parametrize("name", ["einsum-matmul", "einsum-bmm", "einsum-sr"])
def test_require_nki_refuses_matmul_dtypes(guarded, monkeypatch, dtype, name):
    """The matmul kernels take fp32/bf16/fp16; REQUIRE_NKI raises rather than
    silently computing in torch."""
    _require_nki(monkeypatch)
    _set_backend(monkeypatch, "nki")
    fn = dict((c[0], c[2]) for c in CALLS)[name]
    with pytest.raises(NotImplementedError, match="TRNTENSOR_REQUIRE_NKI"):
        fn(dtype, torch.Generator().manual_seed(0))
    assert guarded.violations == []


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool, torch.complex64])
@pytest.mark.parametrize(
    "name",
    ["mp2-rows", "mp2-legacy", "ao2mo-trn2", "ao2mo-trn2-packed", "df_exchange", "df_coulomb"],
)
def test_require_nki_refuses_chemistry_dtypes(guarded, monkeypatch, dtype, name):
    _require_nki(monkeypatch)
    _set_backend(monkeypatch, "nki")
    _, env, fn = next(c for c in CALLS if c[0] == name)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(NotImplementedError, match="TRNTENSOR_REQUIRE_NKI"):
        fn(dtype, torch.Generator().manual_seed(0))
    assert guarded.violations == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "name",
    ["mp2-rows", "mp2-legacy", "ao2mo-trn2", "ao2mo-trn2-packed", "df_exchange", "df_coulomb"],
)
def test_require_nki_chemistry_floats_run_the_kernel(guarded, monkeypatch, dtype, name):
    """float16 (and, on the fp32-only df_* kernels, bfloat16) chemistry operands
    run the kernel in fp32 (a lossless upcast) and come back in the caller's
    dtype; REQUIRE_NKI does not refuse them. float64 is refused (#92), below."""
    _require_nki(monkeypatch)
    _set_backend(monkeypatch, "nki")
    _, env, fn = next(c for c in CALLS if c[0] == name)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = fn(dtype, torch.Generator().manual_seed(0))
    assert out.dtype == dtype
    assert guarded.calls, "the kernel route did not run"
    assert guarded.violations == []


# ── The float64 policy (#92): auto → torch, nki → fp32 kernel + warning,
#    REQUIRE_NKI → NotImplementedError ──────────────────────────────────────

# Calls that reach a kernel with fp32 operands (kahan / dd never do, and the
# decompositions use torch matmuls directly).
_NO_KERNEL = ("einsum-kahan", "einsum-dd", "cp_decompose", "tucker_decompose", "tt_decompose")
KERNEL_CALLS = [c for c in CALLS if c[0] not in _NO_KERNEL]
_KERNEL_IDS = [c[0] for c in KERNEL_CALLS]
_F64 = torch.float64


def _with_env(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def _fp64_warnings(record) -> list[str]:
    """Op names of the fp64 reduced-precision UserWarnings in ``record``."""
    ops = []
    for w in record:
        msg = str(w.message)
        if issubclass(w.category, UserWarning) and "has fp32 accuracy" in msg:
            ops.append(msg.split("trntensor ", 1)[1].split(":", 1)[0])
    return ops


@pytest.mark.parametrize("name,env,fn", KERNEL_CALLS, ids=_KERNEL_IDS)
def test_fp64_auto_runs_torch_at_fp64(guarded, monkeypatch, name, env, fn):
    """``"auto"``: float64 reaches no kernel, does not warn, and gives the
    pytorch backend's float64 result (the stubs would return zeros)."""
    _with_env(monkeypatch, env)
    ref, ref_err = _reference(monkeypatch, fn, _F64)
    assert ref_err is None, ref_err
    _set_backend(monkeypatch, "auto")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out, err = _run(fn, _F64)
    assert err is None, err
    assert guarded.calls == [], guarded.calls
    assert _fp64_warnings(record) == []
    for got, want in zip(_leaves(out), _leaves(ref), strict=True):
        assert got.dtype == want.dtype == _F64
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name,env,fn", KERNEL_CALLS, ids=_KERNEL_IDS)
def test_fp64_explicit_nki_runs_the_fp32_kernel_and_warns_once(guarded, monkeypatch, name, env, fn):
    """Explicit ``"nki"``: the kernel runs on fp32 operands, the result is
    float64, and each op warns exactly once across repeated calls."""
    _with_env(monkeypatch, env)
    _set_backend(monkeypatch, "nki")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for seed in range(3):
            out, err = _run(fn, _F64, seed=seed)
            assert err is None, err
            assert all(t.dtype == _F64 for t in _leaves(out) if t.dtype.is_floating_point)
    assert guarded.calls, "the kernel route did not run"
    assert guarded.violations == []
    assert guarded.float_dtypes == {F32}, guarded.float_dtypes
    ops = _fp64_warnings(record)
    assert ops, "no reduced-precision warning"
    assert sorted(ops) == sorted(set(ops)), f"warned more than once per op: {ops}"


def test_fp64_warning_names_the_op_and_the_way_out(guarded, monkeypatch):
    _set_backend(monkeypatch, "nki")
    a = torch.randn(6, 5, dtype=_F64)
    b = torch.randn(5, 4, dtype=_F64)
    with pytest.warns(UserWarning, match=r'trntensor matmul: .*set_backend\("auto"\)'):
        trntensor.einsum("ij,jk->ik", a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second call must not warn again
        trntensor.einsum("ij,jk->ik", a, b)


@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize("name,env,fn", KERNEL_CALLS, ids=_KERNEL_IDS)
def test_fp64_require_nki_raises(guarded, monkeypatch, name, env, fn, backend):
    """``TRNTENSOR_REQUIRE_NKI=1``: float64 on a kernel route raises, before
    any cast or kernel launch, under explicit ``"nki"`` and ``"auto"``."""
    _with_env(monkeypatch, env)
    _require_nki(monkeypatch)
    _set_backend(monkeypatch, backend)
    with pytest.raises(NotImplementedError, match="float64|TRNTENSOR_REQUIRE_NKI"):
        fn(_F64, torch.Generator().manual_seed(0))
    assert guarded.calls == []


# Mixed-dtype operands of the chemistry calls: (first operand dtype, the rest).
def _mp2_mixed(a, b):
    def run(_dt, g):
        B = _mk((3, 5, 7), a, g)
        eo = (-(torch.rand(3, generator=g)) - 1).to(b)
        ev = (torch.rand(5, generator=g) + 1).to(b)
        return trntensor.mp2_energy(B, eo, ev)

    return run


def _ao2mo_mixed(a, b, packed=False):
    def run(_dt, g):
        eri = _sym(_mk((6, 6, 5), a, g))
        co, cv = _mk((6, 2), b, g), _mk((6, 3), b, g)
        return trntensor.ao_to_mo_transform(trntensor.pack_tril(eri) if packed else eri, co, cv)

    return run


def _dfk_mixed(a, b):
    def run(_dt, g):
        eri = _sym(_mk((5, 5, 4), a, g))
        return trntensor.df_exchange(trntensor.pack_tril(eri), _mk((5, 2), b, g))

    return run


def _dfj_mixed(a, b):
    def run(_dt, g):
        eri = _sym(_mk((23, 23, 3), a, g))
        D = _mk((23, 23), b, g)
        return trntensor.df_coulomb(trntensor.pack_tril(eri), D + D.T)

    return run


def _matmul_mixed(a, b):
    def run(_dt, g):
        return trntensor.einsum("ij,jk->ik", _mk((6, 5), a, g), _mk((5, 4), b, g))

    return run


_MIXED_PAIRS = [(F32, _F64), (_F64, F32), (BF16, F32), (F16, _F64), (BF16, F16), (F16, F32)]
_MIXED_CALLS = [
    ("mp2-rows", {"TRNTENSOR_MP2_ENERGY": "rows"}, _mp2_mixed),
    ("ao2mo-trn2", {"TRNTENSOR_AO_TO_MO": "trn2"}, _ao2mo_mixed),
    ("ao2mo-trn2-packed", {"TRNTENSOR_AO_TO_MO": "trn2"}, lambda a, b: _ao2mo_mixed(a, b, True)),
    ("df_exchange", {"TRNTENSOR_DF_EXCHANGE": "trn2"}, _dfk_mixed),
    ("df_coulomb", {"TRNTENSOR_DF_COULOMB": "trn2"}, _dfj_mixed),
    ("einsum-matmul", {}, _matmul_mixed),
]


@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize("pair", _MIXED_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("name,env,make", _MIXED_CALLS, ids=[c[0] for c in _MIXED_CALLS])
def test_mixed_dtype_result_matches_pytorch_backend(
    guarded, monkeypatch, name, env, make, pair, backend
):
    """The result dtype is the pytorch backend's (its promotion of the operand
    dtypes), and no kernel sees a dtype it does not take."""
    _with_env(monkeypatch, env)
    fn = make(*pair)
    ref, ref_err = _reference(monkeypatch, fn, None)
    _set_backend(monkeypatch, backend)
    out, err = _run(fn, None)
    assert guarded.violations == [], guarded.violations
    if ref_err is not None:
        # The pytorch backend rejects the mix (aten einsum needs one dtype);
        # the kernel routes cast per operand, so there is no dtype to compare.
        return
    assert err is None, err
    want = [t.dtype for t in _leaves(ref)]
    assert [t.dtype for t in _leaves(out)] == want
    if backend == "auto" and _F64 in want:
        assert guarded.calls == []  # a float64 result under "auto": torch


# ── Reduce-parallel all-reduce (nki.collectives) ─────────────────────────────


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_reduce_partials_sends_only_kernel_dtypes_to_the_collective(monkeypatch, dtype):
    import torch.distributed as dist

    from trntensor.nki import collectives

    bad = []

    def fake_collective(t):
        if t.dtype not in COLLECTIVE_TAKES:
            bad.append(t.dtype)
        return t.clone()

    monkeypatch.setattr(collectives, "nki_all_reduce_sum", fake_collective)
    monkeypatch.setattr(parallel, "allreduce_backend", lambda n: "nki")
    monkeypatch.setattr(parallel, "_dist_allreduce", lambda t: t.contiguous().clone())
    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: 0)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    monkeypatch.delenv("TRNTENSOR_REQUIRE_NKI", raising=False)
    part = _mk((3, 4), dtype, torch.Generator().manual_seed(0))
    out = parallel.reduce_partials(lambda i: part, 2)
    assert bad == []
    assert out.dtype == dtype and torch.equal(out, part)


def test_nki_all_reduce_sum_refuses_float64(monkeypatch):
    import torch.distributed as dist

    from trntensor.nki.collectives import nki_all_reduce_sum

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    with pytest.raises(TypeError, match="all_reduce"):
        nki_all_reduce_sum(torch.zeros(2, 2, dtype=torch.float64))


# ── The launch boundary itself ───────────────────────────────────────────────


def test_launch_refuses_unlisted_dtypes(guarded):
    """`_launch` enforces the table even if a wrapper forgets the guard."""
    a = torch.zeros(128, 128, dtype=torch.float64)
    with pytest.raises(TypeError, match="'matmul'"):
        dispatch._launch(None, "matmul", a, a)
    with pytest.raises(TypeError, match="'matmul'"):
        dispatch._launch(None, "matmul", a.float(), a.to(torch.bfloat16))
    with pytest.raises(TypeError, match="no KERNEL_OPERAND_DTYPES entry"):
        dispatch._launch(None, "not_a_kernel", a.float())
    assert guarded.calls == []


def test_library_table_matches(guarded):
    assert {k: tuple(set(s) for s in v) for k, v in dispatch.KERNEL_OPERAND_DTYPES.items()} == {
        **KERNEL_TAKES,
        "all_reduce": (COLLECTIVE_TAKES,),
    }


# ── float64 already on the neuron device (#92): "auto" computes on the host ──
#
# In `guarded` CPU stands in for the neuron device. Here every tensor counts as
# neuron-resident except the host copies the guard makes (`_fp64_to_host`,
# tagged `_fake_host`); the return transfer (`_fp64_to_device`) records each
# tensor and its target device.


class _Transfers:
    def __init__(self):
        self.to_host: list[torch.dtype] = []
        self.to_device: list[tuple[int, torch.device]] = []
        self.aten_on_device = 0

    def back_ids(self):
        return {i for i, _ in self.to_device}


@pytest.fixture
def on_device(guarded, monkeypatch):
    tr = _Transfers()

    def to_host(t):
        tr.to_host.append(t.dtype)
        h = t.clone()  # differentiable, like the real .to("cpu")
        h._fake_host = True
        return h

    def to_device(t, device):
        tr.to_device.append((id(t), torch.device(device)))
        return t

    real_aten = dispatch._aten_on_device

    def aten_spy(fn, *ts):
        tr.aten_on_device += 1
        return real_aten(fn, *ts)

    monkeypatch.setattr(
        dispatch, "_on_neuron_device", lambda t: not getattr(t, "_fake_host", False)
    )
    monkeypatch.setattr(dispatch, "_fp64_to_host", to_host)
    monkeypatch.setattr(dispatch, "_fp64_to_device", to_device)
    monkeypatch.setattr(dispatch, "_aten_on_device", aten_spy)
    tr.rec = guarded
    return tr


def _host_warnings(record) -> list[str]:
    """Op names of the device-resident float64 host-path UserWarnings."""
    ops = []
    for w in record:
        msg = str(w.message)
        if issubclass(w.category, UserWarning) and "computed on the host" in msg:
            ops.append(msg.split("trntensor ", 1)[1].split(":", 1)[0])
    return ops


def _seeded(fn, dt, seed=0):
    torch.manual_seed(seed)  # the decompositions draw their initial factors
    return _run(fn, dt, seed)


def _skip_if_pytorch_rejects(monkeypatch, fn, dt):
    _, err = _reference(monkeypatch, lambda d, g: _seeded_inner(fn, d, g), dt)
    if err is not None:
        pytest.skip(f"the pytorch backend rejects these operands too: {err!r}")


_HOST_CALLS = [c for c in CALLS if c[0] != "einsum-dd"]  # dd raises with NKI (trnblas#22)
_HOST_IDS = [c[0] for c in _HOST_CALLS]
_C128 = torch.complex128


@pytest.mark.parametrize("dtype", [_F64, _C128], ids=str)
@pytest.mark.parametrize("name,env,fn", _HOST_CALLS, ids=_HOST_IDS)
def test_device_fp64_auto_computes_on_host(on_device, monkeypatch, name, env, fn, dtype):
    """``"auto"``, float64 / complex128 on the neuron device: the operands go to
    the host, no kernel or device aten runs, the results come back to the
    operands' device in float64 / complex128 and match the fp64 CPU reference
    (the pytorch backend on host inputs) to fp64 tolerance."""
    _with_env(monkeypatch, env)
    ref, ref_err = _reference(monkeypatch, lambda dt, g: _seeded_inner(fn, dt, g), dtype)
    if ref_err is not None:
        pytest.skip(f"the pytorch backend rejects these operands too: {ref_err!r}")
    _set_backend(monkeypatch, "auto")
    out, err = _seeded(fn, dtype)
    assert err is None, err
    assert on_device.to_host, "the operands were not copied to the host"
    assert set(on_device.to_host) <= {_F64, _C128} | set(on_device.to_host)
    assert on_device.rec.calls == [] and on_device.aten_on_device == 0
    leaves = _leaves(out)
    if name != "multi_einsum":  # multi_einsum returns host tensors by contract
        back = on_device.back_ids()
        assert all(id(t) in back for t in leaves), "a result did not go back to the device"
        assert {d for _, d in on_device.to_device} == {torch.device("cpu")}  # the operands' device
    for got, want in zip(leaves, _leaves(ref), strict=True):
        assert got.dtype == want.dtype
        if got.dtype in (_F64, _C128):
            torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def _seeded_inner(fn, dt, g):
    torch.manual_seed(0)
    return fn(dt, g)


@pytest.mark.parametrize("name,env,fn", _HOST_CALLS, ids=_HOST_IDS)
def test_device_fp64_host_warning_fires_once_per_op(on_device, monkeypatch, name, env, fn):
    _skip_if_pytorch_rejects(monkeypatch, fn, _F64)
    _with_env(monkeypatch, env)
    _set_backend(monkeypatch, "auto")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for seed in range(3):
            _, err = _seeded(fn, _F64, seed)
            assert err is None, err
    ops = _host_warnings(record)
    assert len(ops) == 1, ops
    assert _fp64_warnings(record) == []  # not the explicit-nki fp32 warning


def test_device_fp64_host_warning_text(on_device):
    a, b = torch.randn(6, 5, dtype=_F64), torch.randn(5, 4, dtype=_F64)
    with pytest.warns(UserWarning) as record:
        trntensor.einsum("ij,jk->ik", a, b)
    msg = str(record[0].message)
    assert msg.startswith("trntensor einsum: float64 on the neuron device is computed on the host")
    assert (
        "float64 on the neuron device is computed on the host for float64 accuracy "
        "(device arithmetic is fp32); pass float32 to stay on device" in msg
    )
    assert "TRNTENSOR_FP64_ON_DEVICE=1" in msg


@pytest.mark.parametrize("how", ["env", "global"])
@pytest.mark.parametrize("name,env,fn", _HOST_CALLS, ids=_HOST_IDS)
def test_device_fp64_on_device_override_keeps_the_device_path(
    on_device, monkeypatch, name, env, fn, how
):
    """``TRNTENSOR_FP64_ON_DEVICE=1``: no host copy, no host-path warning; the
    call takes its device route (torch on the operands' device)."""
    _skip_if_pytorch_rejects(monkeypatch, fn, _F64)
    _with_env(monkeypatch, env)
    if how == "env":
        monkeypatch.setenv("TRNTENSOR_FP64_ON_DEVICE", "1")
    else:
        monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", True)
    _set_backend(monkeypatch, "auto")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out, err = _seeded(fn, _F64)
    assert err is None, err
    assert on_device.to_host == [] and on_device.to_device == []
    assert _host_warnings(record) == []
    assert all(t.dtype == _F64 for t in _leaves(out) if t.dtype.is_floating_point)


@pytest.mark.parametrize("name,env,fn", KERNEL_CALLS, ids=_KERNEL_IDS)
def test_device_fp64_require_nki_still_raises(on_device, monkeypatch, name, env, fn):
    _with_env(monkeypatch, env)
    _require_nki(monkeypatch)
    _set_backend(monkeypatch, "auto")
    with pytest.raises(NotImplementedError, match="float64|TRNTENSOR_REQUIRE_NKI"):
        fn(_F64, torch.Generator().manual_seed(0))
    assert on_device.to_host == []


@pytest.mark.parametrize("backend", ["nki", "pytorch"])
@pytest.mark.parametrize("name,env,fn", KERNEL_CALLS, ids=_KERNEL_IDS)
def test_device_fp64_explicit_backends_unchanged(on_device, monkeypatch, name, env, fn, backend):
    """Explicit ``"nki"`` (fp32 kernel + its warning) and ``"pytorch"`` (torch on
    the operands' device) do not take the host path."""
    _with_env(monkeypatch, env)
    _set_backend(monkeypatch, backend)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, err = _seeded(fn, _F64)
    assert err is None, err
    assert on_device.to_host == [] and _host_warnings(record) == []
    if backend == "nki":
        assert on_device.rec.calls, "the kernel route did not run"


@pytest.mark.parametrize("name,env,fn", _HOST_CALLS, ids=_HOST_IDS)
def test_host_fp64_auto_unchanged(guarded, monkeypatch, name, env, fn):
    """Host-resident float64 under ``"auto"``: host torch, no host-path warning,
    no upload (the default predicate: CPU is not the ``neuron`` device)."""
    _skip_if_pytorch_rejects(monkeypatch, fn, _F64)
    _with_env(monkeypatch, env)
    copies = []
    monkeypatch.setattr(dispatch, "_fp64_to_host", lambda t: copies.append(t) or t)
    _set_backend(monkeypatch, "auto")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, err = _seeded(fn, _F64)
    assert err is None, err
    assert copies == [] and _host_warnings(record) == []


def test_device_fp64_einsum_out_is_written_on_device(on_device):
    """``einsum(out=)`` on device: the result is written into the caller's
    device tensor, which is returned (not its host copy)."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(6, 5, generator=g, dtype=_F64)
    b = torch.randn(5, 4, generator=g, dtype=_F64)
    out = torch.randn(6, 4, generator=g, dtype=_F64)
    want = 2.0 * (a @ b) + 0.5 * out
    got = trntensor.einsum("ij,jk->ik", a, b, alpha=2.0, beta=0.5, out=out)
    assert got is out
    torch.testing.assert_close(out, want, rtol=1e-12, atol=1e-12)
    assert on_device.to_device == []  # the caller's tensor was written, not moved


def test_device_fp64_explicit_float32_dtype_stays_on_device(on_device):
    """``einsum(dtype="f32")`` asks for fp32 compute: no host path."""
    a, b = torch.randn(6, 5, dtype=_F64), torch.randn(5, 4, dtype=_F64)
    out = trntensor.einsum("ij,jk->ik", a, b, dtype="f32")
    assert out.dtype == F32 and on_device.to_host == []


@pytest.mark.parametrize("dt", [torch.float64, "f64", "float64"])
def test_device_explicit_float64_dtype_goes_to_host(on_device, dt):
    """``einsum(dtype=float64)`` on float32 device operands asks for float64
    compute, which the device would do at fp32 precision: host path."""
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(6, 5, generator=g), torch.randn(5, 4, generator=g)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = trntensor.einsum("ij,jk->ik", a, b, dtype=dt)
    assert on_device.to_host == [F32, F32] and _host_warnings(record) == ["einsum"]
    assert out.dtype == _F64
    torch.testing.assert_close(out, a.double() @ b.double(), rtol=1e-12, atol=1e-12)


def test_device_fp64_gather_concatenates_on_host(on_device):
    """``gather`` of float64 device chunks: device ``torch.cat`` rounds float64
    to fp32, so the chunks are concatenated on the host."""
    g = torch.Generator().manual_seed(2)
    z = torch.randn(7, 5, generator=g, dtype=_F64)
    st = trntensor.scatter(z, 0, 3)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = trntensor.gather(st)
    assert on_device.to_host == [_F64] * 3 and _host_warnings(record) == ["gather"]
    assert torch.equal(out, z)


def test_device_fp64_mixed_operands_go_to_host(on_device):
    """One float64 operand on the device is enough: the float32 operand goes
    with it, and the result is the pytorch backend's float64."""
    g = torch.Generator().manual_seed(0)
    B = torch.randn(3, 5, 7, generator=g, dtype=F32)
    eo = -torch.rand(3, generator=g, dtype=_F64) - 1
    ev = torch.rand(5, generator=g, dtype=_F64) + 1
    out = trntensor.mp2_energy(B, eo, ev)
    assert sorted(map(str, on_device.to_host)) == sorted(map(str, [F32, _F64, _F64]))
    want = trntensor.quantum._cpu_mp2_energy(B, eo, ev)
    assert out.dtype == want.dtype == _F64
    torch.testing.assert_close(out, want, rtol=1e-12, atol=1e-12)


def test_device_fp64_autograd_flows_back_to_the_device_operands(on_device):
    a = torch.randn(6, 5, dtype=_F64, requires_grad=True)
    b = torch.randn(5, 4, dtype=_F64)
    trntensor.einsum("ij,jk->ik", a, b).sum().backward()
    torch.testing.assert_close(a.grad, torch.ones(6, 4, dtype=_F64) @ b.T)


def test_multi_einsum_does_not_pin_host_fp64(guarded, monkeypatch):
    """Host float64 shared operands are not pinned to the device under
    ``"auto"`` (device aten would compute them at fp32 precision)."""
    pinned = []
    monkeypatch.setattr(dispatch, "_to_device", lambda t: pinned.append(t) or t)
    _set_backend(monkeypatch, "auto")
    a, b = torch.randn(6, 5, dtype=_F64), torch.randn(5, 4, dtype=_F64)
    trntensor.multi_einsum(("ij,jk->ik", a, b), ("ij,jk->ik", a, b))
    assert pinned == []
    af, bf = a.float(), b.float()
    trntensor.multi_einsum(("ij,jk->ik", af, bf), ("ij,jk->ik", af, bf), ("ij,jk->ik", a, b))
    assert pinned == []  # one float64 contraction keeps the batch unpinned
    trntensor.multi_einsum(("ij,jk->ik", af, bf), ("ij,jk->ik", af, bf))
    assert len(pinned) == 2  # float32 still pins


# ── trntensor.stream: routes that place their own compute (#97) ──────────────
# `AuxStream` and `mp2_energy_blocked` take a `devices=` list and upload the
# operands there themselves, so the operand guard cannot see a neuron-bound
# float64 computation. Here "meta" stands for the neuron device: if the policy
# did not move the compute to the host, it would run on meta and give no values.


@pytest.fixture
def meta_is_neuron(guarded, monkeypatch):
    monkeypatch.setattr(dispatch, "_device_is_neuron", lambda d: torch.device(d).type == "meta")
    _set_backend(monkeypatch, "auto")
    return guarded


def _stream_case(dt):
    from trntensor import stream

    g = torch.Generator().manual_seed(3)
    nbasis, naux = 6, 10
    npair = nbasis * (nbasis + 1) // 2
    Bp = torch.randn(npair, naux, generator=g, dtype=torch.float64).to(dt)
    D = torch.randn(nbasis, nbasis, generator=g, dtype=torch.float64)
    pw = stream.pack_density(D + D.T).to(dt)
    B = torch.randn(3, 4, naux, generator=g, dtype=torch.float64).to(dt)
    eo = -1.0 - torch.rand(3, generator=g, dtype=torch.float64).to(dt)
    ev = 1.0 + torch.rand(4, generator=g, dtype=torch.float64).to(dt)
    return stream, nbasis, Bp, pw, B, eo, ev


def test_stream_fp64_bound_for_the_device_computes_on_host(meta_is_neuron):
    stream, nbasis, Bp, pw, B, eo, ev = _stream_case(_F64)
    want_j = stream.AuxStream([Bp], nbasis, dtype=_F64).coulomb(pw)
    want_e = stream.mp2_energy_blocked(B, eo, ev, occ_block=1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        aux = stream.AuxStream([Bp[:, :4], Bp[:, 4:]], nbasis, devices=["meta"], dtype=_F64)
        got_j = aux.coulomb(pw, out_device="cpu")
        stream.AuxStream([Bp], nbasis, devices=["meta"], dtype=_F64)
        got_e = stream.mp2_energy_blocked(B, eo, ev, devices=["meta"], occ_block=1)
    assert aux.devices == [torch.device("cpu")] and aux.device == torch.device("meta")
    assert sorted(_host_warnings(record)) == ["AuxStream", "mp2_energy_blocked"]
    assert meta_is_neuron.calls == []
    torch.testing.assert_close(got_j, want_j, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(got_e, want_e, rtol=1e-12, atol=1e-12)


def test_stream_fp64_device_path_kept_off_the_host_policy(meta_is_neuron, monkeypatch):
    """float32, TRNTENSOR_FP64_ON_DEVICE=1, explicit "nki" / "pytorch" and a
    host-only device list keep the devices the caller gave."""
    stream, nbasis, Bp, *_ = _stream_case(_F64)

    def devices(dt=_F64, devs=("meta",)):
        return stream.AuxStream([Bp.to(dt)], nbasis, devices=list(devs), dtype=dt).devices

    meta = [torch.device("meta")]
    assert devices(F32) == meta
    assert devices(_F64, ("cpu",)) == [torch.device("cpu")]
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", True)
    assert devices() == meta
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", False)
    for backend in ("nki", "pytorch"):
        _set_backend(monkeypatch, backend)
        assert devices() == meta


@pytest.mark.parametrize("override", [False, True])
def test_device_fp64_blocked_tril_gather_concatenates_on_host(guarded, monkeypatch, override):
    """`pack_tril` / `unpack_tril` split device gathers into blocks (#97) and
    join them with `torch.cat`, which rounds float64 to fp32 on the device (see
    `gather`): under "auto" float64 blocks are joined on the host, one warning
    per op; TRNTENSOR_FP64_ON_DEVICE=1 keeps the device `torch.cat`."""
    import trntensor.quantum as q

    monkeypatch.setattr(q, "_gather_in_blocks", lambda t: True)
    monkeypatch.setattr(q, "_GATHER_MAX_BYTES", 256)
    monkeypatch.setattr(dispatch, "_device_is_neuron", lambda d: True)
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", override)
    _set_backend(monkeypatch, "auto")
    e = torch.randn(9, 9, 5, dtype=_F64)
    e = e + e.transpose(0, 1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(2):
            full = trntensor.unpack_tril(trntensor.pack_tril(e), 9)
        trntensor.unpack_tril(trntensor.pack_tril(e.float()), 9)  # fp32: device cat
    assert torch.equal(full, e)
    assert sorted(_host_warnings(record)) == ([] if override else ["pack_tril", "unpack_tril"])
