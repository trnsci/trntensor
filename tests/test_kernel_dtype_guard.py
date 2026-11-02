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


def _fake_torch_neuronx(rec: _Recorder):
    mod = types.ModuleType("torch_neuronx")

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        short = name.split("::", 1)[1]

        def dec(_wrapper):
            def stub(*args, **kwargs):
                rec.calls.append(short)
                ts = [a for a in args if isinstance(a, torch.Tensor)]
                takes = KERNEL_TAKES.get(short)
                dts = [t.dtype for t in ts]
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
    for var in ("TRNTENSOR_FORCE_BACKEND", "TRNTENSOR_REQUIRE_NKI", "TRNSCI_NKI_DISPATCH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
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
    if hasattr(x, "shards"):  # ShardedTensor
        return _leaves(list(x.shards))
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


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "name",
    ["mp2-rows", "mp2-legacy", "ao2mo-trn2", "ao2mo-trn2-packed", "df_exchange", "df_coulomb"],
)
def test_require_nki_chemistry_floats_run_the_kernel(guarded, monkeypatch, dtype, name):
    """float64 / float16 chemistry operands run the kernel in fp32 (#88) and come
    back in the caller's dtype; REQUIRE_NKI does not refuse them."""
    _require_nki(monkeypatch)
    _set_backend(monkeypatch, "nki")
    _, env, fn = next(c for c in CALLS if c[0] == name)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = fn(dtype, torch.Generator().manual_seed(0))
    assert out.dtype == dtype
    assert guarded.calls, "the kernel route did not run"
    assert guarded.violations == []


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
