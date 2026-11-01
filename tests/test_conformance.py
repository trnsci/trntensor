"""Reference-conformance suite (trnsci/trnsci#105).

Every public trntensor API is checked against an *independent* reference:
``torch.einsum`` / ``numpy.einsum`` in float64 on the host, dense formulas,
reconstruction identities and closed-form error bounds. Differential tests
(nki vs pytorch) cannot see a bug that every backend shares; three audit
rounds (trnsci/trnsci#35) found that class repeatedly, e.g. the einsum
index-order bug (#64), which returned ``A @ B`` for ``ij,jk->ki``.

Axes: all index orderings of matmul/bmm-shaped two-operand contractions,
generated 1-3 operand subscripts, awkward sizes (1, 127, 129, 1000), dtypes
(bf16, fp16, fp32, fp64, int, complex), every ``precision=`` mode and
``target_forward_error``, ``alpha``/``beta``/``out=``, non-contiguous and
stride-0 operands, empty and size-1 shapes, fp64 ``gradcheck``, plan
metadata, decompositions, chemistry helpers and sharding.

Known behaviours waiting on a decision are ``xfail(strict=True)`` with their
issue number, so the test flips when the issue is resolved. Bugs whose fix
PR is open are ``xfail(strict=False)``, so merging the fix does not break CI;
remove the marker in (or after) the fix PR.

CPU (CI): ``pytest tests/test_conformance.py`` (pytorch backend, host inputs).

Device (Trn2), the same cases against the same references::

    pytest tests/test_conformance.py --backend nki --conformance-device neuron
    pytest tests/test_conformance.py --backend auto --conformance-device neuron
    pytest tests/test_conformance.py --backend nki      # host inputs

Off the pytorch backend or the host, float tolerances widen to the fp32-ish
device level and the fp64 gradchecks are skipped.
"""

from __future__ import annotations

import itertools
import random
import warnings

import numpy as np
import pytest
import torch

import trntensor as tt

BACKEND = "pytorch"
DEV = torch.device("cpu")


@pytest.fixture(scope="module", autouse=True)
def _conformance_backend(pytestconfig):
    global BACKEND, DEV
    BACKEND = pytestconfig.getoption("--backend")
    DEV = torch.device(pytestconfig.getoption("--conformance-device"))
    prev = tt.get_backend()
    tt.set_backend(BACKEND)
    tt.clear_plan_cache()
    yield
    tt.set_backend(prev)
    tt.clear_plan_cache()


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _strict():
    return BACKEND == "pytorch" and DEV.type == "cpu"


def d(t):
    """Move an input to the conformance device (no-op on CPU)."""
    if isinstance(t, torch.Tensor) and DEV.type != "cpu":
        return t.to(DEV)
    return t


def host(t):
    return t.detach().to("cpu") if isinstance(t, torch.Tensor) else t


def tol_for(dtype):
    if dtype == torch.bfloat16:
        return 6e-2
    if dtype == torch.float16:
        return 1e-2
    if dtype in (torch.float32, torch.complex64):
        return 2e-5 if _strict() else 2e-3
    if dtype in (torch.float64, torch.complex128):
        return 1e-11 if _strict() else 2e-6
    return 0.0  # integers: exact


def rel_err(got, ref):
    g, r = host(got), host(ref)
    if g.numel() == 0:
        return 0.0
    cdt = torch.complex128 if (g.is_complex() or r.is_complex()) else torch.float64
    g, r = g.to(cdt), r.to(cdt)
    if bool((torch.isnan(g) != torch.isnan(r)).any()):
        return float("inf")
    den = float(torch.linalg.norm(r.reshape(-1)))
    num = float(torch.linalg.norm((g - r).reshape(-1)))
    return num / den if den > 0 else num


def assert_ref(got, ref, *, tol=None, check_dtype=True):
    """``got`` matches the independent reference in shape, dtype and value."""
    gots = got if isinstance(got, (list, tuple)) else [got]
    refs = ref if isinstance(ref, (list, tuple)) else [ref]
    assert len(gots) == len(refs)
    for g, r in zip(gots, refs, strict=True):
        g, r = torch.as_tensor(host(g)), torch.as_tensor(host(r))
        assert tuple(g.shape) == tuple(r.shape), f"shape {tuple(g.shape)} != {tuple(r.shape)}"
        if check_dtype:
            assert g.dtype == r.dtype, f"dtype {g.dtype} != {r.dtype}"
        t = tol if tol is not None else tol_for(r.dtype)
        e = rel_err(g, r)
        assert e <= t, f"rel err {e:.3e} > {t:.1e}"


def same_bits(a, b):
    a, b = host(a), host(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point() or a.is_complex():
        return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
    return bool((a == b).all())


def ref_einsum(sub, *ops, dtype=None):
    """torch.einsum on the host in float64/complex128, cast to torch's dtype."""
    hops = [host(o) for o in ops]
    out_dtype = dtype or torch.einsum(sub, *hops).dtype
    if all(not (o.is_floating_point() or o.is_complex()) for o in hops):
        return torch.einsum(sub, *hops)
    hi = [
        o.to(torch.complex128)
        if o.is_complex()
        else o.to(torch.float64)
        if o.is_floating_point()
        else o
        for o in hops
    ]
    return torch.einsum(sub, *hi).to(out_dtype)


def rand(shape, dtype, gen):
    if dtype.is_complex:
        return torch.randn(shape, generator=gen, dtype=dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=gen).to(dtype)
    return torch.randint(-4, 5, shape, generator=gen, dtype=dtype)


def ops_for(sub, sizes, dtype, gen):
    return [rand([sizes[c] for c in t], dtype, gen) for t in sub.split("->")[0].split(",")]


def _skip_dd_on_trainium(mode):
    if mode == "dd" and tt.HAS_NKI:
        pytest.skip("precision='dd' raises NotImplementedError on Trainium hosts (trnblas#22)")


def xfail_open(issue, pr, why):
    return pytest.mark.xfail(strict=False, reason=f"trnsci/trntensor#{issue}, fix in #{pr}: {why}")


def xfail_decide(issue, why):
    return pytest.mark.xfail(strict=True, reason=f"trnsci/trntensor#{issue}: {why}")


XF_COMPLEX_PREC = xfail_open(69, 70, "kahan/dd/target_forward_error drop the imaginary part")
XF_INT_PREC = xfail_open(69, 70, "kahan/dd round large integers")
XF_MULTI_DTYPE = xfail_open(75, 76, "multi_einsum batching promotes mixed dtypes")
XF_OUT = xfail_decide(77, "needs decision: einsum out= is never written")
XF_CP_DTYPE = xfail_decide(78, "cp_decompose/cp_reconstruct hard-code float32 CPU buffers")
XF_SHARD = xfail_decide(78, "output-parallel sharded einsum does not co-partition operands")
XF_FLOPS = xfail_decide(83, "ContractionPlan.estimated_flops is never filled in")


# --------------------------------------------------------------------------- #
# 1. index order: every ordering of matmul/bmm-shaped contractions (#64)       #
# --------------------------------------------------------------------------- #


def _matmul_orders():
    subs = []
    for a, b, o in itertools.product(("ij", "ji"), ("jk", "kj"), ("ik", "ki")):
        subs.append(f"{a},{b}->{o}")
        subs.append(f"{b},{a}->{o}")
    return subs


MATMUL_ORDERS = _matmul_orders()
MATMUL_SIZES = [(1, 1, 1), (1, 127, 129), (127, 129, 1), (129, 1, 127), (3, 4, 5), (127, 129, 1000)]


@pytest.mark.parametrize("ijk", MATMUL_SIZES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("sub", MATMUL_ORDERS)
def test_matmul_index_order(sub, ijk):
    gen = torch.Generator().manual_seed(sum(map(ord, sub)))
    sizes = dict(zip("ijk", ijk, strict=True))
    ops = ops_for(sub, sizes, torch.float32, gen)
    assert_ref(tt.einsum(sub, *[d(o) for o in ops]), ref_einsum(sub, *ops))


BMM_ORDERS = [
    f"{''.join(a)},{''.join(b)}->{''.join(o)}"
    for a in itertools.permutations("bij")
    for b in itertools.permutations("bjk")
    for o in itertools.permutations("bik")
]


@pytest.mark.parametrize("sub", BMM_ORDERS)
def test_bmm_index_order(sub):
    gen = torch.Generator().manual_seed(5)
    ops = ops_for(sub, {"b": 2, "i": 3, "j": 4, "k": 5}, torch.float32, gen)
    assert_ref(tt.einsum(sub, *[d(o) for o in ops]), ref_einsum(sub, *ops))


@pytest.mark.parametrize("out", ["".join(p) for p in itertools.permutations("ijab")])
def test_df_mp2_pattern_output_order(out):
    gen = torch.Generator().manual_seed(9)
    sub = f"iap,jbp->{out}"
    ops = ops_for(sub, {"i": 2, "j": 3, "a": 4, "b": 5, "p": 129}, torch.float32, gen)
    assert_ref(tt.einsum(sub, *[d(o) for o in ops]), ref_einsum(sub, *ops))


@pytest.mark.parametrize("sub", ["ij,j->i", "j,ij->i", "ji,j->i", "i,i->", "i,j->ij", "i,j->ji"])
@pytest.mark.parametrize("n", [1, 127, 129, 1000])
def test_vector_contractions_sizes(sub, n):
    gen = torch.Generator().manual_seed(n)
    ops = ops_for(sub, {"i": n, "j": 129}, torch.float32, gen)
    assert_ref(tt.einsum(sub, *[d(o) for o in ops]), ref_einsum(sub, *ops))


# --------------------------------------------------------------------------- #
# 2. generated subscripts, 1-3 operands (audit-3 generator)                    #
# --------------------------------------------------------------------------- #


def _random_subscripts(rng, n_ops):
    pool = list("abcdefg")
    sizes = {c: rng.randint(1, 5) for c in pool}
    terms = []
    for _ in range(n_ops):
        term = rng.sample(pool[:5], rng.randint(1, 3))
        if rng.random() < 0.15:  # repeated index inside a term (diagonal)
            term.append(term[0])
            rng.shuffle(term)
        terms.append("".join(term))
    letters = sorted({c for t in terms for c in t})
    if rng.random() < 0.2:
        return ",".join(terms), sizes  # implicit output
    out = [c for c in letters if rng.random() < 0.55]
    rng.shuffle(out)
    return ",".join(terms) + "->" + "".join(out), sizes


def _generated_cases():
    rng = random.Random(1234)
    cases = []
    for n_ops in (1, 2, 3):
        for k in range(40):
            sub, sizes = _random_subscripts(rng, n_ops)
            cases.append((sub, sizes, torch.float64 if k % 2 else torch.float32))
    return cases


@pytest.mark.parametrize("precision", ["fast", "kahan"])
@pytest.mark.parametrize(
    "sub,sizes,dtype", _generated_cases(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_generated_subscripts(sub, sizes, dtype, precision):
    gen = torch.Generator().manual_seed(len(sub))
    ops = ops_for(sub, sizes, dtype, gen)
    dops = [d(o) for o in ops]
    snap = [o.clone() for o in ops]
    assert_ref(tt.einsum(sub, *dops, precision=precision), ref_einsum(sub, *ops))
    assert all(same_bits(a, b) for a, b in zip(dops, snap, strict=True)), "input mutated"


# --------------------------------------------------------------------------- #
# 3. dtypes x precision modes on strategy-hitting subscripts                   #
# --------------------------------------------------------------------------- #

CANON = [
    ("ij,jk->ik", [(6, 5), (5, 7)]),
    ("ji,jk->ik", [(5, 6), (5, 7)]),
    ("ij,kj->ik", [(6, 5), (7, 5)]),
    ("ij,jk->ki", [(6, 5), (5, 7)]),
    ("bij,bjk->bik", [(3, 4, 5), (3, 5, 6)]),
    ("iap,jbp->ijab", [(3, 4, 5), (3, 4, 5)]),
    ("ij,jk,kl->il", [(4, 5), (5, 6), (6, 3)]),
    ("mi,na,mnP->iaP", [(5, 2), (5, 3), (5, 5, 4)]),
    ("ii->i", [(5, 5)]),
    ("ii->", [(5, 5)]),
    ("ij->ji", [(4, 6)]),
    ("i,j->ij", [(4,), (6,)]),
    ("ij,ij->", [(4, 6), (4, 6)]),
    ("ij,jk", [(6, 5), (5, 7)]),
    ("Ab,bC->AC", [(3, 4), (4, 5)]),
]
DTYPES = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.complex64,
    torch.complex128,
]
MODES = ["fast", "kahan", "sr", "dd", "target"]


def _mode_cases():
    cases = []
    for (sub, shapes), dt, mode in itertools.product(CANON, DTYPES, MODES):
        is_int = not (dt.is_floating_point or dt.is_complex)
        if is_int and mode != "fast" and mode != "kahan":
            continue
        marks = []
        if dt.is_complex and mode in ("kahan", "dd", "target"):
            marks = [XF_COMPLEX_PREC]
        elif dt == torch.int32 and mode == "kahan" and sub == "ii->":
            marks = [XF_INT_PREC]  # returns int32 where torch.einsum returns int64
        cases.append(pytest.param(sub, shapes, dt, mode, marks=marks, id=f"{sub}-{dt}-{mode}"))
    return cases


@pytest.mark.parametrize("sub,shapes,dtype,mode", _mode_cases())
def test_dtype_precision_modes(sub, shapes, dtype, mode):
    _skip_dd_on_trainium(mode)
    gen = torch.Generator().manual_seed(3)
    ops = [rand(s, dtype, gen) for s in shapes]
    kw = {"target_forward_error": 1e-12} if mode == "target" else {"precision": mode}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # SR / output-floor notices
        got = tt.einsum(sub, *[d(o) for o in ops], **kw)
    # "dd" promises K*u^2 + u_out: hold float64 to the float64 tolerance everywhere
    tol = 1e-11 if mode == "dd" and dtype == torch.float64 else None
    assert_ref(got, ref_einsum(sub, *ops), tol=tol)


@pytest.mark.parametrize("precision", ["kahan", "dd"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_integer_precision_modes_exact(dtype, precision):
    _skip_dd_on_trainium(precision)
    # int64 values above 2**53 / int32 above 2**24 must contract exactly
    big = 2**53 + 1 if dtype == torch.int64 else 2**24 + 1
    a = torch.tensor([[big, 2]], dtype=dtype)
    b = torch.tensor([[1], [1]], dtype=dtype)
    got = tt.einsum("ij,jk->ik", d(a), d(b), precision=precision)
    ref = torch.einsum("ij,jk->ik", a, b)
    if not same_bits(got, ref):
        pytest.xfail(XF_INT_PREC.kwargs["reason"])


@pytest.mark.parametrize("cast", ["bf16", "bfloat16", "fp16", "f32", "float32", torch.float64])
@pytest.mark.parametrize("sub,shapes", CANON[:8], ids=[c[0] for c in CANON[:8]])
def test_dtype_argument(sub, shapes, cast):
    names = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16}
    names |= {"f32": torch.float32, "float32": torch.float32}
    target = names.get(cast, cast)
    gen = torch.Generator().manual_seed(4)
    ops = [torch.randn(s, generator=gen) for s in shapes]
    got = tt.einsum(sub, *[d(o) for o in ops], dtype=cast)
    assert_ref(got, ref_einsum(sub, *[o.to(target) for o in ops]))


@pytest.mark.parametrize("sub", ["ij,jk->ik", "ij,jk->ki", "bij,bjk->bik"])
def test_mixed_dtype_operands(sub):
    """torch.einsum rejects mixed dtypes: raise too, or return the promoted result."""
    gen = torch.Generator().manual_seed(6)
    shapes = dict(CANON)[sub]
    a = torch.randn(shapes[0], generator=gen)
    b = torch.randn(shapes[1], generator=gen, dtype=torch.float64)
    try:
        got = tt.einsum(sub, d(a), d(b))
    except (RuntimeError, TypeError, ValueError):
        return
    assert_ref(got, ref_einsum(sub, a.double(), b), tol=1e-6)


# --------------------------------------------------------------------------- #
# 4. documented forward-error bounds (componentwise)                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("K", [1, 127, 1000])
@pytest.mark.parametrize(
    "dtype,mode",
    [
        (torch.float32, "fast"),
        (torch.float32, "kahan"),
        (torch.float32, "dd"),
        (torch.bfloat16, "fast"),
        (torch.bfloat16, "kahan"),
        (torch.bfloat16, "dd"),
        (torch.float64, "fast"),
        (torch.float64, "kahan"),
        (torch.float64, "dd"),
    ],
    ids=str,
)
def test_forward_error_bound_holds(dtype, mode, K):
    _skip_dd_on_trainium(mode)
    if not _strict() and mode == "fast":
        pytest.skip("device fast mode accumulates in PSUM fp32; bound is for host order")
    gen = torch.Generator().manual_seed(K)
    A = torch.randn(9, K, generator=gen).to(dtype)
    B = torch.randn(K, 11, generator=gen).to(dtype)
    got = tt.einsum("ij,jk->ik", d(A), d(B), precision=mode)
    assert got.dtype == dtype
    exact = A.double() @ B.double()
    scale = A.double().abs() @ B.double().abs()
    err = float(((host(got).double() - exact).abs() / scale.clamp_min(1e-300)).max())
    # small slack for the final rounding being half-ulp vs ulp
    assert err <= 2 * tt.forward_error_bound(mode, dtype, K), err


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("K", [1, 64, 4096])
@pytest.mark.parametrize("mode", ["fast", "sr", "dd", "kahan"])
def test_forward_error_bound_formula(mode, K, dtype):
    u = {torch.bfloat16: 2**-8, torch.float16: 2**-11, torch.float32: 2**-24}.get(dtype, 2**-53)
    accum = {"fast": K * u, "sr": K**0.5 * u, "dd": K * u * u, "kahan": K * 2**-53}[mode]
    assert tt.forward_error_bound(mode, dtype, K) == pytest.approx(accum + u, rel=1e-12)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("K", [1, 300, 10**6])
@pytest.mark.parametrize("target", [1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-10, 1e-15])
def test_select_precision_is_cheapest(dtype, K, target):
    # documented: "sr" and "dd" are only considered for bf16/fp16 operands
    low = dtype in (torch.bfloat16, torch.float16)
    order = ["fast", "sr", "dd", "kahan"] if low else ["fast", "kahan"]
    ok = [m for m in order if tt.forward_error_bound(m, dtype, K) <= target]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            got = tt.select_precision_for_error(dtype, K, target)
        except ValueError:
            # documented: raises only when exact fp64 accumulation cannot meet it
            assert target < K * 2**-53
            return
    if ok:
        assert got == ok[0]
    else:  # below the output floor: documented to run "kahan"
        assert got == "kahan"


def test_einsum_rejects_bad_precision():
    a = torch.randn(2, 2)
    with pytest.raises(ValueError):
        tt.einsum("ij,jk->ik", a, a, precision="quad")
    with pytest.raises((ValueError, TypeError)):
        tt.einsum("ij,jk->ik", a, a, precision="kahan", target_forward_error=1e-3)


# --------------------------------------------------------------------------- #
# 5. edge shapes                                                              #
# --------------------------------------------------------------------------- #

EDGE = [
    ("ij,jk->ik", [(0, 5), (5, 7)]),
    ("ij,jk->ik", [(6, 0), (0, 7)]),
    ("ij,jk->ik", [(6, 5), (5, 0)]),
    ("ij,jk->ki", [(6, 0), (0, 7)]),
    ("bij,bjk->bik", [(0, 4, 5), (0, 5, 6)]),
    ("bij,bjk->bik", [(3, 4, 0), (3, 0, 6)]),
    ("ij,jk,kl->il", [(4, 0), (0, 6), (6, 3)]),
    ("ij,jk->ik", [(1, 1), (1, 1)]),
    ("ij,jk->ik", [(1, 5), (5, 1)]),
    ("bij,bjk->bik", [(1, 1, 5), (1, 5, 1)]),
    ("ij,jk->ik", [(2048, 2), (2, 3)]),
    ("ij,jk->ik", [(2, 4096), (4096, 2)]),
    ("ij,jk->ik", [(3, 2), (2, 4099)]),
    (",->", [(), ()]),
    ("i,->i", [(4,), ()]),
    ("i->", [(0,)]),
    ("ij->j", [(0, 3)]),
    ("ii->i", [(0, 0)]),
    ("iij->j", [(3, 3, 4)]),
    ("iji->j", [(3, 4, 3)]),
    ("ij,ji->", [(4, 5), (5, 4)]),
    ("iap,jbp->ijab", [(1, 1, 1), (1, 1, 1)]),
    ("iap,jbp->ijab", [(2, 3, 0), (2, 3, 0)]),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sub,shapes", EDGE, ids=[f"{s}-{sh}" for s, sh in EDGE])
def test_edge_shapes(sub, shapes, dtype):
    gen = torch.Generator().manual_seed(8)
    ops = [torch.randn(s, generator=gen, dtype=dtype) for s in shapes]
    assert_ref(tt.einsum(sub, *[d(o) for o in ops]), ref_einsum(sub, *ops))


@pytest.mark.parametrize(
    "sub,shapes",
    [
        ("->", [()]),
        ("...ij,...jk->...ik", [(2, 3, 4), (2, 4, 5)]),
        ("i...->...", [(3, 4, 5)]),
        ("ij -> ji", [(3, 4)]),
        ("bij,bjk->bik", [(1, 4, 5), (3, 5, 6)]),  # size-1 broadcast
        ("ij,j->i", [(3, 1), (4,)]),  # size-1 broadcast
    ],
)
def test_unsupported_notation_is_loud(sub, shapes):
    """Ellipsis, spaces and size-1 broadcast: match torch.einsum or raise ValueError."""
    gen = torch.Generator().manual_seed(2)
    ops = [torch.randn(s, generator=gen) for s in shapes]
    try:
        got = tt.einsum(sub, *[d(o) for o in ops])
    except ValueError:
        return
    assert_ref(got, ref_einsum(sub, *ops))


# --------------------------------------------------------------------------- #
# 6. non-contiguous operands                                                   #
# --------------------------------------------------------------------------- #

VIEWS = {
    "fortran": lambda b: b[:6, :5].T.contiguous().T,
    "transposed": lambda b: b[:5, :6].T,
    "step": lambda b: b[::2, ::2][:6, :5],
    "flip": lambda b: b[:6, :5].flip(0),
    "expanded": lambda b: b[0:1, :5].expand(6, 5),
    "as_strided": lambda b: b.as_strided((6, 5), (3, 1)),
    "stride0col": lambda b: b[:6, 0:1].expand(6, 5),
}
VIEW_SUBS = ["ij,jk->ik", "ji,ik->jk", "ij,jk->ki", "ij->ji", "ij,ij->i", "ij,jk,kl->il"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sub", VIEW_SUBS)
@pytest.mark.parametrize("view", list(VIEWS))
def test_non_contiguous_operands(view, sub, dtype):
    gen = torch.Generator().manual_seed(10)
    base = torch.randn(12, 14, generator=gen, dtype=dtype)
    a = VIEWS[view](base)
    b = torch.randn(5, 7, generator=gen, dtype=dtype)
    ops = {
        "ij,jk->ik": (a, b),
        "ji,ik->jk": (a.T, a),
        "ij,jk->ki": (a, b),
        "ij->ji": (a,),
        "ij,ij->i": (a, a),
        "ij,jk,kl->il": (a, b, b.T),
    }[sub]
    assert_ref(tt.einsum(sub, *[d(o) for o in ops]), ref_einsum(sub, *ops))


def test_non_contiguous_batched_and_channels_last():
    gen = torch.Generator().manual_seed(12)
    base3 = torch.randn(4, 10, 12, generator=gen, dtype=torch.float64)
    a3, b3 = base3[:, :, :5], base3.transpose(1, 2)[:, :5, :7]
    assert_ref(tt.einsum("bij,bjk->bik", d(a3), d(b3)), ref_einsum("bij,bjk->bik", a3, b3))
    x = torch.randn(2, 3, 4, 5, generator=gen).to(memory_format=torch.channels_last)
    w = torch.randn(6, 3, generator=gen)
    wk = torch.randn(2, 5, 3, generator=gen)
    assert_ref(tt.einsum("nchw,oc->nohw", d(x), d(w)), ref_einsum("nchw,oc->nohw", x, w))
    assert_ref(tt.einsum("nchw,nwk->nchk", d(x), d(wk)), ref_einsum("nchw,nwk->nchk", x, wk))
    xc = torch.randn(3, 5, 6, generator=gen).permute(0, 2, 1)
    yc = torch.randn(3, 7, 5, generator=gen).transpose(1, 2)
    assert_ref(tt.einsum("bij,bjk->bik", d(xc), d(yc)), ref_einsum("bij,bjk->bik", xc, yc))


# --------------------------------------------------------------------------- #
# 7. alpha / beta / out= / aliasing                                            #
# --------------------------------------------------------------------------- #

ALPHA_SUBS = [
    ("ij,jk->ik", [(6, 5), (5, 7)]),
    ("ij,jk->ki", [(6, 5), (5, 7)]),
    ("bij,bjk->bik", [(3, 4, 5), (3, 5, 6)]),
    ("ij,jk,kl->il", [(4, 5), (5, 6), (6, 3)]),
    ("ij->ji", [(4, 6)]),
    ("ij->ij", [(4, 6)]),
]
ALPHA_BETA = [(2.0, 0.0), (2.0, 0.5), (1.0, 1.0), (-1.0, 3.0), (0.0, 1.0), (0.0, 0.0)]


@pytest.mark.parametrize("alpha,beta", ALPHA_BETA)
@pytest.mark.parametrize("sub,shapes", ALPHA_SUBS, ids=[s for s, _ in ALPHA_SUBS])
def test_alpha_beta_out(sub, shapes, alpha, beta):
    gen = torch.Generator().manual_seed(7)
    ops = [torch.randn(s, generator=gen) for s in shapes]
    C = torch.randn(torch.einsum(sub, *ops).shape, generator=gen)
    dops, dC = [d(o) for o in ops], d(C)
    snap = [o.clone() for o in ops]
    got = tt.einsum(sub, *dops, alpha=alpha, beta=beta, out=dC)
    assert_ref(got, (alpha * ref_einsum(sub, *ops).double() + beta * C.double()).float())
    assert all(same_bits(a, b) for a, b in zip(dops, snap, strict=True)), "operand mutated"


@pytest.mark.parametrize("sub,shapes", ALPHA_SUBS, ids=[s for s, _ in ALPHA_SUBS])
def test_alpha_and_beta0_nonfinite_out(sub, shapes):
    gen = torch.Generator().manual_seed(17)
    ops = [torch.randn(s, generator=gen) for s in shapes]
    dops = [d(o) for o in ops]
    snap = [o.clone() for o in ops]
    assert_ref(tt.einsum(sub, *dops, alpha=3.0), 3 * ref_einsum(sub, *ops))
    assert all(same_bits(a, b) for a, b in zip(dops, snap, strict=True)), "input mutated"
    nan = torch.full(torch.einsum(sub, *ops).shape, float("nan"))
    # BLAS / cuTENSOR / torch.addmm convention: beta=0 does not read out
    assert_ref(tt.einsum(sub, *dops, beta=0.0, out=d(nan)), ref_einsum(sub, *ops))


def test_out_is_an_operand():
    gen = torch.Generator().manual_seed(18)
    A = torch.randn(4, 6, generator=gen)
    dA = d(A.clone())
    got = tt.einsum("ij->ij", dA, alpha=2.0, beta=1.0, out=dA)
    assert_ref(got, 3 * A)


@XF_OUT
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_out_is_written(beta):
    """torch.einsum(out=) and cuTENSOR write the output operand; the docs example relies on it."""
    gen = torch.Generator().manual_seed(19)
    A, B = torch.randn(4, 5, generator=gen), torch.randn(5, 3, generator=gen)
    acc = d(torch.ones(4, 3))
    ret = tt.einsum("ij,jk->ik", d(A), d(B), beta=beta, out=acc)
    assert_ref(acc, (A.double() @ B.double() + beta).float())
    assert ret.data_ptr() == acc.data_ptr()


def test_results_do_not_alias_across_calls():
    gen = torch.Generator().manual_seed(20)
    A = torch.randn(4, 6, generator=gen)
    r1 = tt.einsum("ij->ji", d(A))
    r1c = host(r1).clone()
    tt.einsum("ij->ji", d(A), alpha=5.0)
    assert same_bits(r1, r1c)
    X, Y = torch.randn(4, 5, generator=gen), torch.randn(5, 6, generator=gen)
    res = tt.multi_einsum(("ij,jk->ik", d(X), d(Y)), ("ij,jk->ik", d(X * 2), d(Y)))
    res[0].add_(1000)
    assert_ref(res[1], (2 * X.double() @ Y.double()).float())


# --------------------------------------------------------------------------- #
# 8. autograd (fp64, host)                                                     #
# --------------------------------------------------------------------------- #


def _gradcheck(fn, inputs, gg=True):
    if not _strict():
        pytest.skip("fp64 gradcheck runs on the host pytorch backend only")
    ins = [x.detach().clone().requires_grad_(True) for x in inputs]
    kw = {"eps": 1e-6, "atol": 1e-5, "rtol": 1e-4}
    assert torch.autograd.gradcheck(fn, ins, **kw)
    if gg:
        assert torch.autograd.gradgradcheck(fn, ins, **kw)


GRAD_SUBS = [
    ("ij,jk->ik", [(4, 3), (3, 5)]),
    ("ji,jk->ik", [(3, 4), (3, 5)]),
    ("ij,kj->ik", [(4, 3), (5, 3)]),
    ("ij,jk->ki", [(4, 3), (3, 5)]),
    ("bij,bjk->bik", [(2, 3, 4), (2, 4, 2)]),
    ("bij,bkj->bki", [(2, 3, 4), (2, 2, 4)]),
    ("ij,jk,kl->il", [(3, 4), (4, 2), (2, 3)]),
    ("ii->i", [(4, 4)]),
    ("iap,jbp->ijab", [(2, 3, 4), (2, 3, 4)]),
]


@pytest.mark.parametrize("precision", ["fast", "kahan"])
@pytest.mark.parametrize("sub,shapes", GRAD_SUBS, ids=[s for s, _ in GRAD_SUBS])
def test_gradcheck_einsum(sub, shapes, precision):
    gen = torch.Generator().manual_seed(11)
    ops = [torch.randn(s, generator=gen, dtype=torch.float64) for s in shapes]
    _gradcheck(lambda *x: tt.einsum(sub, *x, precision=precision, alpha=2.0), ops)


@pytest.mark.parametrize("sub,shapes", GRAD_SUBS, ids=[s for s, _ in GRAD_SUBS])
def test_grad_matches_torch_einsum(sub, shapes):
    gen = torch.Generator().manual_seed(13)
    ops = [torch.randn(s, generator=gen, dtype=torch.float64) for s in shapes]
    a = [d(o.clone()).requires_grad_(True) for o in ops]
    b = [o.clone().requires_grad_(True) for o in ops]
    out = tt.einsum(sub, *a, alpha=-1.5)
    w = torch.randn(out.shape, generator=gen, dtype=torch.float64)
    (out * d(w)).sum().backward()
    (-1.5 * torch.einsum(sub, *b) * w).sum().backward()
    for x, y in zip(a, b, strict=True):
        assert_ref(x.grad, y.grad, tol=1e-9 if _strict() else 2e-6)


def test_gradcheck_alpha_beta_out():
    gen = torch.Generator().manual_seed(14)
    A, B, C = (torch.randn(s, generator=gen, dtype=torch.float64) for s in ((4, 3), (3, 5), (4, 5)))
    _gradcheck(lambda a, b, c: tt.einsum("ij,jk->ik", a, b, alpha=2.0, beta=0.5, out=c), [A, B, C])


def test_gradcheck_reconstructions_and_chemistry():
    gen = torch.Generator().manual_seed(15)

    def r(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float64)

    f1, w1 = [r(4, 2) for _ in range(3)], torch.rand(2, generator=gen, dtype=torch.float64)
    _gradcheck(lambda *x: tt.cp_reconstruct(list(x[:-1]), x[-1]), [*f1, w1])
    _gradcheck(
        lambda c, *f: tt.tucker_reconstruct(c, list(f)), [r(2, 3, 2), r(4, 2), r(5, 3), r(3, 2)]
    )
    _gradcheck(lambda *c: tt.tt_reconstruct(list(c)), [r(1, 3, 2), r(2, 4, 2), r(2, 3, 1)])
    eo = -torch.rand(3, generator=gen, dtype=torch.float64) - 1
    ev = torch.rand(4, generator=gen, dtype=torch.float64) + 1
    _gradcheck(tt.mp2_energy, [r(3, 4, 5), eo, ev])
    eri = r(4, 4, 3)
    eri = eri + eri.transpose(0, 1)
    pk = tt.pack_tril(eri)
    co, cv = r(4, 2), r(4, 3)
    _gradcheck(tt.ao_to_mo_transform, [eri, co, cv])
    _gradcheck(tt.ao_to_mo_transform, [pk, co, cv])
    _gradcheck(lambda b, c: tt.df_exchange(b, c, alpha=2.0), [pk, co])
    D = r(4, 4)
    _gradcheck(tt.df_coulomb, [pk, D + D.T])
    _gradcheck(tt.pack_tril, [eri])
    _gradcheck(lambda p: tt.unpack_tril(p, 4), [pk])


# --------------------------------------------------------------------------- #
# 9. chemistry helpers vs dense formulas                                        #
# --------------------------------------------------------------------------- #


def dense_mp2(B, eo, ev):
    B, eo, ev = (host(x).double().numpy() for x in (B, eo, ev))
    T = np.einsum("iap,jbp->ijab", B, B)
    den = eo[:, None, None, None] + eo[None, :, None, None]
    den = den - ev[None, None, :, None] - ev[None, None, None, :]
    e = float((T * (2 * T - T.transpose(0, 1, 3, 2)) / den).sum())
    return torch.tensor(e, dtype=torch.float64)


MP2_SIZES = [(3, 5, 7), (1, 1, 1), (4, 6, 1), (0, 3, 2), (2, 0, 3), (2, 3, 0), (5, 9, 130)]
MP2_SIZES += [(2, 127, 129)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("sizes", MP2_SIZES, ids=lambda s: "x".join(map(str, s)))
def test_mp2_energy(sizes, dtype):
    nocc, nvir, naux = sizes
    gen = torch.Generator().manual_seed(21)
    B = (torch.randn(nocc, nvir, naux, generator=gen) / naux**0.5).to(dtype)
    eo = (-torch.rand(nocc, generator=gen) - 1).to(dtype)
    ev = (torch.rand(nvir, generator=gen) + 1).to(dtype)
    got = tt.mp2_energy(d(B), d(eo), d(ev))
    tol = 0.15 if dtype == torch.bfloat16 else (1e-4 if dtype == torch.float32 else None)
    assert_ref(got, dense_mp2(B, eo, ev).to(dtype), tol=tol)


def test_mp2_energy_non_contiguous_and_mixed():
    gen = torch.Generator().manual_seed(22)
    Bnc = torch.randn(5, 8, 12, generator=gen, dtype=torch.float64)[:, ::2, :7]
    eo = -torch.rand(5, generator=gen, dtype=torch.float64) - 1
    ev = torch.rand(4, generator=gen, dtype=torch.float64) + 1
    assert_ref(tt.mp2_energy(d(Bnc), d(eo), d(ev)), dense_mp2(Bnc, eo, ev))
    got = tt.mp2_energy(d(Bnc.float()), d(eo), d(ev))
    assert_ref(got, dense_mp2(Bnc, eo, ev), tol=1e-4, check_dtype=False)


AO_SIZES = [(5, 2, 3, 4), (6, 1, 5, 1), (4, 0, 2, 3), (3, 3, 0, 2), (4, 2, 2, 0), (129, 2, 3, 127)]


def _sym_eri(nb, naux, gen, dtype):
    e = torch.randn(nb, nb, naux, generator=gen, dtype=torch.float64)
    return (e + e.transpose(0, 1)).to(dtype)


@pytest.mark.parametrize("form", ["full", "packed", "pyscf"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("sizes", AO_SIZES, ids=lambda s: "x".join(map(str, s)))
def test_ao_to_mo_transform(sizes, dtype, form):
    nb, nocc, nvir, naux = sizes
    gen = torch.Generator().manual_seed(23)
    e = _sym_eri(nb, naux, gen, dtype)
    co = torch.randn(nb, nocc, generator=gen).to(dtype)
    cv = torch.randn(nb, nvir, generator=gen).to(dtype)
    eri = {"full": e, "packed": tt.pack_tril(e), "pyscf": tt.pack_tril(e).T.contiguous()}[form]
    if form == "pyscf" and eri.shape[0] == eri.shape[1]:
        pytest.skip("naux == npair is ambiguous; documented to pass cderi.T")
    ref = torch.einsum("mi,na,mnP->iaP", co.double(), cv.double(), e.double()).to(dtype)
    deri = d(eri)
    snap = eri.clone()
    assert_ref(tt.ao_to_mo_transform(deri, d(co), d(cv)), ref)
    assert same_bits(deri, snap), "eri mutated"


def test_ao_to_mo_non_contiguous():
    gen = torch.Generator().manual_seed(24)
    e = _sym_eri(6, 5, gen, torch.float64)
    Cfull = torch.randn(6, 8, generator=gen, dtype=torch.float64)
    co, cv = Cfull[:, :2], Cfull[:, 2::2]
    ref = torch.einsum("mi,na,mnP->iaP", co, cv, e)
    assert_ref(tt.ao_to_mo_transform(d(e), d(co), d(cv)), ref)
    ref_t = torch.einsum("mi,na,nmP->iaP", co, cv, e)
    assert_ref(tt.ao_to_mo_transform(d(e.transpose(0, 1)), d(co), d(cv)), ref_t)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.complex64, torch.bfloat16])
@pytest.mark.parametrize("nb,naux", [(1, 1), (5, 3), (7, 1), (0, 2), (4, 0), (129, 2)])
def test_pack_unpack_tril(nb, naux, dtype):
    gen = torch.Generator().manual_seed(25)
    x = rand((nb, nb, naux), dtype, gen)
    x = x + x.transpose(0, 1)
    r = torch.tril_indices(nb, nb)
    packed_ref = x[r[0], r[1]]
    assert_ref(tt.pack_tril(d(x)), packed_ref)
    assert_ref(tt.pack_tril(d(x.transpose(0, 1))), packed_ref)
    assert_ref(tt.unpack_tril(tt.pack_tril(d(x)), nb), x)


DF_SIZES = [(5, 2, 4), (4, 1, 1), (3, 0, 2), (6, 3, 0), (1, 1, 1), (129, 3, 33)]


@pytest.mark.parametrize("form", ["packed", "pyscf", "full"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sizes", DF_SIZES, ids=lambda s: "x".join(map(str, s)))
def test_df_exchange(sizes, dtype, form):
    nb, nocc, naux = sizes
    gen = torch.Generator().manual_seed(26)
    e = _sym_eri(nb, naux, gen, dtype)
    co = torch.randn(nb, nocc, generator=gen).to(dtype)
    pk = tt.pack_tril(e)
    B = {"packed": pk, "pyscf": pk.T.contiguous(), "full": e}[form]
    if form == "pyscf" and pk.shape[0] == pk.shape[1]:
        pytest.skip("naux == npair is ambiguous")
    ed, cd = e.double(), co.double()
    ref = (2.0 * torch.einsum("lmp,li,knp,ki->mn", ed, cd, ed, cd)).to(dtype)
    assert_ref(tt.df_exchange(d(B), d(co), alpha=2.0), ref)


@pytest.mark.parametrize("dform", ["square", "packed", "transposed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sizes", DF_SIZES, ids=lambda s: "x".join(map(str, s)))
def test_df_coulomb(sizes, dtype, dform):
    nb, _, naux = sizes
    gen = torch.Generator().manual_seed(27)
    e = _sym_eri(nb, naux, gen, dtype)
    D = torch.randn(nb, nb, generator=gen, dtype=torch.float64)
    D = (D + D.T).to(dtype)
    ref = torch.einsum("mnp,lsp,ls->mn", e.double(), e.double(), D.double()).to(dtype)
    r = torch.tril_indices(nb, nb)
    if dform == "packed":
        w = torch.where(r[0] == r[1], 1.0, 2.0).to(dtype)
        Din, ref = D[r[0], r[1]] * w, ref[r[0], r[1]]
    else:
        Din = D.T if dform == "transposed" else D
    assert_ref(tt.df_coulomb(d(tt.pack_tril(e)), d(Din)), ref)


# --------------------------------------------------------------------------- #
# 10. decompositions                                                          #
# --------------------------------------------------------------------------- #


def _tucker_proj(x, ranks):
    P = []
    for m, rk in enumerate(ranks):
        U = torch.linalg.svd(x.moveaxis(m, 0).reshape(x.shape[m], -1), full_matrices=False)[0]
        P.append(U[:, :rk] @ U[:, :rk].conj().T)
    return torch.einsum("abc,ia,jb,kc->ijk", x, *P)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_tucker(dtype):
    gen = torch.Generator().manual_seed(31)
    x = torch.randn(4, 5, 3, generator=gen, dtype=dtype)
    dx = d(x)
    full = tt.tucker_reconstruct(*tt.tucker_decompose(dx, (4, 5, 3)))
    assert_ref(full, x, tol=max(10 * tol_for(dtype), 1e-10))
    core, fs = tt.tucker_decompose(dx, (2, 2, 2))
    assert tuple(core.shape) == (2, 2, 2)
    for f, n in zip(fs, x.shape, strict=True):
        assert tuple(f.shape) == (n, 2)
        eye = torch.eye(2, dtype=dtype)
        assert_ref(host(f).conj().T @ host(f), eye, tol=max(100 * tol_for(dtype), 1e-9))
    # truncated HOSVD reconstruct equals the projection T x_n U_n U_n^H
    assert_ref(tt.tucker_reconstruct(core, fs), _tucker_proj(x, (2, 2, 2)), tol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
@pytest.mark.parametrize("max_rank", [1, 2, 64])
def test_tt_svd_error_bound(dtype, max_rank):
    """Oseledets (2011) Thm 2.2: ||T - TT(T)||^2 <= sum_k tail_k^2 of the unfoldings."""
    gen = torch.Generator().manual_seed(32)
    x = torch.randn(4, 5, 3, 2, generator=gen, dtype=dtype)
    cores = tt.tt_decompose(d(x), max_rank)
    r_prev = 1
    for c, n in zip(cores, x.shape, strict=True):
        assert c.shape[0] == r_prev and c.shape[1] == n and c.shape[2] <= max_rank
        r_prev = c.shape[2]
    assert r_prev == 1
    err = float(torch.linalg.norm((host(tt.tt_reconstruct(cores)) - x).reshape(-1)))
    bound = 0.0
    for k in range(1, x.dim()):
        hi = torch.complex128 if dtype.is_complex else torch.float64
        s = torch.linalg.svdvals(x.reshape(int(np.prod(x.shape[:k])), -1).to(hi))
        bound += float((s[max_rank:] ** 2).sum())
    assert err <= bound**0.5 * (1 + 1e-4) + 10 * tol_for(dtype) * float(torch.linalg.norm(x))


def test_decompose_non_contiguous_and_edge():
    gen = torch.Generator().manual_seed(33)
    xnc = torch.randn(6, 10, 4, generator=gen, dtype=torch.float64)[:, ::2, :].permute(2, 0, 1)
    assert_ref(tt.tt_reconstruct(tt.tt_decompose(d(xnc), 64)), xnc)
    assert_ref(tt.tucker_reconstruct(*tt.tucker_decompose(d(xnc), tuple(xnc.shape))), xnc)
    for shp in ((1, 4, 3), (4, 1, 1), (1, 1)):
        x = torch.randn(*shp, generator=gen, dtype=torch.float64)
        assert_ref(tt.tt_reconstruct(tt.tt_decompose(d(x), 8)), x)
        assert_ref(tt.tucker_reconstruct(*tt.tucker_decompose(d(x), x.shape)), x)
    with pytest.raises(ValueError):
        tt.tt_decompose(torch.randn(4), 2)


@pytest.mark.parametrize("nonneg", [False, True])
def test_cp_rank1_recovery_fp32(nonneg):
    gen = torch.Generator().manual_seed(34)
    a, b, c = (torch.rand(n, generator=gen) + 0.1 for n in (4, 5, 3))
    x = torch.einsum("i,j,k->ijk", a, b, c)
    torch.manual_seed(0)
    factors, w = tt.cp_decompose(d(x), 1, max_iter=200, nonneg=nonneg)
    assert len(factors) == 3 and tuple(w.shape) == (1,)
    if nonneg:
        assert all(bool((host(f) >= 0).all()) for f in factors)
    assert_ref(tt.cp_reconstruct(factors, w), x, tol=1e-2 if nonneg else 1e-3)


def test_cp_low_rank_and_warm_start_fp32():
    gen = torch.Generator().manual_seed(35)
    A, B, C = (torch.randn(n, 2, generator=gen) for n in (8, 6, 4))
    x = torch.einsum("ir,jr,kr->ijk", A, B, C)
    snap = A.clone()
    fs, w = tt.cp_decompose(d(x), 2, max_iter=300, tol=1e-8, factors=[d(A), d(B), d(C)])
    assert_ref(tt.cp_reconstruct(fs, w), x, tol=1e-4)
    # warm start must not mutate the caller's factors
    assert same_bits(A, snap)


def test_cp_reconstruct_formula():
    gen = torch.Generator().manual_seed(36)
    for dt in (torch.float32, torch.float64, torch.complex64):
        f = [torch.randn(n, 3, generator=gen, dtype=dt) for n in (3, 4, 2, 2)]
        w = torch.rand(3, generator=gen, dtype=dt)
        got = tt.cp_reconstruct([d(x) for x in f], d(w))
        tol = 1e-10 if dt == torch.float64 else 1e-5
        assert_ref(got, torch.einsum("r,ir,jr,kr,lr->ijkl", w, *f), tol=tol, check_dtype=False)


@XF_CP_DTYPE
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cp_reconstruct_keeps_dtype(dtype):
    gen = torch.Generator().manual_seed(37)
    f = [torch.randn(n, 2, generator=gen).to(dtype) for n in (3, 4, 2)]
    w = torch.rand(2, generator=gen).to(dtype)
    assert_ref(tt.cp_reconstruct([d(x) for x in f], d(w)), torch.einsum("r,ir,jr,kr->ijk", w, *f))


@XF_CP_DTYPE
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex64])
def test_cp_decompose_non_fp32(dtype, warm):
    gen = torch.Generator().manual_seed(38)
    vs = [torch.randn(n, generator=gen, dtype=dtype) for n in (4, 5, 3)]
    x = torch.einsum("i,j,k->ijk", *vs)
    fac = [d(v[:, None].clone()) for v in vs] if warm else None
    fs, w = tt.cp_decompose(d(x), 1, max_iter=50, factors=fac)
    assert fs[0].dtype == dtype
    assert_ref(tt.cp_reconstruct(fs, w), x, tol=1e-6)


def test_cp_decompose_seeded_determinism():
    x = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(39))
    torch.manual_seed(5)
    f1, w1 = tt.cp_decompose(d(x), 2, max_iter=20)
    torch.manual_seed(5)
    f2, w2 = tt.cp_decompose(d(x), 2, max_iter=20)
    assert all(same_bits(a, b) for a, b in zip(f1, f2, strict=True)) and same_bits(w1, w2)


# --------------------------------------------------------------------------- #
# 11. multi_einsum, sharding, plans, transfer helpers                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", ["fast", "kahan"])
@pytest.mark.parametrize("sub", ["ij,jk->ik", "ij,kj->ik", "ij,jk->ki", "bij,bjk->bik"])
def test_multi_einsum_homogeneous(sub, precision):
    gen = torch.Generator().manual_seed(41)
    shapes = {"ij,jk->ik": [(4, 5), (5, 6)], "ij,kj->ik": [(4, 5), (6, 5)]}
    shapes |= {"ij,jk->ki": [(4, 5), (5, 6)], "bij,bjk->bik": [(2, 4, 5), (2, 5, 6)]}
    calls = [[torch.randn(s, generator=gen) for s in shapes[sub]] for _ in range(3)]
    got = tt.multi_einsum(*[(sub, *[d(o) for o in c]) for c in calls], precision=precision)
    assert_ref(list(got), [ref_einsum(sub, *c) for c in calls])


def test_multi_einsum_heterogeneous():
    gen = torch.Generator().manual_seed(42)
    X = [torch.randn(4, 5, generator=gen) for _ in range(3)]
    Y = torch.randn(5, 6, generator=gen)
    calls = [("ij,jk->ik", X[0], Y), ("ij,jk->ki", X[1], Y), ("ii->", X[2][:4, :4])]
    got = tt.multi_einsum(*[(s, *[d(o) for o in ops]) for s, *ops in calls])
    assert_ref(list(got), [ref_einsum(s, *ops) for s, *ops in calls])


@XF_MULTI_DTYPE
def test_multi_einsum_mixed_dtype_batch():
    gen = torch.Generator().manual_seed(43)
    X, Y = torch.randn(4, 5, generator=gen), torch.randn(5, 6, generator=gen)
    got = tt.multi_einsum(("ij,jk->ik", d(X), d(Y)), ("ij,jk->ik", d(X.double()), d(Y.double())))
    assert_ref(list(got), [ref_einsum("ij,jk->ik", X, Y), ref_einsum("ij,jk->ik", X.double(), Y)])


_A = torch.randn(7, 5, generator=torch.Generator().manual_seed(44), dtype=torch.float64)
_B = torch.randn(5, 6, generator=torch.Generator().manual_seed(45), dtype=torch.float64)
_C = torch.randn(6, 3, generator=torch.Generator().manual_seed(46), dtype=torch.float64)
_x = torch.randn(7, generator=torch.Generator().manual_seed(47), dtype=torch.float64)
SHARD_CASES = [
    ("ij,jk->ik", (_A, _B), [(0, 0, 3)]),
    ("ij,jk->ki", (_A, _B), [(0, 0, 3)]),
    ("ij,jk->ik", (_A, _B), [(0, 1, 2)]),
    ("ij,jk->ik", (_A, _B), [(1, 0, 2)]),
    ("ij,jk->ik", (_A, _B), [(0, 1, 2), (1, 0, 2)]),
    ("ij,jk->ik", (_A, _B), [(0, 0, 3), (1, 0, 2)]),
    ("ij,jk,kl->il", (_A, _B, _C), [(1, 1, 3)]),
    ("ij,jk,kl->il", (_A, _B, _C), [(0, 0, 2)]),
    ("ij,jk->ik", (_A, _B), [(1, 1, 3)]),
    ("ij,ij->i", (_A, _A), [(0, 1, 2)]),
    ("ij,jk->ik", (_A, _B), [(0, 0, 7)]),
    pytest.param("ij,ij->ij", (_A, _A), [(0, 0, 3)], marks=XF_SHARD),
    pytest.param("ij,i->ij", (_A, _x), [(0, 0, 3)], marks=XF_SHARD),
]


@pytest.mark.parametrize("alpha", [1.0, -2.0])
@pytest.mark.parametrize("sub,ops,shard", SHARD_CASES)
def test_sharded_einsum(sub, ops, shard, alpha):
    o = [d(x) for x in ops]
    for opi, dim, n in shard:
        o[opi] = tt.scatter(o[opi], dim, n)
    r = tt.einsum(sub, *o, alpha=alpha)
    r = tt.gather(r) if isinstance(r, tt.ShardedTensor) else r
    assert_ref(r, alpha * torch.einsum(sub, *ops))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_scatter_gather_round_trip(n):
    gen = torch.Generator().manual_seed(48)
    z = torch.randn(6, 7, generator=gen).T  # non-contiguous
    st = tt.scatter(d(z), 1, n)
    assert isinstance(st, tt.ShardedTensor)
    assert len(st.chunks) == n and sum(host(c).shape[1] for c in st.chunks) == 6
    assert same_bits(tt.gather(st), z)
    assert same_bits(tt.gather(tt.scatter(d(torch.zeros(5, 0)), 0, 2)), torch.zeros(5, 0))


def _expected_plan_indices(sub):
    ins, out = sub.split("->")
    a, b = ins.split(",")
    contracted = (set(a) & set(b)) - set(out)
    batch = set(a) & set(b) & set(out)
    return contracted, batch, list(out)


@pytest.mark.parametrize("sub", MATMUL_ORDERS[:8] + BMM_ORDERS[::23])
def test_plan_metadata(sub):
    gen = torch.Generator().manual_seed(49)
    ops = ops_for(sub, {"i": 3, "j": 4, "k": 5, "b": 2}, torch.float32, gen)
    plan = tt.plan_contraction(sub, *ops)
    contracted, batch, out = _expected_plan_indices(sub)
    assert set(plan.contraction_indices) == contracted
    assert set(plan.batch_indices) == batch
    assert plan.output_indices == out
    assert plan.strategy in ("matmul", "bmm", "torch")
    if plan.strategy == "matmul":
        a, b = sub.split("->")[0].split(",")
        (c,) = contracted
        assert plan.transA == (a.index(c) == 0) and plan.transB == (b.index(c) == 1)


@pytest.mark.parametrize(
    "sub,shapes",
    [
        ("ij,jk->ik", [(6, 5), (5, 7)]),
        ("bij,bjk->bik", [(3, 4, 5), (3, 5, 6)]),
        ("ij,jk->ki", [(6, 5), (5, 7)]),
        ("ij,jk,kl->il", [(4, 5), (5, 6), (6, 3)]),
        ("ii->i", [(5, 5)]),
        ("ij,jk->ik", [(0, 5), (5, 7)]),
    ],
)
def test_estimate_flops(sub, shapes):
    ops = [torch.randn(s) for s in shapes]
    sizes = {}
    for term, o in zip(sub.split("->")[0].split(","), ops, strict=True):
        sizes.update(zip(term, o.shape, strict=True))
    assert tt.estimate_flops(sub, *ops) == int(np.prod(list(sizes.values())))


@XF_FLOPS
@pytest.mark.parametrize(
    "sub,shapes", [("ij,jk->ik", [(6, 5), (5, 7)]), ("ij,jk->ki", [(6, 5), (5, 7)])]
)
def test_plan_estimated_flops_filled(sub, shapes):
    ops = [torch.randn(s) for s in shapes]
    assert tt.plan_contraction(sub, *ops).estimated_flops == tt.estimate_flops(sub, *ops)


def test_plan_cache():
    tt.clear_plan_cache()
    assert tt.plan_cache_info()["size"] == 0
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    p1 = tt.plan_contraction("ij,jk->ik", a, b)
    p2 = tt.plan_contraction("ij,jk->ik", a, b)
    assert p1 is p2 and tt.plan_cache_info()["size"] == 1
    assert tt.plan_contraction("ij,jk->ik", a, b, precision="kahan").precision == "kahan"
    # a cached real plan must not change the result for complex or re-laid-out operands
    assert_ref(tt.einsum("ij,jk->ik", d(a.T.contiguous().T), d(b)), ref_einsum("ij,jk->ik", a, b))
    ac, bc = a.to(torch.complex64) * 1j, b.to(torch.complex64)
    assert_ref(tt.einsum("ij,jk->ik", d(ac), d(bc)), ref_einsum("ij,jk->ik", ac, bc))
    with pytest.raises(ValueError):
        tt.plan_contraction("ij,jk->ik", a, b, precision="bogus")
    tt.clear_plan_cache()


def test_einsum_repeatable():
    x = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(50), dtype=torch.float64)
    assert same_bits(tt.einsum("iap,jbp->ijab", d(x), d(x)), tt.einsum("iap,jbp->ijab", d(x), d(x)))


def test_device_transfer_helpers():
    x = torch.randn(3, 4)
    assert same_bits(tt.from_device(x), x)
    if DEV.type != "cpu":
        y = tt.to_device(x)
        assert y.device.type != "cpu"
        assert same_bits(tt.from_device(y), x)
