"""Tail-tile audit: every NKI kernel at awkward sizes against an fp64 reference.

trnsci/trnfft#100 found released kernels that dropped the partial tail tile
in M, K or N and returned silently wrong results, because the suites only
used friendly sizes. These cases run each trntensor kernel (forced through
``set_backend("nki")`` with ``TRNTENSOR_REQUIRE_NKI`` semantics, so a kernel
failure raises instead of quietly falling back to torch) at sizes of 1, 127,
129, 200, 1000 and 4097, odd K and N, and free dims that are not multiples
of 512, and compare with fp64 torch.

``CASES`` is also imported by the standalone probe script, which prints one
``OK`` / ``SILENT`` / ``LOUD`` line per case.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from trntensor.nki import HAS_NKI
from trntensor.nki.dispatch import _get_dispatch_path, _use_simulator


def _native() -> bool:
    if not HAS_NKI or _use_simulator():
        return False
    try:
        return _get_dispatch_path() == "native"
    except Exception:
        return False


# Normalised max error |out - ref|_max / |ref|_max. A dropped tail tile is
# an O(1) error; fp32 accumulation over K ≤ 4097 stays far below these.
TOL_FP32 = 2e-4
TOL_ENERGY = 1e-4


@contextlib.contextmanager
def forced_nki():
    """``set_backend("nki")``, every size threshold at 0, kernel errors raise."""
    import trntensor
    from trntensor.nki import dispatch

    saved = {
        name: getattr(dispatch, name)
        for name in ("_MIN_NKI_FLOPS_NATIVE", "_MIN_NKI_FLOPS_NATIVE_PINNED", "_REQUIRE_NKI")
    }
    prev = trntensor.get_backend()
    dispatch._MIN_NKI_FLOPS_NATIVE = 0
    dispatch._MIN_NKI_FLOPS_NATIVE_PINNED = 0
    dispatch._REQUIRE_NKI = True
    trntensor.set_backend("nki")
    try:
        yield dispatch
    finally:
        trntensor.set_backend(prev)
        for name, value in saved.items():
            setattr(dispatch, name, value)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _norm_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    out = out.detach().cpu().double()
    ref = ref.double()
    if out.shape != ref.shape:
        raise AssertionError(f"shape {tuple(out.shape)} != {tuple(ref.shape)}")
    if not torch.isfinite(out).all():
        return float("inf")
    scale = max(float(ref.abs().max()), 1e-30) if ref.numel() else 1.0
    return float((out - ref).abs().max()) / scale if ref.numel() else 0.0


# ── case builders: each returns (out, ref, tol) ─────────────────────────────


def _matmul(M, K, N, dtype=torch.float32):
    def run(d):
        g = _gen(M * 7 + K * 3 + N)
        A, B = torch.randn(M, K, generator=g), torch.randn(K, N, generator=g)
        out = d.nki_matmul(A.to(dtype), B.to(dtype))
        tol = TOL_FP32 if dtype == torch.float32 else 2e-2
        return out.float(), A.to(dtype).double() @ B.to(dtype).double(), tol

    return run


def _bmm(Bsz, M, K, N):
    def run(d):
        g = _gen(Bsz + M * 7 + K * 3 + N)
        A, B = torch.randn(Bsz, M, K, generator=g), torch.randn(Bsz, K, N, generator=g)
        return d.nki_batched_matmul(A, B), torch.bmm(A.double(), B.double()), TOL_FP32

    return run


def _einsum(subs, *shapes):
    def run(d):
        import trntensor

        g = _gen(sum(sum(s) for s in shapes))
        ops = [torch.randn(*s, generator=g) for s in shapes]
        out = trntensor.einsum(subs, *ops)
        return out, torch.einsum(subs, *(o.double() for o in ops)), TOL_FP32

    return run


def _mp2_inputs(nocc, nvir, naux):
    g = _gen(nocc * 13 + nvir * 5 + naux)
    B = torch.randn(nocc, nvir, naux, generator=g) * (0.5 / max(naux, 1) ** 0.5)
    eo = -torch.rand(nocc, generator=g) - 1.0
    ev = torch.rand(nvir, generator=g) + 0.5
    return B, eo, ev


def _mp2_ref(B, eo, ev):
    from trntensor.quantum import _cpu_mp2_energy

    return _cpu_mp2_energy(B.double(), eo.double(), ev.double()).reshape(1)


def _mp2_rows(nocc, nvir, naux):
    def run(d):
        B, eo, ev = _mp2_inputs(nocc, nvir, naux)
        return d._nki_mp2_energy_rows(B, eo, ev).reshape(1), _mp2_ref(B, eo, ev), TOL_ENERGY

    return run


def _mp2_legacy(nocc, nvir, naux):
    def run(d):
        B, eo, ev = _mp2_inputs(nocc, nvir, naux)
        return d._nki_mp2_energy(B, eo, ev).reshape(1), _mp2_ref(B, eo, ev), TOL_ENERGY

    return run


def _mp2_public(nocc, nvir, naux):
    def run(d):
        import trntensor

        B, eo, ev = _mp2_inputs(nocc, nvir, naux)
        return trntensor.mp2_energy(B, eo, ev).reshape(1), _mp2_ref(B, eo, ev), TOL_ENERGY

    return run


def _ao_inputs(nbasis, nocc, nvir, naux, sym=False):
    g = _gen(nbasis * 11 + nocc * 7 + nvir * 3 + naux)
    eri = torch.randn(nbasis, nbasis, naux, generator=g) * 0.1
    if sym:
        eri = 0.5 * (eri + eri.transpose(0, 1))
    C_occ = torch.randn(nbasis, nocc, generator=g)
    C_vir = torch.randn(nbasis, nvir, generator=g)
    ref = torch.einsum("mi,na,mnP->iaP", C_occ.double(), C_vir.double(), eri.double())
    return eri, C_occ, C_vir, ref


def _ao_legacy(nbasis, nocc, nvir, naux):
    def run(d):
        eri, co, cv, ref = _ao_inputs(nbasis, nocc, nvir, naux)
        return d._nki_ao_to_mo_transform(eri, co, cv), ref, TOL_FP32

    return run


def _ao_trn2(nbasis, nocc, nvir, naux):
    def run(d):
        eri, co, cv, ref = _ao_inputs(nbasis, nocc, nvir, naux)
        return d._nki_ao_to_mo_trn2(eri, co, cv), ref, TOL_FP32

    return run


def _ao_trn2_packed(nbasis, nocc, nvir, naux):
    def run(d):
        import trntensor

        eri, co, cv, ref = _ao_inputs(nbasis, nocc, nvir, naux, sym=True)
        out = d._nki_ao_to_mo_trn2_packed(trntensor.pack_tril(eri), co, cv)
        return out, ref, TOL_FP32

    return run


def _ao_public(nbasis, nocc, nvir, naux):
    def run(d):
        import trntensor

        eri, co, cv, ref = _ao_inputs(nbasis, nocc, nvir, naux, sym=True)
        return trntensor.ao_to_mo_transform(eri, co, cv), ref, TOL_FP32

    return run


def _shape(*dims) -> str:
    return "x".join(str(x) for x in dims)


CASES: list[tuple[str, str, object]] = []


def _add(kernel, builder, *dims, label=None):
    CASES.append((kernel, label or _shape(*dims), builder(*dims)))


# matmul_kernel (M, K, N): tails in each dim, N on both sides of 512.
for dims in [
    (1, 1, 1),
    (127, 129, 200),
    (129, 127, 513),
    (200, 1000, 129),
    (1000, 200, 1025),
    (4097, 129, 127),
    (129, 4097, 1),
    (1, 4097, 1000),
]:
    _add("matmul_kernel", _matmul, *dims)
CASES.append(("matmul_kernel[bf16]", "129x200x513", _matmul(129, 200, 513, torch.bfloat16)))
# batched_matmul_kernel (B, M, K, N).
for dims in [(1, 1, 1, 1), (3, 129, 127, 200), (2, 200, 129, 1000), (5, 127, 1000, 513)]:
    _add("batched_matmul_kernel", _bmm, *dims)
# Public einsum routes into the same kernels (incl. pre-transposed operands).
CASES.append(("einsum[ij,jk]", "1000x129x200", _einsum("ij,jk->ik", (1000, 129), (129, 200))))
CASES.append(("einsum[ik,jk]", "129x127x513", _einsum("ik,jk->ij", (129, 127), (513, 127))))
CASES.append(("einsum[ki,kj]", "200x4097x129", _einsum("ki,kj->ij", (4097, 200), (4097, 129))))
CASES.append(
    ("einsum[bij,bjk]", "3x129x127x200", _einsum("bij,bjk->bik", (3, 129, 127), (3, 127, 200)))
)
# mp2_energy_row_kernel (nocc, nvir, naux): odd/even nocc, tails in nvir/naux.
for dims in [
    (1, 1, 1),
    (3, 127, 129),
    (4, 129, 200),
    (7, 200, 127),
    (2, 1000, 129),
    (3, 129, 4097),
]:
    _add("mp2_energy_row_kernel", _mp2_rows, *dims)
# mp2_energy_kernel (original single-launch kernel; unrolls nocc², keep small).
for dims in [(1, 1, 1), (3, 129, 200), (2, 200, 129), (2, 127, 1000)]:
    _add("mp2_energy_kernel", _mp2_legacy, *dims)
_add("mp2_energy[public]", _mp2_public, 5, 129, 200)
# ao_to_mo_transform_kernel (nbasis, nocc, nvir, naux): unrolls naux, keep it small.
for dims in [(1, 1, 1, 1), (129, 7, 513, 3), (200, 127, 129, 2), (127, 1, 1000, 1)]:
    _add("ao_to_mo_transform_kernel", _ao_legacy, *dims)
# ao_to_mo_trn2_kernel and the packed variant.
for dims in [(1, 1, 1, 1), (129, 7, 129, 129), (200, 127, 200, 1000), (127, 1, 1000, 4097)]:
    _add("ao_to_mo_trn2_kernel", _ao_trn2, *dims)
for dims in [(1, 1, 1, 1), (129, 7, 129, 129), (200, 127, 200, 1000), (127, 3, 513, 4097)]:
    _add("ao_to_mo_trn2_packed_kernel", _ao_trn2_packed, *dims)
_add("ao_to_mo_transform[public]", _ao_public, 200, 9, 129, 200)


def run_case(fn) -> tuple[str, float]:
    """``("OK" | "SILENT", err)``; exceptions propagate (the caller reports LOUD)."""
    with forced_nki() as d:
        out, ref, tol = fn(d)
    err = _norm_err(out, ref)
    return ("OK" if err <= tol else "SILENT"), err


@pytest.mark.neuron
@pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path")
@pytest.mark.parametrize("kernel,shape,fn", CASES, ids=[f"{k}-{s}" for k, s, _ in CASES])
def test_tail_tiles_match_fp64(kernel, shape, fn):
    verdict, err = run_case(fn)
    assert verdict == "OK", f"{kernel} {shape}: normalised max error {err:.3e}"
