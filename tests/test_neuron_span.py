"""Row-block matmul for the >= 4 GiB partition-span bug on the neuron device (#123).

CPU tests: the thresholds are lowered so that small host tensors stand in for
the 32 MiB-row operands, and `_on_neuron_device` is patched so the host tensors
take the device logic. A spy on `torch.matmul` checks that no product the
blocking issues still has an unsafe operand or result. The on-device
regression tests at the real sizes are in test_native_device.py
(`TestSpan4GiB`).
"""

import pytest
import torch

import trntensor
from trntensor import quantum
from trntensor.nki import dispatch as D

SPAN = 1 << 14  # stands in for 2**32
BLOCK = 1 << 13  # stands in for 2**31


@pytest.fixture
def lowered(monkeypatch):
    monkeypatch.setattr(D, "_NEURON_SPAN_BYTES", SPAN)
    monkeypatch.setattr(D, "_NEURON_BLOCK_BYTES", BLOCK)
    monkeypatch.setattr(D, "_on_neuron_device", lambda t: True)
    calls = []
    real = torch.matmul

    def spy(a, b, *args, **kwargs):
        calls.append((tuple(a.shape), tuple(b.shape)))
        if a.dim() == 2 and b.dim() == 2:
            assert not D._span_block(a), f"unsafe lhs {tuple(a.shape)}"
            assert not D._span_block(b), f"unsafe rhs {tuple(b.shape)}"
            assert not D._out_block(a.shape[0], b.shape[1], a.element_size()), "unsafe result"
        return real(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", spy)
    return calls


def _r(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64).float()


def _ref(a, b):
    return (a.double() @ b.double()).float()


def test_real_thresholds():
    assert D._NEURON_SPAN_BYTES == 1 << 32
    assert D._out_block(128, 8_389_632, 4) == 64
    assert D._out_block(128, 8_388_480, 4) == 0
    assert D._out_block(64, 8_389_632, 4) == 0


@pytest.mark.parametrize("M,K,N", [(12, 128, 40), (12, 300, 33), (128, 40, 6), (130, 128, 40)])
def test_blocked_matches_plain(lowered, M, K, N):
    a, b = _r(M, K), _r(K, N, seed=1)
    assert D._neuron_mm_unsafe(a, b)
    torch.testing.assert_close(D._neuron_matmul(a, b), _ref(a, b), rtol=1e-5, atol=1e-4)
    assert len(lowered) > 1


def test_batched_and_bmm(lowered):
    a, b = _r(3, 12, 128), _r(3, 128, 40, seed=1)
    torch.testing.assert_close(D._neuron_bmm(a, b), _ref(a, b), rtol=1e-5, atol=1e-4)
    assert len(lowered) > 1


def test_short_row_batches_stay_one_call(monkeypatch):
    monkeypatch.setattr(D, "_NEURON_SPAN_BYTES", SPAN)
    monkeypatch.setattr(D, "_on_neuron_device", lambda t: True)
    # (nocc, nbasis, naux) halves with short rows: no blocking even when large.
    a, b = _r(4, 6, 32), _r(4, 32, 8, seed=1)
    assert not D._neuron_mm_unsafe(a, b)


def test_safe_and_host_products_untouched(monkeypatch):
    a, b = _r(12, 128), _r(128, 40, seed=1)
    assert torch.equal(D._neuron_matmul(a, b), torch.matmul(a, b))  # host
    monkeypatch.setattr(D, "_on_neuron_device", lambda t: True)
    calls = []
    real = torch.matmul
    monkeypatch.setattr(torch, "matmul", lambda x, y: calls.append((x, y)) or real(x, y))
    out = D._neuron_matmul(a, b)  # below the real 4 GiB rule
    assert len(calls) == 1 and calls[0][0] is a and calls[0][1] is b
    assert torch.equal(out, real(a, b))


@pytest.mark.parametrize("backend", ["auto", "pytorch"])
def test_einsum_matmul_and_bmm_routes(lowered, backend):
    old = trntensor.get_backend()
    trntensor.set_backend(backend)
    try:
        a, b = _r(12, 128), _r(128, 40, seed=1)
        out = trntensor.einsum("ij,jk->ik", a, b)
        torch.testing.assert_close(out, _ref(a, b), rtol=1e-5, atol=1e-4)
        assert len(lowered) > 1
        lowered.clear()
        a3, b3 = _r(2, 12, 128), _r(2, 128, 40, seed=1)
        out = trntensor.einsum("bij,bjk->bik", a3, b3)
        torch.testing.assert_close(out, _ref(a3, b3), rtol=1e-5, atol=1e-4)
    finally:
        trntensor.set_backend(old)


def test_ao_to_mo_contiguous_route_blocks(lowered, monkeypatch):
    # nbasis 128, nbasis * naux * 4 B = 2 KiB rows: the #123 layout at this scale.
    nb, naux, nocc, nvir = 128, 4, 5, 7
    eri = _r(nb, nb, naux)
    C_occ, C_vir = _r(nb, nocc, seed=1), _r(nb, nvir, seed=2)
    ref = torch.einsum(
        "na,inP->iaP", C_vir.double(), torch.einsum("mi,mnP->inP", C_occ.double(), eri.double())
    )
    out = quantum._ao_to_mo_contiguous_lhs(eri, C_occ, C_vir)
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-5)
    assert len(lowered) > 2  # the half-transform was blocked over K
