"""On-device regression tests for #123 (trnblas#112, aws-neuron-sdk#1457) at real sizes.

A device matmul is silently wrong once an operand's (or the result's) 128 DMA
partitions span 4 GiB or more: fp32 rows of 32 MiB or more. Operands are
uploaded above a same-size decoy holding different data, or produced on the
device by `clone`, and every result is compared with a host reference. Each
case needs about 9-14 GB of one logical core's device memory.
"""

import gc

import pytest
import torch

import trntensor
from trntensor import quantum
from trntensor.nki import HAS_NKI
from trntensor.nki import dispatch as D


def _native() -> bool:
    if not HAS_NKI or D._use_simulator():
        return False
    try:
        return D._get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

DEV = "neuron"
N_BAD = 8_389_632  # 32 MiB + 4 KiB fp32 rows
N_OK = 8_388_480  # 32 MiB - 512 B


def _host(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _rel(out, ref):
    out, ref = out.cpu().double(), ref.double()
    return ((out - ref).norm() / ref.norm()).item()


@pytest.fixture(autouse=True)
def _free():
    yield
    gc.collect()


@pytest.fixture(params=["upload", "device"])
def source(request):
    return request.param


def _device_operand(X, source):
    decoy = (-X).to(DEV)  # different data directly below
    return decoy, (X.to(DEV) if source == "upload" else X.to(DEV).clone())


class TestSpan4GiB:
    def test_einsum_matmul(self, source):
        A, B = _host(96, 128), _host(128, N_BAD, seed=1)
        decoy, Bd = _device_operand(B, source)
        out = trntensor.einsum("ij,jk->ik", A.to(DEV), Bd)
        assert _rel(out, A @ B) < 1e-5

    def test_einsum_bmm(self, source):
        A, B = _host(1, 96, 128), _host(1, 128, N_BAD, seed=1)
        decoy, Bd = _device_operand(B, source)
        out = trntensor.einsum("bij,bjk->bik", A.to(DEV), Bd)
        assert _rel(out, A @ B) < 1e-5

    def test_ao_to_mo_contiguous_route(self, source):
        # nbasis * naux = 8,389,632: the #98 contiguous route's B rows are 32 MiB + 4 KiB.
        nb, naux, nocc, nvir = 128, 65_544, 8, 16
        eri = _host(nb, nb, naux)
        C_occ, C_vir = _host(nb, nocc, seed=1), _host(nb, nvir, seed=2)
        decoy, eri_d = _device_operand(eri, source)
        half = (C_occ.T @ eri.reshape(nb, nb * naux)).reshape(nocc, nb, naux)
        ref = torch.matmul(C_vir.T, half)
        del half
        out = quantum._cpu_ao_to_mo_transform(eri_d, C_occ.to(DEV), C_vir.to(DEV))
        assert out.shape == (nocc, nvir, naux)
        assert _rel(out, ref) < 1e-5

    def test_below_rule_is_the_plain_call(self):
        A, B = _host(96, 128), _host(128, N_OK, seed=1)
        decoy, Bd = _device_operand(B, "upload")
        Ad = A.to(DEV)
        assert not D._neuron_mm_unsafe(Ad, Bd)
        out = trntensor.einsum("ij,jk->ik", Ad, Bd)
        assert torch.equal(out.cpu(), torch.matmul(Ad, Bd).cpu())
        assert _rel(out, A @ B) < 1e-5
