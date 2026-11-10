"""On-device regression tests for trnsci/trnsci#135 (aws-neuron/aws-neuron-sdk#1466).

On Trn2 (SDK 2.33), every device op that materializes a transposed int32 /
int64 layout returns int32(float32(x)) (per 32-bit half for int64), and
integer matmul runs in fp32. Integer einsum on the device therefore
contracts on the host, and ``out=`` writes go through bit-exact views.
"""

import pytest
import torch

import trntensor
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
BIG = 2**24 + 1


def _rand_int(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo = torch.randint(-(2**31), 2**31 - 1, (*shape, 2), generator=g, dtype=torch.int32)
    return lo[..., 0].contiguous() if dtype == torch.int32 else lo.view(torch.int64).reshape(shape)


@pytest.mark.parametrize("subscripts", ["ij,jk->ik", "ij,jk->ki", "ji,jk->ik", "ij,jk,kl->il"])
@pytest.mark.parametrize("precision", ["fast", "kahan"])
def test_int32_contraction_exact_on_device(subscripts, precision):
    a = torch.tensor([[BIG, 2], [3, 4]], dtype=torch.int32)
    eye = torch.eye(2, dtype=torch.int32)
    ops = (a, eye, eye)[: subscripts.count(",") + 1]
    got = trntensor.einsum(subscripts, *(o.to(DEV) for o in ops), precision=precision)
    assert got.device.type == DEV
    assert torch.equal(got.cpu(), torch.einsum(subscripts, *ops))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(64, 96), (256, 384), (1000, 600)])
@pytest.mark.parametrize("out_layout", ["contiguous", "transposed"])
def test_transpose_out_exact_on_device(dtype, shape, out_layout):
    x = _rand_int(shape, dtype)
    if out_layout == "contiguous":
        out = torch.empty(shape[::-1], dtype=dtype, device=DEV)
    else:
        out = torch.empty(shape, dtype=dtype, device=DEV).t()
    trntensor.einsum("ij->ji", x.to(DEV), out=out)
    assert torch.equal(out.cpu(), x.t())


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_transpose_contiguous_result_on_device(dtype):
    x = _rand_int((256, 384), dtype, seed=1)
    got = trntensor.einsum("ij->ji", x.to(DEV))
    assert got.is_contiguous()
    assert torch.equal(got.cpu(), x.t())
