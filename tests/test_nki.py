"""Backend dispatch tests (CPU path)."""

import pytest
import torch

import trntensor
from trntensor.nki.dispatch import HAS_NKI


def test_set_backend_auto():
    trntensor.set_backend("auto")
    assert trntensor.get_backend() == "auto"


def test_set_backend_pytorch():
    trntensor.set_backend("pytorch")
    assert trntensor.get_backend() == "pytorch"
    trntensor.set_backend("auto")


def test_set_backend_invalid():
    with pytest.raises(AssertionError):
        trntensor.set_backend("tpu")


@pytest.mark.skipif(HAS_NKI, reason="Test asserts the no-NKI guard")
def test_set_backend_nki_raises_without_neuronxcc():
    with pytest.raises(RuntimeError, match="neuronxcc"):
        trntensor.set_backend("nki")
    # State should not have changed
    assert trntensor.get_backend() != "nki"


def test_from_xla_cpu_tensor_is_noop():
    t = torch.randn(3, 4)
    assert trntensor.from_xla(t) is t


@pytest.mark.skipif(HAS_NKI, reason="to_xla without NKI raises; hardware has NKI")
def test_to_xla_without_nki_raises():
    t = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="NKI runtime"):
        trntensor.to_xla(t)
