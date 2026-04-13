"""Backend dispatch tests (CPU path)."""

import pytest

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
