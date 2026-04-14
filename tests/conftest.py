"""Test configuration — custom markers for hardware and simulator paths."""

import pytest  # noqa: F401  (pytest imports this module by name)


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")
    config.addinivalue_line(
        "markers",
        "nki_simulator: runs NKI kernels via nki.simulate on CPU "
        "(requires TRNTENSOR_USE_SIMULATOR=1 + nki>=0.3.0)",
    )
