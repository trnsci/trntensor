"""Test configuration — custom markers for hardware and simulator paths."""

import os

import pytest

# The Neuron Runtime reads NEURON_RT_STOCHASTIC_ROUNDING_EN once, at init.
# Capture it at conftest import — before any test touches a Neuron device —
# so the SR fixture can tell whether this process really runs with SR (#41).
SR_ENV_VAR = "NEURON_RT_STOCHASTIC_ROUNDING_EN"
_SR_AT_PROCESS_START = os.environ.get(SR_ENV_VAR)


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")
    config.addinivalue_line(
        "markers",
        "nki_simulator: runs NKI kernels via nki.simulate on CPU "
        "(requires TRNTENSOR_USE_SIMULATOR=1 + nki>=0.3.0)",
    )
    config.addinivalue_line(
        "markers",
        "neuron_sr: stochastic-rounding hardware test; requires "
        f"{SR_ENV_VAR}=1 at process start (skipped otherwise)",
    )


@pytest.fixture
def sr_runtime(monkeypatch):
    """Require a process launched with ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``.

    Setting the variable from inside the test process is too late — the
    runtime has already (or will have) latched its rounding mode — so skip
    unless it was set at process start. Pins it in ``os.environ`` for the
    test so trntensor's SR guard sees the same value the runtime did.
    """
    if _SR_AT_PROCESS_START != "1":
        pytest.skip(
            f"SR hardware tests need {SR_ENV_VAR}=1 set before pytest starts "
            "(Neuron Runtime reads it once at init)"
        )
    monkeypatch.setenv(SR_ENV_VAR, "1")
    yield
