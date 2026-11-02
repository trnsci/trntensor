"""Test configuration — custom markers for hardware and simulator paths."""

import atexit
import os
import shutil
import tempfile

import pytest

# The Neuron Runtime reads NEURON_RT_STOCHASTIC_ROUNDING_EN once, at init.
# Capture it at conftest import — before any test touches a Neuron device —
# so the SR fixture can tell whether this process really runs with SR (#41).
SR_ENV_VAR = "NEURON_RT_STOCHASTIC_ROUNDING_EN"
_SR_AT_PROCESS_START = os.environ.get(SR_ENV_VAR)

# NKI trace cache (aws-neuron/aws-neuron-sdk#1454, trnsci/trnsci#35).
# `import torch_neuronx` turns the cache on (NKI_ENABLE_TRACE_CACHE=1), so
# that is how users run these kernels, and the neuron tests run the same way.
# The cache lives on disk and is versioned by wheel .dist-info, which an
# editable install or source checkout does not change, so a shared cache could
# serve a trace of an older kernel. Each session therefore gets a fresh,
# empty cache directory (NKI_TRACE_CACHE_URL), removed at exit.
# tests/test_trace_cache_safety.py checks that no kernel depends on a value
# the cache key misses (closure or global). Overrides: NKI_ENABLE_TRACE_CACHE=0
# runs with the cache off; an explicit NKI_TRACE_CACHE_URL is kept as is.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "1")
if os.environ["NKI_ENABLE_TRACE_CACHE"] != "0" and "NKI_TRACE_CACHE_URL" not in os.environ:
    _trace_cache_dir = tempfile.mkdtemp(prefix="trntensor-nki-trace-cache-")
    os.environ["NKI_TRACE_CACHE_URL"] = _trace_cache_dir
    atexit.register(shutil.rmtree, _trace_cache_dir, ignore_errors=True)


def pytest_addoption(parser):
    # Reference-conformance suite (tests/test_conformance.py, trnsci/trnsci#105).
    # The CPU CI job runs it on the pytorch backend with host inputs; a Trn2 run
    # passes --backend nki|auto and --conformance-device neuron to execute the
    # same cases against the same independent references.
    group = parser.getgroup("trntensor conformance")
    group.addoption(
        "--backend",
        default="pytorch",
        choices=("pytorch", "nki", "auto"),
        help="trntensor backend for tests/test_conformance.py (default: pytorch)",
    )
    group.addoption(
        "--conformance-device",
        default="cpu",
        help="device for conformance-test inputs, e.g. neuron (default: cpu)",
    )


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
