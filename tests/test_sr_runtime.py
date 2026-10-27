"""SR runtime guard (#41) — CPU tests with the native NKI path stubbed.

On NKI 0.7+ stochastic rounding is a process-wide Neuron Runtime setting
(``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``), so ``use_sr=True`` alone is a
no-op on hardware. The dispatcher warns once when SR is requested on the
native path without the variable. These tests replace the kernels and the
XLA transfer with ``torch`` stand-ins so the guard runs without hardware.
"""

import warnings

import pytest
import torch

import trntensor
from trntensor.nki import _kernels, dispatch


@pytest.fixture
def native_path(monkeypatch):
    """Route nki_matmul / nki_batched_matmul through stubbed native kernels."""
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
    monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS", 0)
    monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS_PINNED", 0)
    monkeypatch.setattr(dispatch, "_to_xla", lambda *ts: (ts, ts[0].device))
    monkeypatch.setattr(_kernels, "matmul_kernel", lambda a, b, use_sr=False: a @ b, raising=False)
    monkeypatch.setattr(
        _kernels,
        "batched_matmul_kernel",
        lambda a, b, use_sr=False: torch.bmm(a, b),
        raising=False,
    )
    monkeypatch.setattr(dispatch, "_sr_warning_emitted", False)
    monkeypatch.delenv(dispatch.SR_ENV_VAR, raising=False)
    trntensor.clear_plan_cache()
    yield
    trntensor.clear_plan_cache()


def _sr_warnings(record):
    return [w for w in record if issubclass(w.category, RuntimeWarning)]


def _mats(b=None):
    torch.manual_seed(0)
    if b is None:
        return torch.randn(128, 128).bfloat16(), torch.randn(128, 256).bfloat16()
    return torch.randn(b, 128, 128).bfloat16(), torch.randn(b, 128, 256).bfloat16()


class TestSrRuntimeEnabled:
    @pytest.mark.parametrize("value", [None, "0", "", "true", "yes"])
    def test_disabled_values(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv(dispatch.SR_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(dispatch.SR_ENV_VAR, value)
        assert not dispatch.sr_runtime_enabled()

    def test_enabled(self, monkeypatch):
        monkeypatch.setenv(dispatch.SR_ENV_VAR, "1")
        assert dispatch.sr_runtime_enabled()


class TestSrGuard:
    def test_matmul_warns_once_when_unset(self, native_path):
        A, B = _mats()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out1 = dispatch.nki_matmul(A, B, use_sr=True)
            out2 = dispatch.nki_matmul(A, B, use_sr=True)
        msgs = _sr_warnings(rec)
        assert len(msgs) == 1
        assert dispatch.SR_ENV_VAR + "=1" in str(msgs[0].message)
        assert "round-to-nearest" in str(msgs[0].message)
        # Still computes.
        torch.testing.assert_close(out1, A @ B)
        torch.testing.assert_close(out2, A @ B)

    def test_warns_when_set_to_zero(self, native_path, monkeypatch):
        monkeypatch.setenv(dispatch.SR_ENV_VAR, "0")
        A, B = _mats()
        with pytest.warns(RuntimeWarning, match="NEURON_RT_STOCHASTIC_ROUNDING_EN"):
            dispatch.nki_matmul(A, B, use_sr=True)

    def test_batched_matmul_warns(self, native_path):
        A, B = _mats(b=3)
        with pytest.warns(RuntimeWarning, match="NEURON_RT_STOCHASTIC_ROUNDING_EN"):
            out = dispatch.nki_batched_matmul(A, B, use_sr=True)
        torch.testing.assert_close(out, torch.bmm(A, B))

    def test_no_warning_when_enabled(self, native_path, monkeypatch):
        monkeypatch.setenv(dispatch.SR_ENV_VAR, "1")
        A, B = _mats()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            dispatch.nki_matmul(A, B, use_sr=True)
            dispatch.nki_batched_matmul(*_mats(b=2), use_sr=True)

    def test_no_warning_without_sr(self, native_path):
        A, B = _mats()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            dispatch.nki_matmul(A, B)
            dispatch.nki_batched_matmul(*_mats(b=2))

    def test_no_warning_on_cpu_fallback(self, monkeypatch):
        """Without NKI the CPU path emulates SR in software — no warning."""
        monkeypatch.setattr(dispatch, "_use_nki", lambda: False)
        monkeypatch.setattr(dispatch, "_sr_warning_emitted", False)
        monkeypatch.delenv(dispatch.SR_ENV_VAR, raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            dispatch.nki_matmul(*_mats(), use_sr=True)

    def test_no_warning_in_simulator(self, native_path, monkeypatch):
        """nki.simulate runs on CPU; the runtime variable is irrelevant there."""
        import types

        monkeypatch.setattr(dispatch, "_use_simulator", lambda: True)
        fake_nki = types.SimpleNamespace(simulate=lambda k: lambda a, b, use_sr=False: a @ b)
        monkeypatch.setattr(dispatch, "nki", fake_nki, raising=False)
        A, B = (t.half() for t in _mats())
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            dispatch.nki_matmul(A, B, use_sr=True)
        assert not dispatch._sr_warning_emitted


class TestSrGuardPublicApi:
    def test_einsum_precision_sr(self, native_path):
        A, B = _mats()
        with pytest.warns(RuntimeWarning, match="NEURON_RT_STOCHASTIC_ROUNDING_EN"):
            out = trntensor.einsum("ij,jk->ik", A, B, precision="sr")
        torch.testing.assert_close(out, A @ B)

    def test_einsum_bmm_precision_sr(self, native_path):
        A, B = _mats(b=2)
        with pytest.warns(RuntimeWarning, match="NEURON_RT_STOCHASTIC_ROUNDING_EN"):
            trntensor.einsum("bij,bjk->bik", A, B, precision="sr")

    def test_multi_einsum_precision_sr(self, native_path):
        A, B = _mats()
        with pytest.warns(RuntimeWarning, match="NEURON_RT_STOCHASTIC_ROUNDING_EN"):
            trntensor.multi_einsum(("ij,jk->ik", A, B), ("ij,jk->ik", A, B), precision="sr")

    def test_target_forward_error_selecting_sr(self, native_path):
        A, B = _mats()
        K = A.shape[1]
        u = 2.0**-8
        # Between √K·u (sr) and K·u (fast): selector picks "sr".
        target = 2 * K**0.5 * u
        assert trntensor.select_precision_for_error(torch.bfloat16, K, target) == "sr"
        with pytest.warns(RuntimeWarning, match="NEURON_RT_STOCHASTIC_ROUNDING_EN"):
            trntensor.einsum("ij,jk->ik", A, B, target_forward_error=target)

    def test_einsum_fast_does_not_warn(self, native_path):
        A, B = _mats()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            trntensor.einsum("ij,jk->ik", A, B)
