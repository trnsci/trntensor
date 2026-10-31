"""
Host results through the caching host allocator (trnblas#61).

CPU-only: the pinned allocation is replaced by a recording stand-in (a CPU-only
host has no accelerator to pin for), and the dispatch path is forced to
"native" where a test needs the cached-allocator route.
"""

import pytest
import torch

from trntensor.nki import dispatch as D

MiB = 1 << 20


@pytest.fixture
def native_host_cache(monkeypatch):
    """Pretend we are on the native path; record pinned allocations."""
    monkeypatch.delenv("TRNTENSOR_PINNED_OUTPUT", raising=False)
    monkeypatch.setattr(D, "_get_dispatch_path", lambda: "native")
    monkeypatch.setattr(D, "_use_simulator", lambda: False)
    calls = []
    real_empty = torch.empty

    def fake_empty(*args, pin_memory=False, **kw):
        if pin_memory:
            calls.append(args[0])
        return real_empty(*args, **kw)

    monkeypatch.setattr(D.torch, "empty", fake_empty)
    return calls


class TestUseHostCache:
    def test_off_the_native_path(self, monkeypatch):
        monkeypatch.setattr(D, "_get_dispatch_path", lambda: "xla")
        monkeypatch.setattr(D, "_use_simulator", lambda: False)
        assert not D._use_host_cache("cpu", 64 * MiB)

    def test_native_large_cpu_destination(self, native_host_cache):
        assert D._use_host_cache("cpu", MiB)
        assert D._use_host_cache(torch.device("cpu"), 64 * MiB)

    def test_small_results_skip(self, native_host_cache):
        assert not D._use_host_cache("cpu", MiB - 1)

    def test_device_destination_skips(self, native_host_cache):
        assert not D._use_host_cache("meta", 64 * MiB)

    @pytest.mark.parametrize("value", ["0", "false", "off", "no"])
    def test_env_disables(self, native_host_cache, monkeypatch, value):
        monkeypatch.setenv("TRNTENSOR_PINNED_OUTPUT", value)
        assert not D._use_host_cache("cpu", 64 * MiB)

    def test_simulator_skips(self, native_host_cache, monkeypatch):
        monkeypatch.setattr(D, "_use_simulator", lambda: True)
        assert not D._use_host_cache("cpu", 64 * MiB)


class TestToHost:
    def test_large_result_uses_cache_and_is_fresh(self, native_host_cache):
        src = torch.randn(512, 1024)  # 2 MiB, standing in for a device result
        out = D._host_empty(src.shape, src.dtype, "cpu").copy_(src)
        assert native_host_cache == [(512, 1024)]
        assert torch.equal(out, src)
        assert out.data_ptr() != src.data_ptr()

    def test_cpu_tensor_passes_through(self, native_host_cache):
        t = torch.randn(512, 1024)
        assert D._to_host(t, "cpu") is t
        assert D._from_device(t) is t
        assert native_host_cache == []

    def test_host_empty_small_is_plain(self, native_host_cache):
        t = D._host_empty((4, 4), torch.float32, "cpu")
        assert t.shape == (4, 4) and native_host_cache == []

    def test_aten_on_device_downloads_through_to_host(self, monkeypatch):
        seen = []
        monkeypatch.setattr(D, "_NEURON_DEVICE", "cpu")
        monkeypatch.setattr(D, "_to_host", lambda t, device: seen.append(device) or t)
        A, B = torch.randn(8, 8), torch.randn(8, 8)
        torch.testing.assert_close(D._aten_on_device(torch.matmul, A, B), A @ B)
        assert seen == [torch.device("cpu")]


class TestScaleAddEpilogue:
    """einsum's alpha/beta epilogue runs in place on a fresh result, same bits."""

    @pytest.mark.parametrize("alpha", [1.0, 0.5, -2.0])
    @pytest.mark.parametrize("beta", [0.0, 1.0, 0.25])
    @pytest.mark.parametrize("with_out", [True, False])
    def test_matches_out_of_place(self, alpha, beta, with_out):
        import trntensor

        g = torch.Generator().manual_seed(0)
        A, B, C = (
            torch.randn(17, 23, generator=g),
            torch.randn(23, 9, generator=g),
            torch.randn(17, 9, generator=g),
        )
        out = C if with_out else None
        want = torch.einsum("ij,jk->ik", A, B).mul(alpha)
        if with_out:
            want = want.add(C, alpha=beta)
        C0 = C.clone()
        got = trntensor.einsum("ij,jk->ik", A, B, alpha=alpha, beta=beta, out=out)
        assert torch.equal(got, want)
        assert torch.equal(C, C0)  # out is read, never written

    def test_view_of_operand_is_not_overwritten(self):
        import trntensor

        A = torch.randn(8, 8)
        A0 = A.clone()
        got = trntensor.einsum("ij->ji", A, alpha=2.0)
        assert torch.equal(got, 2.0 * A0.T) and torch.equal(A, A0)

    def test_broadcast_out_and_promotion_fall_back(self):
        from trntensor.einsum import _scale_add

        r = torch.ones(4, 1)
        got = _scale_add(r, 1.0, 1.0, torch.ones(4, 3), ())
        assert got.shape == (4, 3) and r.shape == (4, 1)
        ri = torch.ones(3, dtype=torch.int64)
        assert _scale_add(ri, 0.5, 1.0, None, ()).dtype == torch.float32
        r64 = torch.ones(3, dtype=torch.float32)
        assert _scale_add(r64, 1.0, 1.0, torch.ones(3, dtype=torch.float64), ()).dtype == (
            torch.float64
        )

    def test_autograd_kept(self):
        import trntensor

        A = torch.randn(5, 6, requires_grad=True)
        B = torch.randn(6, 4)
        trntensor.einsum("ij,jk->ik", A, B, alpha=3.0).sum().backward()
        torch.testing.assert_close(A.grad, 3.0 * B.sum(1).expand(5, 6))
