"""
Dispatch-path selection (native TorchNeuron vs XLA vs CPU), lazy
registration of the ``trntensor::`` custom ops, launch routing, path-aware
FLOP thresholds, residency helpers, and the dormant XLA deprecation hook.

CPU-only: ``torch_neuronx`` and ``torch_xla`` are replaced with stand-ins via
``sys.modules`` / ``find_spec`` monkeypatching, so these run in the default
``pytest -m "not neuron"`` job on any host.
"""

import importlib.util
import subprocess
import sys
import types
import warnings

import pytest
import torch

import trntensor
import trntensor.nki
from trntensor.nki import _kernels, dispatch

_REAL_FIND_SPEC = importlib.util.find_spec

EXPECTED_OPS = {
    "matmul",
    "batched_matmul",
    "mp2_energy",
    "ao_to_mo_transform",
    "mp2_energy_row",
    "ao_to_mo_trn2",
}


@pytest.fixture(autouse=True)
def reset_dispatch_state(monkeypatch):
    """Each test sees an undetected path and an unregistered op table."""
    monkeypatch.setattr(dispatch, "_dispatch_path", None)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.setattr(dispatch, "_xla_deprecation_emitted", False)
    monkeypatch.setattr(dispatch, "_api_deprecations_emitted", set())
    monkeypatch.delenv("TRNSCI_NKI_DISPATCH", raising=False)
    monkeypatch.delenv("TRNSCI_XLA_DEPRECATION", raising=False)
    # Drop any `_native` left by an earlier test so registration re-runs.
    monkeypatch.delitem(sys.modules, "trntensor.nki._native", raising=False)
    monkeypatch.delattr(trntensor.nki, "_native", raising=False)
    trntensor.clear_plan_cache()
    yield
    trntensor.clear_plan_cache()


def _fake_native_torch_neuronx(device_count=1, recorded=None):
    """Stand-in for the TorchNeuron `torch_neuronx` (exports `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            if recorded is not None:
                recorded[name] = (f, mutates_args, fake_fn)
            return f

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: device_count
    return mod


def _fake_xla_torch_neuronx():
    """Stand-in for the XLA-lineage `torch_neuronx` (no `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")
    mod.trace = lambda *a, **k: None
    return mod


def _set_torch_xla_present(monkeypatch, present: bool):
    def find_spec(name, *args, **kwargs):
        if name == "torch_xla":
            return object() if present else None
        return _REAL_FIND_SPEC(name, *args, **kwargs)

    monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)


def _accept_neuron_device(monkeypatch):
    # Off-hardware, "neuron" is not a registered torch device type.
    real_device = torch.device

    def device(spec, *args, **kwargs):
        if spec == dispatch._NEURON_DEVICE:
            return real_device("cpu")
        return real_device(spec, *args, **kwargs)

    monkeypatch.setattr(dispatch.torch, "device", device)


def _no_xla(*_):
    raise AssertionError("_to_xla must not run on the native path")


@pytest.fixture
def native_host(monkeypatch):
    """The native path with CPU standing in for the `neuron` device.

    Kernels resolve to `_native_ops`, which each test fills with torch
    stand-ins; `_to_xla` raises so any XLA leak fails the test.
    """
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setattr(dispatch, "_DEVICE_TYPES", ("xla", "cpu"))
    for name in (
        "_MIN_NKI_FLOPS",
        "_MIN_NKI_FLOPS_PINNED",
        "_MIN_NKI_FLOPS_NATIVE",
        "_MIN_NKI_FLOPS_NATIVE_PINNED",
    ):
        monkeypatch.setattr(dispatch, name, 0)
    monkeypatch.setattr(dispatch, "_to_xla", _no_xla)

    def no_raw_kernel(*a, **k):
        raise AssertionError("raw kernel must not be called on the native path")

    for kname in (
        "matmul_kernel",
        "batched_matmul_kernel",
        "mp2_energy_kernel",
        "ao_to_mo_transform_kernel",
        "mp2_energy_row_kernel",
        "ao_to_mo_trn2_kernel",
    ):
        monkeypatch.setattr(_kernels, kname, no_raw_kernel, raising=False)
    # Force NKI: under "auto" the native path runs the fused quantum
    # kernels as aten (measured faster on Trn2), and these tests check the
    # kernel launch itself.
    monkeypatch.setattr(dispatch, "_backend", "nki")
    monkeypatch.setenv("TRNTENSOR_REQUIRE_NKI", "1")
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    calls: list[tuple[str, dict]] = []
    yield calls


def _install_ops(monkeypatch, calls, **impls):
    def wrap(name, fn):
        def op(*args, **kwargs):
            calls.append((name, kwargs))
            return fn(*args, **kwargs)

        return op

    monkeypatch.setattr(dispatch, "_native_ops", {n: wrap(n, f) for n, f in impls.items()})


class TestDispatchPathSelection:
    def test_native_when_nki_op_and_device(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "native"

    def test_native_probe_does_not_touch_torch_xla(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch_xla", None)

        def find_spec(name, *args, **kwargs):
            assert name != "torch_xla", "native probe must not look up torch_xla"
            return _REAL_FIND_SPEC(name, *args, **kwargs)

        monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)
        assert dispatch._get_dispatch_path() == "native"

    def test_xla_when_torch_neuronx_has_no_nki_op(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_xla_when_native_has_no_device(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(device_count=0)
        )
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_not_native_when_neuron_device_type_unregistered(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_not_native_when_device_count_raises(self, monkeypatch):
        mod = _fake_native_torch_neuronx()

        def boom():
            raise RuntimeError("no runtime")

        mod.device_count = boom
        monkeypatch.setitem(sys.modules, "torch_neuronx", mod)
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_cpu_when_nothing_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", None)  # import raises
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_detection_is_cached(self, monkeypatch):
        calls = []

        def detect():
            calls.append(1)
            return "xla"

        monkeypatch.setattr(dispatch, "_detect_dispatch_path", detect)
        assert dispatch._get_dispatch_path() == "xla"
        assert dispatch._get_dispatch_path() == "xla"
        assert len(calls) == 1


class TestDispatchOverride:
    """TRNSCI_NKI_DISPATCH=auto|native|xla (suite-wide) pins the path."""

    def _both_lineages(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)

    def test_auto_prefers_native(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "auto")
        assert dispatch._get_dispatch_path() == "native"

    def test_force_xla_when_both_available(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "XLA")
        assert dispatch._get_dispatch_path() == "xla"

    def test_force_native_without_backend_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "native")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=native"):
            dispatch._get_dispatch_path()

    def test_force_xla_without_torch_xla_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "xla")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=xla"):
            dispatch._get_dispatch_path()

    def test_unknown_value_raises(self, monkeypatch):
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "cuda")
        with pytest.raises(ValueError, match="auto, native, or xla"):
            dispatch._get_dispatch_path()


class TestLaunchRouting:
    def test_native_launch_uses_custom_op_not_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        seen = {}

        def op(a, b, use_sr=False):
            seen["use_sr"] = use_sr
            return a + b

        monkeypatch.setattr(dispatch, "_native_ops", {"fake": op})

        def no_kernel(*t, **k):
            raise AssertionError("raw kernel must not be called on the native path")

        monkeypatch.setattr(dispatch, "_to_xla", _no_xla)
        out = dispatch._launch(no_kernel, "fake", torch.ones(2, 1), torch.ones(2, 1), use_sr=True)
        assert torch.equal(out, torch.full((2, 1), 2.0))
        assert out.device.type == "cpu"
        assert seen["use_sr"] is True

    def test_xla_launch_calls_kernel_via_to_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        moved = []

        def fake_to_xla(*tensors):
            moved.append(len(tensors))
            return list(tensors), tensors[0].device

        monkeypatch.setattr(dispatch, "_to_xla", fake_to_xla)

        def no_ops():
            raise AssertionError("native ops must not load on the XLA path")

        monkeypatch.setattr(dispatch, "_get_native_ops", no_ops)
        out = dispatch._launch(lambda x, use_sr=False: x * 3, "unused", torch.ones(4, 1))
        assert moved == [1]
        assert torch.equal(out, torch.full((4, 1), 3.0))

    def test_cpu_launch_keeps_pre_native_behaviour(self, monkeypatch):
        # No device path: the launch goes through `_to_xla`, which raises
        # when torch_xla is absent — the same failure as before.
        monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", None)
        with pytest.raises(ImportError):
            dispatch._launch(lambda x: x, "unused", torch.ones(1, 1))


class TestPublicApiOnNativePath:
    """Every NKI entry point reaches its trntensor:: op through `_launch`."""

    def test_einsum_matmul_routes_to_matmul_op(self, native_host, monkeypatch):
        _install_ops(monkeypatch, native_host, matmul=lambda a, b, use_sr=False: a @ b)
        A, B = torch.randn(100, 70), torch.randn(70, 30)
        out = trntensor.einsum("ij,jk->ik", A, B)
        torch.testing.assert_close(out, A @ B)
        assert native_host == [("matmul", {"use_sr": False})]

    def test_transposed_matmul(self, native_host, monkeypatch):
        _install_ops(monkeypatch, native_host, matmul=lambda a, b, use_sr=False: a @ b)
        A, B = torch.randn(70, 100), torch.randn(30, 70)
        out = trntensor.einsum("ji,kj->ik", A, B)
        torch.testing.assert_close(out, A.T @ B.T)
        assert [c[0] for c in native_host] == ["matmul"]

    def test_einsum_bmm_routes_to_batched_matmul_op(self, native_host, monkeypatch):
        _install_ops(
            monkeypatch, native_host, batched_matmul=lambda a, b, use_sr=False: torch.bmm(a, b)
        )
        A, B = torch.randn(3, 40, 50), torch.randn(3, 50, 20)
        out = trntensor.einsum("bij,bjk->bik", A, B)
        torch.testing.assert_close(out, torch.bmm(A, B))
        assert [c[0] for c in native_host] == ["batched_matmul"]

    def test_precision_sr_passes_use_sr_and_warns_once(self, native_host, monkeypatch):
        _install_ops(monkeypatch, native_host, matmul=lambda a, b, use_sr=False: a @ b)
        monkeypatch.setattr(dispatch, "_sr_warning_emitted", False)
        monkeypatch.delenv(dispatch.SR_ENV_VAR, raising=False)
        A, B = torch.randn(128, 128), torch.randn(128, 128)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            trntensor.einsum("ij,jk->ik", A, B, precision="sr")
            trntensor.einsum("ij,jk->ik", A, B, precision="sr")
        assert native_host[0] == ("matmul", {"use_sr": True})
        assert len([w for w in rec if issubclass(w.category, RuntimeWarning)]) == 1

    def test_kahan_stays_on_host_fp64(self, native_host, monkeypatch):
        _install_ops(monkeypatch, native_host)  # any kernel launch would KeyError
        A, B = torch.randn(64, 64), torch.randn(64, 64)
        out = trntensor.einsum("ij,jk->ik", A, B, precision="kahan")
        torch.testing.assert_close(out, (A.double() @ B.double()).float())
        assert native_host == []

    def test_multi_einsum_batched_routes_to_batched_matmul(self, native_host, monkeypatch):
        _install_ops(
            monkeypatch, native_host, batched_matmul=lambda a, b, use_sr=False: torch.bmm(a, b)
        )
        A = torch.randn(64, 32)
        Bs = [torch.randn(32, 16) for _ in range(3)]
        outs = trntensor.multi_einsum(*[("ij,jk->ik", A, b) for b in Bs])
        for o, b in zip(outs, Bs, strict=True):
            torch.testing.assert_close(o, A @ b)
        assert [c[0] for c in native_host] == ["batched_matmul"]

    def test_multi_einsum_heterogeneous_routes_to_matmul(self, native_host, monkeypatch):
        _install_ops(monkeypatch, native_host, matmul=lambda a, b, use_sr=False: a @ b)
        A = torch.randn(64, 32)
        outs = trntensor.multi_einsum(
            ("ij,jk->ik", A, torch.randn(32, 16)), ("ij,jk->ik", A, torch.randn(32, 8))
        )
        assert [o.shape for o in outs] == [(64, 16), (64, 8)]
        assert [c[0] for c in native_host] == ["matmul", "matmul"]

    def test_ao_to_mo_transform_routes_to_trn2_op(self, native_host, monkeypatch):
        def ao_to_mo(eri, C_occ, C_vir):
            assert eri.shape[0] % 128 == 0 and eri.shape[2] % 128 == 0
            assert C_vir.shape[1] % 128 == 0
            return torch.einsum("mi,na,mnP->iaP", C_occ, C_vir, eri)

        _install_ops(monkeypatch, native_host, ao_to_mo_trn2=ao_to_mo)
        torch.manual_seed(0)
        eri, C_occ, C_vir = torch.randn(40, 40, 6), torch.randn(40, 5), torch.randn(40, 9)
        out = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
        assert out.shape == (5, 9, 6)
        torch.testing.assert_close(
            out, torch.einsum("mi,na,mnP->iaP", C_occ, C_vir, eri), atol=1e-4, rtol=1e-4
        )
        assert [c[0] for c in native_host] == ["ao_to_mo_trn2"]

    def test_ao_to_mo_transform_legacy_routes_and_slices(self, native_host, monkeypatch):
        monkeypatch.setenv("TRNTENSOR_AO_TO_MO", "legacy")
        monkeypatch.setattr(dispatch, "_native_kernel_deprecations_emitted", set())

        def ao_to_mo(eri, C_occ, C_vir):
            return torch.einsum("mi,na,mnP->iaP", C_occ, C_vir, eri)

        _install_ops(monkeypatch, native_host, ao_to_mo_transform=ao_to_mo)
        torch.manual_seed(0)
        eri, C_occ, C_vir = torch.randn(40, 40, 6), torch.randn(40, 5), torch.randn(40, 9)
        with pytest.warns(FutureWarning, match="deprecated on the native"):
            out = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
        torch.testing.assert_close(
            out, torch.einsum("mi,na,mnP->iaP", C_occ, C_vir, eri), atol=1e-4, rtol=1e-4
        )
        assert [c[0] for c in native_host] == ["ao_to_mo_transform"]

    def test_mp2_energy_routes_to_row_op(self, native_host, monkeypatch):
        from trntensor.quantum import _cpu_mp2_energy

        def row(bt_i, Bw, eps_occ_i, eps_occ_w, eps_vir_col, eps_vir_row):
            # Torch stand-in: per-partition partials (128, ND), all in row 0.
            out = torch.zeros(128, Bw.shape[0])
            for d in range(Bw.shape[0]):
                T = bt_i.T @ Bw[d]
                denom = eps_occ_i[0, 0] + eps_occ_w[0, d] - eps_vir_col - eps_vir_row
                out[0, d] = (T * (2 * T - T.T) / denom).sum()
            return out

        _install_ops(monkeypatch, native_host, mp2_energy_row=row)
        torch.manual_seed(1)
        nocc, nvir, naux = 5, 7, 10
        B = torch.randn(nocc, nvir, naux) * 0.1
        eps_occ = -torch.rand(nocc) - 0.5
        eps_vir = torch.rand(nvir) + 0.1
        e = trntensor.mp2_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(e, _cpu_mp2_energy(B, eps_occ, eps_vir), atol=1e-5, rtol=1e-4)
        assert [c[0] for c in native_host] == ["mp2_energy_row"] * nocc

    def test_mp2_energy_legacy_routes_to_op(self, native_host, monkeypatch):
        from trntensor.quantum import _cpu_mp2_energy

        monkeypatch.setenv("TRNTENSOR_MP2_ENERGY", "legacy")
        monkeypatch.setattr(dispatch, "_native_kernel_deprecations_emitted", set())

        def mp2(B_T, eps_occ_b, eps_vir, eps_vir_b):
            # Torch stand-in on the kernel's padded, transposed layout.
            B = B_T.transpose(1, 2)
            eo, ev = eps_occ_b[0], eps_vir[:, 0]
            nocc = B.shape[0]
            partial = torch.zeros(nocc, nocc)
            for i in range(nocc):
                for j in range(nocc):
                    T = B[i] @ B[j].T
                    denom = eo[i] + eo[j] - ev[:, None] - ev[None, :]
                    partial[i, j] = (T * (2 * T - T.T) / denom).sum()
            return partial

        _install_ops(monkeypatch, native_host, mp2_energy=mp2)
        torch.manual_seed(1)
        nocc, nvir, naux = 3, 7, 10
        B = torch.randn(nocc, nvir, naux) * 0.1
        eps_occ = -torch.rand(nocc) - 0.5
        eps_vir = torch.rand(nvir) + 0.1
        with pytest.warns(FutureWarning, match="deprecated on the native"):
            e = trntensor.mp2_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(e, _cpu_mp2_energy(B, eps_occ, eps_vir), atol=1e-5, rtol=1e-4)
        assert [c[0] for c in native_host] == ["mp2_energy"]


class TestPathAwareThresholds:
    def _cpu_ops(self):
        return torch.empty(2, 2), torch.empty(2, 2)

    def test_native_uses_native_thresholds(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        assert dispatch._min_nki_flops(*self._cpu_ops()) == dispatch._MIN_NKI_FLOPS_NATIVE
        assert dispatch._nki_flop_threshold(pinned=True) == dispatch._MIN_NKI_FLOPS_NATIVE_PINNED

    def test_native_pinned_means_on_neuron_device(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")  # CPU stands in for neuron
        assert dispatch._min_nki_flops(*self._cpu_ops()) == dispatch._MIN_NKI_FLOPS_NATIVE_PINNED

    def test_xla_keeps_xla_thresholds(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        assert dispatch._min_nki_flops(*self._cpu_ops()) == dispatch._MIN_NKI_FLOPS
        assert dispatch._nki_flop_threshold(pinned=True) == dispatch._MIN_NKI_FLOPS_PINNED

    def test_simulator_keeps_xla_thresholds(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: True)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        assert dispatch._min_nki_flops(*self._cpu_ops()) == dispatch._MIN_NKI_FLOPS

    def test_xla_defaults_unchanged(self, monkeypatch, tmp_path):
        for v in (
            "TRNTENSOR_MIN_NKI_FLOPS",
            "TRNTENSOR_MIN_NKI_FLOPS_PINNED",
            "TRNTENSOR_MIN_NKI_FLOPS_NATIVE",
            "TRNTENSOR_MIN_NKI_FLOPS_NATIVE_PINNED",
        ):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("TRNTENSOR_AUTOTUNE_CACHE", str(tmp_path / "missing.json"))
        assert dispatch._load_min_nki_flops() == (2_000_000_000, 1_000_000_000)
        # Measured on Trn2 (#41): offload host work from 500 M multiply-adds;
        # the NKI kernels never beat aten on the device.
        assert dispatch._load_min_nki_flops_native() == (500_000_000, dispatch.NKI_NEVER)

    def test_never_parses(self, monkeypatch, tmp_path):
        cache = tmp_path / "threshold.json"
        cache.write_text('{"trntensor_min_nki_flops_native_pinned": "never"}')
        monkeypatch.setenv("TRNTENSOR_AUTOTUNE_CACHE", str(cache))
        monkeypatch.delenv("TRNTENSOR_MIN_NKI_FLOPS_NATIVE_PINNED", raising=False)
        monkeypatch.setenv("TRNTENSOR_MIN_NKI_FLOPS_NATIVE", "inf")
        assert dispatch._load_min_nki_flops_native() == (dispatch.NKI_NEVER, dispatch.NKI_NEVER)

    def test_native_env_and_cache_overrides(self, monkeypatch, tmp_path):
        cache = tmp_path / "threshold.json"
        cache.write_text(
            '{"trntensor_min_nki_flops": 3000, "trntensor_min_nki_flops_native": 500,'
            ' "trntensor_min_nki_flops_native_pinned": 50}'
        )
        monkeypatch.setenv("TRNTENSOR_AUTOTUNE_CACHE", str(cache))
        monkeypatch.delenv("TRNTENSOR_MIN_NKI_FLOPS", raising=False)
        monkeypatch.delenv("TRNTENSOR_MIN_NKI_FLOPS_PINNED", raising=False)
        monkeypatch.delenv("TRNTENSOR_MIN_NKI_FLOPS_NATIVE", raising=False)
        monkeypatch.setenv("TRNTENSOR_MIN_NKI_FLOPS_NATIVE_PINNED", "7")
        assert dispatch._load_min_nki_flops() == (3000, 1_000_000_000)
        assert dispatch._load_min_nki_flops_native() == (500, 7)

    def test_plan_backend_uses_native_threshold(self, monkeypatch):
        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS", 10**12)
        monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS_NATIVE", 64**3)
        monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS_NATIVE_PINNED", 64**3)
        plan = trntensor.plan_contraction("ij,jk->ik", torch.empty(64, 64), torch.empty(64, 64))
        assert plan.backend == "nki"
        # Default pinned threshold (NKI never beats aten): offload is aten.
        trntensor.clear_plan_cache()
        monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS_NATIVE_PINNED", dispatch.NKI_NEVER)
        plan = trntensor.plan_contraction("ij,jk->ik", torch.empty(64, 64), torch.empty(64, 64))
        assert plan.backend == "pytorch"


class TestNativeRoute:
    """Executor choice on the native path (_route), measured policy (#41)."""

    @pytest.fixture(autouse=True)
    def _native(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS_NATIVE", 1000)
        monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS_NATIVE_PINNED", dispatch.NKI_NEVER)

    def test_host_small_stays_on_host(self):
        assert dispatch._route(999, pinned=False) == "host"

    def test_host_large_offloads_to_aten(self):
        assert dispatch._route(1000, pinned=False) == "aten"
        assert dispatch._route(10**15, pinned=False) == "aten"

    def test_device_operands_run_aten_on_device(self):
        assert dispatch._route(10**15, pinned=True) == "host"

    def test_pinned_threshold_reenables_nki(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS_NATIVE_PINNED", 5000)
        assert dispatch._route(5000, pinned=True) == "nki"
        assert dispatch._route(4999, pinned=True) == "host"
        assert dispatch._route(5000, pinned=False) == "nki"
        assert dispatch._route(4999, pinned=False) == "aten"

    def test_xla_path_unchanged(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        assert dispatch._route(dispatch._MIN_NKI_FLOPS, pinned=False) == "nki"
        assert dispatch._route(dispatch._MIN_NKI_FLOPS - 1, pinned=False) == "host"

    def test_matmul_offload_uses_aten_not_kernel(self, monkeypatch):
        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "meta")

        def no_launch(*a, **k):
            raise AssertionError("NKI kernel must not launch below the pinned threshold")

        monkeypatch.setattr(dispatch, "_launch", no_launch)
        moved = []
        monkeypatch.setattr(dispatch, "_aten_on_device", lambda fn, *t: moved.append(fn) or fn(*t))
        A, B = torch.randn(16, 16), torch.randn(16, 16)
        torch.testing.assert_close(dispatch.nki_matmul(A, B), A @ B)
        torch.testing.assert_close(
            dispatch.nki_batched_matmul(A[None], B[None]), torch.bmm(A[None], B[None])
        )
        assert moved == [torch.matmul, torch.bmm]

    def test_fused_quantum_kernels_off_under_auto(self, monkeypatch):
        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
        monkeypatch.setattr(dispatch, "_backend", "auto")
        monkeypatch.delenv("TRNTENSOR_FORCE_BACKEND", raising=False)
        assert dispatch._use_fused_kernel() is False
        monkeypatch.setattr(dispatch, "_backend", "nki")
        assert dispatch._use_fused_kernel() is True
        monkeypatch.setattr(dispatch, "_backend", "auto")
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        assert dispatch._use_fused_kernel() is True


class TestNativeKernelRetirement:
    """#51: native policy for the matmul/bmm/AO→MO kernels and the MP2 row kernel."""

    @pytest.fixture(autouse=True)
    def _native(self, monkeypatch):
        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_native_kernel_deprecations_emitted", set())
        monkeypatch.setattr(dispatch, "_backend", "auto")
        monkeypatch.delenv("TRNTENSOR_FORCE_BACKEND", raising=False)
        monkeypatch.delenv("TRNTENSOR_MP2_ENERGY", raising=False)

    def _forced_matmul(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_backend", "nki")
        monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS_NATIVE", 0)
        monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS_NATIVE_PINNED", 0)
        monkeypatch.setattr(dispatch, "_launch", lambda k, name, a, b, **kw: a @ b)
        for name in ("matmul_kernel", "batched_matmul_kernel"):
            monkeypatch.setattr(_kernels, name, None, raising=False)

    def test_forced_native_matmul_warns_once(self, monkeypatch):
        self._forced_matmul(monkeypatch)
        A, B = torch.randn(128, 128), torch.randn(128, 128)
        with pytest.warns(FutureWarning, match="matmul kernel is deprecated"):
            dispatch.nki_matmul(A, B)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dispatch.nki_matmul(A, B)
        with pytest.warns(FutureWarning, match="batched_matmul kernel is deprecated"):
            dispatch.nki_batched_matmul(A[None], B[None])

    def test_xla_matmul_does_not_warn(self, monkeypatch):
        self._forced_matmul(monkeypatch)
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dispatch.nki_matmul(torch.randn(128, 128), torch.randn(128, 128))

    def test_mp2_route_auto_by_row_work(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "meta")
        small = torch.empty(16, 128, 512, device="meta")
        large = torch.empty(64, 448, 1536, device="meta")
        assert dispatch._mp2_energy_route(small) == "aten"
        assert dispatch._mp2_energy_route(large) == "rows"
        # Host B stays on aten under auto.
        assert dispatch._mp2_energy_route(torch.empty(64, 448, 1536)) == "aten"
        # Forced NKI: the row kernel whenever it fits in SBUF.
        monkeypatch.setattr(dispatch, "_backend", "nki")
        assert dispatch._mp2_energy_route(small) == "rows"
        too_big = torch.empty(8, 1024, 4096, device="meta")
        assert dispatch._mp2_row_dims(*too_big.shape)["sbuf_bytes"] > dispatch._MP2_ROW_SBUF_BUDGET
        assert dispatch._mp2_energy_route(too_big) == "aten"

    def test_mp2_route_env_override_and_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
        B = torch.empty(4, 8, 8)
        monkeypatch.setenv("TRNTENSOR_MP2_ENERGY", "legacy")
        assert dispatch._mp2_energy_route(B) == "legacy"
        monkeypatch.setenv("TRNTENSOR_MP2_ENERGY", "bogus")
        with pytest.raises(ValueError, match="TRNTENSOR_MP2_ENERGY"):
            dispatch._mp2_energy_route(B)
        monkeypatch.delenv("TRNTENSOR_MP2_ENERGY")
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        assert dispatch._mp2_energy_route(B) == "legacy"
        monkeypatch.setattr(dispatch, "_use_nki", lambda: False)
        assert dispatch._mp2_energy_route(B) == "aten"

    def test_ao_to_mo_route(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "meta")
        small = torch.empty(128, 128, 256, device="meta")
        large = torch.empty(256, 256, 512, device="meta")
        assert dispatch._ao_to_mo_route(small, 16, 112) == "aten"
        assert dispatch._ao_to_mo_route(large, 32, 224) == "trn2"
        assert dispatch._ao_to_mo_route(torch.empty(256, 256, 512), 32, 224) == "aten"
        assert dispatch._ao_to_mo_route(large, 129, 224) == "aten"
        huge = torch.empty(4096, 4096, 512, device="meta")
        assert dispatch._ao_to_mo_route(huge, 32, 4000) == "aten"  # SBUF
        monkeypatch.setattr(dispatch, "_backend", "nki")
        assert dispatch._ao_to_mo_route(small, 16, 112) == "trn2"
        monkeypatch.setenv("TRNTENSOR_AO_TO_MO", "legacy")
        assert dispatch._ao_to_mo_route(small, 16, 112) == "legacy"

    @pytest.mark.parametrize("nocc", [1, 2, 5, 6, 7])
    def test_mp2_row_walk_matches_reference(self, nocc):
        from trntensor.quantum import _cpu_mp2_energy

        g = torch.Generator().manual_seed(nocc)
        B = torch.randn(nocc, 20, 40, generator=g, dtype=torch.float64) * 0.1
        eo = -torch.rand(nocc, generator=g, dtype=torch.float64) - 1.0
        ev = torch.rand(20, generator=g, dtype=torch.float64) + 0.5
        dims = dispatch._mp2_row_dims(nocc, 20, 40)
        assert dims["nd_pad"] % 2 == 0
        torch.testing.assert_close(
            dispatch._mp2_row_reference(B, eo, ev).double(),
            _cpu_mp2_energy(B, eo, ev).double(),
            rtol=1e-6,
            atol=1e-9,
        )


class TestResidency:
    def test_to_device_native_places_on_neuron(self, monkeypatch):
        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_to_xla", _no_xla)
        moved = []

        class T:
            device = torch.device("cpu")

            def to(self, dev):
                moved.append(dev)
                return "on-" + dev

        assert trntensor.to_device(T()) == "on-neuron"
        assert trntensor.to_xla(T()) == "on-neuron"  # automatic placement
        assert moved == ["neuron", "neuron"]

    def test_to_device_xla_path_uses_to_xla_helper(self, monkeypatch):
        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        sentinel = torch.ones(1)
        monkeypatch.setattr(dispatch, "_to_xla", lambda *ts: ([sentinel], ts[0].device))
        assert trntensor.to_device(torch.zeros(1)) is sentinel
        assert trntensor.to_xla(torch.zeros(1)) is sentinel

    def test_to_device_without_nki_raises(self, monkeypatch):
        monkeypatch.setattr(dispatch, "HAS_NKI", False)
        with pytest.raises(RuntimeError, match="NKI runtime"):
            trntensor.to_device(torch.zeros(1))

    def test_from_device_cpu_noop(self):
        t = torch.zeros(2)
        assert trntensor.from_device(t) is t
        assert trntensor.from_xla(t) is t


class TestXlaDeprecationHook:
    def _xla_selected(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)

    def test_off_by_default(self, monkeypatch):
        self._xla_selected(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert dispatch._get_dispatch_path() == "xla"
            trntensor.from_xla(torch.zeros(1))
            trntensor.from_device(torch.zeros(1))

    def test_fires_once_when_enabled(self, monkeypatch):
        self._xla_selected(monkeypatch)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            dispatch._get_dispatch_path()
            dispatch._get_dispatch_path()
        dep = [w for w in rec if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert str(dep[0].message) == (
            "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
            "Trn1/Inf2 remain supported in maintenance mode."
        )

    def test_not_on_native_path(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert dispatch._get_dispatch_path() == "native"

    def test_to_xla_from_xla_warn_once_when_enabled(self, monkeypatch):
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        monkeypatch.setattr(dispatch, "HAS_NKI", False)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in range(2):
                trntensor.from_xla(torch.zeros(1))
                with pytest.raises(RuntimeError):
                    trntensor.to_xla(torch.zeros(1))
        msgs = [str(w.message) for w in rec if issubclass(w.category, DeprecationWarning)]
        assert len(msgs) == 2
        assert any("trntensor.to_xla is deprecated; use trntensor.to_device" in m for m in msgs)
        assert any("trntensor.from_xla is deprecated; use trntensor.from_device" in m for m in msgs)

    def test_internal_residency_never_warns(self, native_host, monkeypatch):
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        _install_ops(monkeypatch, native_host, matmul=lambda a, b, use_sr=False: a @ b)
        A = torch.randn(64, 32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            trntensor.multi_einsum(
                ("ij,jk->ik", A, torch.randn(32, 16)), ("ij,jk->ik", A, torch.randn(32, 8))
            )


class TestLazyRegistration:
    def test_import_does_not_register_ops(self):
        # Fresh interpreter: importing trntensor (and the dispatch, parallel and
        # collectives modules) must not load `_native` or import torch_neuronx /
        # torch_xla, and must not run path detection.
        code = (
            "import sys, trntensor, trntensor.nki.dispatch as d, trntensor.parallel, "
            "trntensor.nki.collectives; "
            "bad = [m for m in ('trntensor.nki._native', 'torch_neuronx', 'torch_xla') "
            "if m in sys.modules]; "
            "assert not bad, bad; assert d._dispatch_path is None"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_ops_registered_on_first_native_use(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        assert "trntensor.nki._native" not in sys.modules
        assert recorded == {}

        ops = dispatch._get_native_ops()

        assert set(ops) == EXPECTED_OPS
        assert set(recorded) == {f"trntensor::{name}" for name in EXPECTED_OPS}
        assert all(mutates == {} for _, mutates, _ in recorded.values())
        assert all(fake is not None for _, _, fake in recorded.values())

        recorded.clear()
        assert dispatch._get_native_ops() is ops
        assert recorded == {}

    def test_op_signatures_build_a_schema(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()
        assert len(recorded) == len(EXPECTED_OPS)
        schemas = {}
        for name, (fn, mutates, _) in recorded.items():
            schema = torch.library.infer_schema(fn, mutates_args=mutates)
            assert schema.endswith("-> Tensor"), (name, schema)
            assert "Tensor" in schema.split("->")[0], (name, schema)
            schemas[name] = schema
        assert "bool use_sr=False" in schemas["trntensor::matmul"]
        assert "bool use_sr=False" in schemas["trntensor::batched_matmul"]

    def test_wrappers_call_kernels_with_use_sr(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        seen = []
        monkeypatch.setattr(
            _kernels,
            "matmul_kernel",
            lambda a, b, use_sr=False: seen.append(use_sr) or a @ b,
            raising=False,
        )
        ops = dispatch._get_native_ops()
        a = torch.ones(2, 2)
        torch.testing.assert_close(ops["matmul"](a, a, True), a @ a)
        assert seen == [True]

    def test_fake_fns_give_kernel_output_shapes(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()

        def fake(name):
            return recorded[f"trntensor::{name}"][2]

        def m(*shape, dtype=torch.float32):
            return torch.empty(*shape, dtype=dtype, device="meta")

        out = fake("matmul")(m(256, 128, dtype=torch.bfloat16), m(128, 512, dtype=torch.bfloat16))
        assert out.shape == (256, 512) and out.dtype == torch.bfloat16
        assert fake("matmul")(m(128, 128), m(128, 64), True).shape == (128, 64)
        assert fake("batched_matmul")(m(3, 128, 256), m(3, 256, 512)).shape == (3, 128, 512)
        mp2 = fake("mp2_energy")(m(5, 128, 256), m(128, 5), m(256, 1), m(128, 256))
        assert mp2.shape == (5, 5) and mp2.dtype == torch.float32
        b = fake("ao_to_mo_transform")(m(256, 256, 4), m(256, 16), m(256, 512))
        assert b.shape == (16, 512, 4)
