"""
Native TorchNeuron dispatch on hardware (Trn2/Trn3, torch-neuronx 2.14+).

Skipped unless the dispatch path resolves to "native": these do not run on
the XLA lineage (Trn1) or off-hardware. The existing kernel-correctness
suite (test_nki_kernels.py) also exercises the native path automatically on
a TorchNeuron host; this file covers what is specific to it — custom-op
registration, no torch_xla, direct op calls, torch.compile, and residency
on the ``neuron`` device.
"""

import os
import subprocess
import sys

import pytest
import torch

from trntensor.nki import HAS_NKI
from trntensor.nki.dispatch import _get_dispatch_path, _use_simulator


def _native() -> bool:
    if not HAS_NKI or _use_simulator():
        return False
    try:
        return _get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

ATOL = 1e-3
RTOL = 1e-4


def _ops():
    from trntensor.nki.dispatch import _get_native_ops

    return _get_native_ops()


def _dev(*ts):
    return [t.to("neuron") for t in ts]


@pytest.fixture
def nki_backend(monkeypatch):
    import trntensor
    from trntensor.nki import dispatch

    for name in ("_MIN_NKI_FLOPS_NATIVE", "_MIN_NKI_FLOPS_NATIVE_PINNED"):
        monkeypatch.setattr(dispatch, name, 0)
    prev = trntensor.get_backend()
    trntensor.set_backend("nki")
    yield
    trntensor.set_backend(prev)


class TestNativeDispatch:
    def test_native_path_never_imports_torch_xla(self):
        code = (
            "import sys, torch, trntensor; "
            "from trntensor.nki.dispatch import _get_dispatch_path; "
            "a = torch.randn(128, 128); b = torch.randn(128, 256); "
            "c = trntensor.einsum('ij,jk->ik', a, b); "
            "assert _get_dispatch_path() == 'native'; "
            "assert 'trntensor.nki._native' in sys.modules, 'kernel did not launch'; "
            "assert 'torch_xla' not in sys.modules"
        )
        env = dict(
            os.environ,
            TRNTENSOR_FORCE_BACKEND="nki",
            TRNTENSOR_MIN_NKI_FLOPS_NATIVE="0",
            TRNTENSOR_REQUIRE_NKI="1",
        )
        subprocess.run([sys.executable, "-c", code], check=True, env=env)

    def test_ops_registered_in_torch_ops(self):
        _ops()
        for name in ("matmul", "batched_matmul", "mp2_energy", "ao_to_mo_transform"):
            assert hasattr(torch.ops.trntensor, name)

    def test_matmul_op_eager_matches_reference(self):
        torch.manual_seed(0)
        a, b = torch.randn(256, 128), torch.randn(128, 512)
        out = torch.ops.trntensor.matmul(*_dev(a, b))
        assert out.device.type == "neuron"
        torch.testing.assert_close(out.cpu(), a @ b, atol=ATOL, rtol=RTOL)

    def test_matmul_op_use_sr_compiles_and_matches(self):
        torch.manual_seed(0)
        a, b = torch.randn(128, 128), torch.randn(128, 512)
        out = torch.ops.trntensor.matmul(*_dev(a, b), use_sr=True)
        torch.testing.assert_close(out.cpu(), a @ b, atol=ATOL, rtol=RTOL)

    def test_batched_matmul_op_matches_reference(self):
        torch.manual_seed(1)
        a, b = torch.randn(3, 128, 128), torch.randn(3, 128, 256)
        out = torch.ops.trntensor.batched_matmul(*_dev(a, b)).cpu()
        torch.testing.assert_close(out, torch.bmm(a, b), atol=ATOL, rtol=RTOL)

    def test_matmul_op_inside_torch_compile(self):
        matmul = _ops()["matmul"]

        @torch.compile(backend="neuron", fullgraph=True)
        def scaled(a, b):
            return matmul(a, b) * 2.0

        torch.manual_seed(2)
        a, b = torch.randn(128, 128), torch.randn(128, 256)
        out = scaled(*_dev(a, b)).cpu()
        torch.testing.assert_close(out, (a @ b) * 2.0, atol=2 * ATOL, rtol=RTOL)

    def test_einsum_matches_reference(self, nki_backend):
        import trntensor

        torch.manual_seed(3)
        a, b = torch.randn(200, 130), torch.randn(130, 70)  # exercises padding
        out = trntensor.einsum("ij,jk->ik", a, b)
        assert out.device.type == "cpu"
        torch.testing.assert_close(out, a @ b, atol=ATOL, rtol=RTOL)

    def test_residency_on_neuron_device(self, nki_backend):
        import trntensor

        torch.manual_seed(4)
        a = trntensor.to_device(torch.randn(128, 128))
        b = trntensor.to_device(torch.randn(128, 256))
        assert a.device.type == "neuron"
        out = trntensor.einsum("ij,jk->ik", a, b)
        assert out.device.type == "neuron"
        ref = trntensor.from_device(a) @ trntensor.from_device(b)
        torch.testing.assert_close(trntensor.from_device(out), ref, atol=ATOL, rtol=RTOL)

    def test_df_mp2_pipeline_on_device(self, nki_backend):
        """ao_to_mo_transform -> mp2_energy with every operand resident on
        ``neuron``; the intermediate B never leaves the device."""
        import trntensor
        from trntensor.quantum import _cpu_mp2_energy

        torch.manual_seed(7)
        nbasis, nocc, nvir, naux = 32, 5, 10, 16
        eri = torch.randn(nbasis, nbasis, naux) * 0.1
        C_occ = torch.randn(nbasis, nocc)
        C_vir = torch.randn(nbasis, nvir)
        eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
        eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1

        dev = [trntensor.to_device(t) for t in (eri, C_occ, C_vir, eps_occ, eps_vir)]
        B_x = trntensor.ao_to_mo_transform(*dev[:3])
        assert B_x.device.type == "neuron"
        E = trntensor.from_device(trntensor.mp2_energy(B_x, *dev[3:]))

        B_ref = torch.einsum("mi,na,mnP->iaP", C_occ, C_vir, eri)
        torch.testing.assert_close(
            E, _cpu_mp2_energy(B_ref, eps_occ, eps_vir), atol=1e-3, rtol=1e-4
        )
