"""
Native TorchNeuron dispatch on hardware (Trn2/Trn3, torch-neuronx 2.14+).

Skipped unless the dispatch path resolves to "native": these do not run on
the XLA lineage (Trn1) or off-hardware. The existing kernel-correctness
suite (test_nki_kernels.py) also exercises the native path automatically on
a TorchNeuron host; this file covers what is specific to it — custom-op
registration, no torch_xla, direct op calls, torch.compile, and residency
on the ``neuron`` device.
"""

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
        """Path selection and op registration in a fresh interpreter never
        import torch_xla.

        The subprocess deliberately stops before any kernel launch: inside
        the suite the parent pytest already holds the visible NeuronCore, so
        a launch in a child process fails in ``nrt_init`` (NRT_FAILURE).
        The launch half of the check runs in-process in
        ``test_launch_does_not_import_torch_xla``.
        """
        code = (
            "import sys, torch, trntensor; "
            "from trntensor.nki import dispatch as d; "
            "assert d._dispatch_path is None; "
            "d._dispatch_path = 'native'; "  # skip the device probe: no runtime init
            "d._get_native_ops(); "
            "assert 'trntensor.nki._native' in sys.modules; "
            "assert hasattr(torch.ops.trntensor, 'matmul'); "
            "assert 'torch_xla' not in sys.modules, 'torch_xla imported'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_launch_does_not_import_torch_xla(self, nki_backend):
        import trntensor

        if "torch_xla" in sys.modules:
            pytest.skip("torch_xla was imported by another test in this process")
        a, b = torch.randn(128, 128), torch.randn(128, 256)
        out = trntensor.einsum("ij,jk->ik", a, b)
        torch.testing.assert_close(out, a @ b, atol=ATOL, rtol=RTOL)
        assert "trntensor.nki._native" in sys.modules, "kernel did not launch"
        assert "torch_xla" not in sys.modules

    def test_ops_registered_in_torch_ops(self):
        _ops()
        for name in (
            "matmul",
            "batched_matmul",
            "mp2_energy",
            "ao_to_mo_transform",
            "mp2_energy_row",
            "ao_to_mo_trn2",
        ):
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


class TestNativeTrn2Kernels:
    """#51: the Trn2 AO→MO and MP2 row kernels on hardware (LNC=1 and LNC=2)."""

    @pytest.mark.parametrize("nbasis,nocc,nvir,naux", [(128, 8, 120, 128), (256, 32, 200, 512)])
    def test_ao_to_mo_trn2_matches_reference(self, nbasis, nocc, nvir, naux):
        from trntensor.nki import dispatch

        g = torch.Generator().manual_seed(nbasis)
        eri = torch.randn(nbasis, nbasis, naux, generator=g) * 0.1
        C_occ = torch.randn(nbasis, nocc, generator=g)
        C_vir = torch.randn(nbasis, nvir, generator=g)
        out = dispatch._nki_ao_to_mo_trn2(*_dev(eri, C_occ, C_vir))
        assert out.device.type == "neuron" and out.shape == (nocc, nvir, naux)
        ref = torch.einsum("na,inP->iaP", C_vir, torch.einsum("mi,mnP->inP", C_occ, eri))
        torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-4)

    @pytest.mark.parametrize("nocc,nvir,naux", [(1, 64, 100), (5, 128, 256), (8, 200, 300)])
    def test_mp2_row_kernel_matches_reference(self, nocc, nvir, naux):
        from trntensor.nki import dispatch
        from trntensor.quantum import _cpu_mp2_energy

        g = torch.Generator().manual_seed(nocc)
        B = torch.randn(nocc, nvir, naux, generator=g) * 0.05
        eo = -torch.rand(nocc, generator=g) - 1.0
        ev = torch.rand(nvir, generator=g) + 0.5
        e = dispatch._nki_mp2_energy_rows(*_dev(B, eo, ev))
        ref = _cpu_mp2_energy(B.double(), eo.double(), ev.double())
        assert abs(float(e.cpu()) - float(ref)) <= 1e-5 * abs(float(ref))

    def test_auto_routes_large_problems_to_trn2_kernels(self, monkeypatch):
        import trntensor
        from trntensor.nki import dispatch

        calls = []
        ops = dict(_ops())
        for name in ("mp2_energy_row", "ao_to_mo_trn2"):
            fn = ops[name]
            ops[name] = lambda *a, _fn=fn, _n=name, **k: calls.append(_n) or _fn(*a, **k)
        monkeypatch.setattr(dispatch, "_native_ops", ops)
        prev = trntensor.get_backend()
        trntensor.set_backend("auto")
        try:
            B = torch.randn(32, 320, 1024) * 0.05
            eo, ev = -torch.rand(32) - 1.0, torch.rand(320) + 0.5
            trntensor.mp2_energy(*_dev(B, eo, ev))
            eri = torch.randn(128, 128, 128)
            trntensor.ao_to_mo_transform(*_dev(eri, torch.randn(128, 8), torch.randn(128, 120)))
            # Small MP2 stays on aten.
            trntensor.mp2_energy(*_dev(B[:4, :32, :64].contiguous(), eo[:4], ev[:32]))
        finally:
            trntensor.set_backend(prev)
        assert calls.count("mp2_energy_row") == 32
        assert calls.count("ao_to_mo_trn2") == 1
