"""Simulator-backed kernel correctness tests (NKI 0.3.0 Stable).

Run with ``TRNTENSOR_USE_SIMULATOR=1`` on any x86_64 Linux host that has
``nki>=0.3.0`` installed. Bypasses torch_xla + NEFF compile; routes
kernel dispatch through ``nki.simulate(kernel)(np_args)``.

Intentionally curated to small shapes — the CPU simulator is slow above
1024³. Correctness parity with hardware at these scales is what we
verify; perf lives in the hardware test suite.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.nki_simulator


@pytest.fixture(autouse=True)
def _simulator_enabled():
    """Skip the whole module unless ``TRNTENSOR_USE_SIMULATOR`` is set and
    ``nki`` is importable. Fail loudly vs silently falling through to
    the hardware/CPU fallback path.
    """
    if os.environ.get("TRNTENSOR_USE_SIMULATOR", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("TRNTENSOR_USE_SIMULATOR=1 required")

    from trntensor.nki import HAS_NKI

    if not HAS_NKI:
        pytest.skip("nki package not importable on this host")


ATOL = 1e-3
RTOL = 1e-4


class TestMatmulSimulator:
    def test_aligned_128(self):
        import trntensor
        from trntensor.nki import dispatch

        trntensor.set_backend("nki")
        prev = dispatch._MIN_NKI_FLOPS
        dispatch._MIN_NKI_FLOPS = 0
        try:
            torch.manual_seed(0)
            A = torch.randn(128, 128)
            B = torch.randn(128, 256)
            out = trntensor.einsum("ij,jk->ik", A, B)
            torch.testing.assert_close(out, A @ B, atol=ATOL, rtol=RTOL)
        finally:
            dispatch._MIN_NKI_FLOPS = prev
            trntensor.set_backend("auto")


class TestBatchedMatmulSimulator:
    def test_aligned_small(self):
        import trntensor
        from trntensor.nki import dispatch

        trntensor.set_backend("nki")
        prev = dispatch._MIN_NKI_FLOPS
        dispatch._MIN_NKI_FLOPS = 0
        try:
            torch.manual_seed(1)
            A = torch.randn(2, 128, 128)
            B = torch.randn(2, 128, 128)
            out = trntensor.einsum("bij,bjk->bik", A, B)
            torch.testing.assert_close(out, torch.bmm(A, B), atol=ATOL, rtol=RTOL)
        finally:
            dispatch._MIN_NKI_FLOPS = prev
            trntensor.set_backend("auto")


class TestAoToMoTransformSimulator:
    def test_small(self):
        import trntensor

        trntensor.set_backend("nki")
        try:
            torch.manual_seed(0)
            nbasis, nocc, nvir, naux = 32, 5, 10, 16
            eri = torch.randn(nbasis, nbasis, naux) * 0.1
            C_occ = torch.randn(nbasis, nocc)
            C_vir = torch.randn(nbasis, nvir)
            got = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
            ref = torch.einsum("mi,na,mnP->iaP", C_occ, C_vir, eri)
            torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL)
        finally:
            trntensor.set_backend("auto")

    def test_k_tiled_nbasis_256(self):
        """K-tiling path: nbasis=256 = 2 × TILE_K. Verifies the multi-tile
        K loop against the CPU einsum reference via the simulator.
        """
        import trntensor

        trntensor.set_backend("nki")
        try:
            torch.manual_seed(10)
            nbasis, nocc, nvir, naux = 256, 8, 16, 16
            eri = torch.randn(nbasis, nbasis, naux) * 0.1
            C_occ = torch.randn(nbasis, nocc)
            C_vir = torch.randn(nbasis, nvir)
            got = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
            ref = torch.einsum("mi,na,mnP->iaP", C_occ, C_vir, eri)
            torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL)
        finally:
            trntensor.set_backend("auto")

    def test_k_tiled_nbasis_non_multiple(self):
        """nbasis=200 is not a multiple of TILE_K; dispatch pads to 256.
        Padded zeros must not corrupt the result.
        """
        import trntensor

        trntensor.set_backend("nki")
        try:
            torch.manual_seed(11)
            nbasis, nocc, nvir, naux = 200, 6, 12, 12
            eri = torch.randn(nbasis, nbasis, naux) * 0.1
            C_occ = torch.randn(nbasis, nocc)
            C_vir = torch.randn(nbasis, nvir)
            got = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
            ref = torch.einsum("mi,na,mnP->iaP", C_occ, C_vir, eri)
            torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL)
        finally:
            trntensor.set_backend("auto")


class TestMp2EnergySimulator:
    def test_demo_shape(self):
        import trntensor
        from trntensor.quantum import _cpu_mp2_energy

        trntensor.set_backend("nki")
        try:
            torch.manual_seed(42)
            nocc, nvir, naux = 5, 19, 72
            B = torch.randn(nocc, nvir, naux) * 0.1
            eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
            eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1

            got = trntensor.mp2_energy(B, eps_occ, eps_vir)
            expected = _cpu_mp2_energy(B, eps_occ, eps_vir)
            torch.testing.assert_close(got, expected, atol=ATOL, rtol=RTOL)
        finally:
            trntensor.set_backend("auto")
