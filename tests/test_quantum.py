"""CPU-side tests for the quantum-chemistry primitives."""

import pytest
import torch

import trntensor
from trntensor.quantum import _cpu_mp2_energy


class TestMp2EnergyReference:
    def test_matches_manual_small(self):
        """4-index DF-MP2 on a hand-sized case; compare against a
        fully unrolled reference.
        """
        torch.manual_seed(0)
        nocc, nvir, naux = 2, 3, 5
        B = torch.randn(nocc, nvir, naux) * 0.1
        eps_occ = torch.tensor([-0.5, -0.4])
        eps_vir = torch.tensor([0.1, 0.2, 0.3])

        # Unrolled reference: explicitly build T_{ijab} and sum.
        T = torch.einsum("iap,jbp->ijab", B, B)
        denom = (
            eps_occ.view(-1, 1, 1, 1)
            + eps_occ.view(1, -1, 1, 1)
            - eps_vir.view(1, 1, -1, 1)
            - eps_vir.view(1, 1, 1, -1)
        )
        term = T * (2 * T - T.transpose(-1, -2)) / denom
        expected = term.sum()

        got = _cpu_mp2_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)

    def test_shape_errors(self):
        B3 = torch.randn(3, 4, 5)
        eps_occ = torch.randn(3)
        eps_vir = torch.randn(4)

        with pytest.raises(ValueError, match="must be 3D"):
            trntensor.mp2_energy(torch.randn(3, 4), eps_occ, eps_vir)

        with pytest.raises(ValueError, match="eps_occ"):
            trntensor.mp2_energy(B3, torch.randn(2), eps_vir)

        with pytest.raises(ValueError, match="eps_vir"):
            trntensor.mp2_energy(B3, eps_occ, torch.randn(5))

    def test_public_api_matches_reference(self):
        torch.manual_seed(1)
        B = torch.randn(3, 4, 6) * 0.1
        eps_occ = -torch.arange(3, dtype=torch.float32) - 0.5
        eps_vir = torch.arange(4, dtype=torch.float32) + 0.5
        e_pub = trntensor.mp2_energy(B, eps_occ, eps_vir)
        e_ref = _cpu_mp2_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(e_pub, e_ref, atol=1e-6, rtol=1e-6)


class TestAoToMoTransformReference:
    def test_matches_fused_einsum(self):
        """The CPU reference is the fused einsum by construction —
        verify it also matches the two-step decomposed form.
        """
        torch.manual_seed(0)
        nbasis, nocc, nvir, naux = 8, 3, 5, 12
        eri = torch.randn(nbasis, nbasis, naux) * 0.1
        C_occ = torch.randn(nbasis, nocc)
        C_vir = torch.randn(nbasis, nvir)

        got = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)

        # Two-step reference (materializes the intermediate).
        intermediate = torch.einsum("mi,mnP->inP", C_occ, eri)
        expected = torch.einsum("na,inP->iaP", C_vir, intermediate)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_composes_with_mp2_energy(self):
        """B = ao_to_mo_transform(eri, C_occ, C_vir); mp2_energy(B, ...) is
        the full DF-MP2 pipeline from AO integrals.
        """
        torch.manual_seed(2)
        nbasis, nocc, nvir, naux = 8, 3, 5, 12
        eri = torch.randn(nbasis, nbasis, naux) * 0.1
        C_occ = torch.randn(nbasis, nocc)
        C_vir = torch.randn(nbasis, nvir)
        eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
        eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1

        B = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
        assert B.shape == (nocc, nvir, naux)
        e = trntensor.mp2_energy(B, eps_occ, eps_vir)
        assert e.shape == ()
        assert torch.isfinite(e)

    def test_shape_errors(self):
        eri = torch.randn(6, 6, 8)
        C_occ = torch.randn(6, 2)
        C_vir = torch.randn(6, 3)

        with pytest.raises(ValueError, match="must be 3D"):
            trntensor.ao_to_mo_transform(torch.randn(6, 6), C_occ, C_vir)

        with pytest.raises(ValueError, match="first two dims must match"):
            trntensor.ao_to_mo_transform(torch.randn(6, 7, 8), C_occ, C_vir)

        with pytest.raises(ValueError, match="C_occ"):
            trntensor.ao_to_mo_transform(eri, torch.randn(5, 2), C_vir)

        with pytest.raises(ValueError, match="C_vir"):
            trntensor.ao_to_mo_transform(eri, C_occ, torch.randn(7, 3))
