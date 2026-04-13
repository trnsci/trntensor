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
