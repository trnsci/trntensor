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

    def test_chunked_matches_unchunked(self, monkeypatch):
        """Chunking over i (bounded T memory) gives the same energy."""
        import trntensor.quantum as q

        torch.manual_seed(3)
        B = torch.randn(5, 6, 7, dtype=torch.float64) * 0.1
        eps_occ = -torch.rand(5, dtype=torch.float64) - 0.5
        eps_vir = torch.rand(6, dtype=torch.float64) + 0.1
        full = q._cpu_mp2_energy(B, eps_occ, eps_vir)
        monkeypatch.setattr(q, "_MP2_CHUNK_ELEMS", 2 * 5 * 6 * 6)  # chunk = 2 -> 2,2,1
        torch.testing.assert_close(q._cpu_mp2_energy(B, eps_occ, eps_vir), full)

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

    def test_chunked_matches_unchunked(self, monkeypatch):
        """Chunking over i (bounded T memory) gives the same energy."""
        import trntensor.quantum as q

        torch.manual_seed(3)
        B = torch.randn(5, 6, 7, dtype=torch.float64) * 0.1
        eps_occ = -torch.rand(5, dtype=torch.float64) - 0.5
        eps_vir = torch.rand(6, dtype=torch.float64) + 0.1
        full = q._cpu_mp2_energy(B, eps_occ, eps_vir)
        monkeypatch.setattr(q, "_MP2_CHUNK_ELEMS", 2 * 5 * 6 * 6)  # chunk = 2 -> 2,2,1
        torch.testing.assert_close(q._cpu_mp2_energy(B, eps_occ, eps_vir), full)

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


class TestPackedAoToMo:
    """#54: packed (μ≥ν) three-centre integrals."""

    @staticmethod
    def _sym(nb, naux, seed=0, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        e = torch.randn(nb, nb, naux, generator=g, dtype=dtype)
        return 0.5 * (e + e.transpose(0, 1))

    def test_pack_tril_is_pyscf_order(self):
        e = self._sym(7, 3)
        pk = trntensor.pack_tril(e)
        assert pk.shape == (28, 3)
        expected = torch.cat([e[mu, : mu + 1] for mu in range(7)])  # trnblas.parallel.pack_tril
        assert torch.equal(pk, expected)
        assert torch.equal(pk[5 * 6 // 2 + 2], e[5, 2])

    def test_unpack_round_trip(self):
        e = self._sym(9, 4)
        assert torch.equal(trntensor.unpack_tril(trntensor.pack_tril(e), 9), e)
        with pytest.raises(ValueError, match="nbasis=8 needs 36"):
            trntensor.unpack_tril(trntensor.pack_tril(e), 8)

    @pytest.mark.parametrize("layout", ["rows", "pyscf", "pyscf_view"])
    def test_packed_matches_full(self, layout):
        nb, naux, nocc = 11, 5, 3
        e = self._sym(nb, naux)
        g = torch.Generator().manual_seed(1)
        C_occ = torch.randn(nb, nocc, generator=g, dtype=torch.float64)
        C_vir = torch.randn(nb, nb - nocc, generator=g, dtype=torch.float64)
        pk = trntensor.pack_tril(e)
        arg = {"rows": pk, "pyscf": pk.T.contiguous(), "pyscf_view": pk.T.contiguous().T}[layout]
        out = trntensor.ao_to_mo_transform(arg, C_occ, C_vir)
        assert torch.equal(out, trntensor.ao_to_mo_transform(e, C_occ, C_vir))

    def test_packed_shape_errors(self):
        C = torch.randn(6, 2)
        with pytest.raises(ValueError, match="npair = nbasis"):
            trntensor.ao_to_mo_transform(torch.randn(20, 4), C, C)
        with pytest.raises(ValueError, match="3D"):
            trntensor.ao_to_mo_transform(torch.randn(21), C, C)
        with pytest.raises(ValueError, match="C_occ must have shape"):
            trntensor.ao_to_mo_transform(torch.randn(6, 6, 4), torch.randn(5, 2), C)


class TestDfExchange:
    """#60: K = Σ_{iP} X X^T from packed (μ≥ν) integrals."""

    @staticmethod
    def _case(n=13, naux=7, nocc=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        eri = torch.randn(n, n, naux, generator=g, dtype=torch.float64)
        eri = 0.5 * (eri + eri.transpose(0, 1))
        C = torch.randn(n, nocc, generator=g, dtype=torch.float64)
        X = torch.einsum("li,lnP->inP", C, eri)
        return eri, C, torch.einsum("imP,inP->mn", X, X)

    @pytest.mark.parametrize("layout", ["packed", "cderi", "full"])
    def test_matches_reference(self, layout):
        import trntensor

        eri, C, K = self._case()
        pk = trntensor.pack_tril(eri)
        B = {"packed": pk, "cderi": pk.T.contiguous(), "full": eri}[layout]
        out = trntensor.df_exchange(B, C, alpha=2.0)
        assert out.shape == (13, 13)
        torch.testing.assert_close(out, 2.0 * K)

    def test_gather_blocks(self, monkeypatch):
        """Several ν blocks of gathered rows give the same K."""
        import trntensor
        import trntensor.quantum as q

        eri, C, K = self._case(n=20, naux=9, nocc=3, seed=1)
        monkeypatch.setattr(q, "_DF_K_GATHER_BYTES", 20 * 9 * 8 * 3)
        torch.testing.assert_close(trntensor.df_exchange(trntensor.pack_tril(eri), C), K)

    def test_shape_errors(self):
        import trntensor

        eri, C, _ = self._case()
        with pytest.raises(ValueError):
            trntensor.df_exchange(torch.zeros(10, 7), C)
        with pytest.raises(ValueError):
            trntensor.df_exchange(eri[:, :5], C)
        with pytest.raises(ValueError):
            trntensor.df_exchange(trntensor.pack_tril(eri), C[:, 0])


class TestDfCoulomb:
    """#60: J = Σ_P B[:, P] (Σ_{λσ} D_λσ B[λσ, P]) from packed integrals."""

    @pytest.mark.parametrize("cderi", [False, True])
    def test_matches_reference(self, cderi):
        g = torch.Generator().manual_seed(5)
        n = 11
        eri = torch.randn(n, n, 6, generator=g, dtype=torch.float64)
        eri = 0.5 * (eri + eri.transpose(0, 1))
        D = torch.randn(n, n, generator=g, dtype=torch.float64)
        D = D + D.T
        pk = trntensor.pack_tril(eri)
        J = trntensor.df_coulomb(pk.T.contiguous() if cderi else pk, D)
        ref = torch.einsum("mnP,P->mn", eri, torch.einsum("lsP,ls->P", eri, D))
        torch.testing.assert_close(J, ref)
        # Packed weighted density in, packed J out.
        w = 2.0 * trntensor.pack_tril(D[:, :, None])[:, 0]
        r = torch.tril_indices(n, n)
        w[r[0] == r[1]] *= 0.5
        torch.testing.assert_close(
            trntensor.df_coulomb(pk, w), trntensor.pack_tril(ref[:, :, None])[:, 0]
        )

    def test_shape_errors(self):
        with pytest.raises(ValueError):
            trntensor.df_coulomb(torch.zeros(10, 3), torch.zeros(4, 5))
        with pytest.raises(ValueError):
            trntensor.df_coulomb(torch.zeros(10, 3), torch.zeros(11))
        with pytest.raises(ValueError):
            trntensor.df_coulomb(torch.zeros(4, 4, 3), torch.zeros(4, 4))
