"""Test tensor decompositions."""

import numpy as np
import pytest
import torch

import trntensor


class TestCP:
    def test_rank1(self):
        """Rank-1 tensor should decompose exactly."""
        a = torch.randn(4)
        b = torch.randn(5)
        c = torch.randn(3)
        T = torch.einsum("i,j,k->ijk", a, b, c)
        factors, weights = trntensor.cp_decompose(T, rank=1, tol=1e-4)
        reconstructed = trntensor.cp_reconstruct(factors, weights)
        error = torch.linalg.norm(T - reconstructed) / torch.linalg.norm(T)
        assert error.item() < 0.01

    def test_low_rank(self):
        """Low-rank tensor should be well-approximated."""
        torch.manual_seed(42)
        rank = 3
        a = torch.randn(8, rank)
        b = torch.randn(6, rank)
        c = torch.randn(4, rank)
        T = torch.zeros(8, 6, 4)
        for r in range(rank):
            T += torch.einsum("i,j,k->ijk", a[:, r], b[:, r], c[:, r])

        factors, weights = trntensor.cp_decompose(T, rank=rank, tol=1e-4, max_iter=200)
        reconstructed = trntensor.cp_reconstruct(factors, weights)
        error = torch.linalg.norm(T - reconstructed) / torch.linalg.norm(T)
        assert error.item() < 0.05

    def test_factor_shapes(self):
        T = torch.randn(5, 4, 3)
        factors, weights = trntensor.cp_decompose(T, rank=2, max_iter=5)
        assert len(factors) == 3
        assert factors[0].shape == (5, 2)
        assert factors[1].shape == (4, 2)
        assert factors[2].shape == (3, 2)
        assert weights.shape == (2,)


class TestTucker:
    def test_full_rank(self):
        """Full-rank Tucker should reconstruct exactly."""
        T = torch.randn(4, 3, 5)
        core, factors = trntensor.tucker_decompose(T, ranks=(4, 3, 5))
        reconstructed = trntensor.tucker_reconstruct(core, factors)
        np.testing.assert_allclose(reconstructed.numpy(), T.numpy(), atol=1e-4)

    def test_low_rank(self):
        """Low-rank Tucker should approximate."""
        torch.manual_seed(42)
        T = torch.randn(10, 8, 6)
        core, factors = trntensor.tucker_decompose(T, ranks=(3, 3, 3))
        assert core.shape == (3, 3, 3)
        assert factors[0].shape == (10, 3)
        assert factors[1].shape == (8, 3)
        assert factors[2].shape == (6, 3)
        reconstructed = trntensor.tucker_reconstruct(core, factors)
        # Rank-3 Tucker of a full-rank random tensor captures limited variance
        error = torch.linalg.norm(T - reconstructed) / torch.linalg.norm(T)
        assert error.item() < 1.0  # Just verify it's a reasonable approximation

    def test_factor_orthogonality(self):
        """Tucker factors from HOSVD should be orthogonal."""
        T = torch.randn(8, 6, 4)
        _, factors = trntensor.tucker_decompose(T, ranks=(3, 3, 3))
        for U in factors:
            UtU = U.T @ U
            np.testing.assert_allclose(UtU.numpy(), np.eye(3), atol=1e-5)

    def test_orthogonality_unequal_ranks(self):
        """Each mode's factor must be orthonormal at its own rank."""
        T = torch.randn(10, 8, 6)
        ranks = (4, 3, 2)
        _, factors = trntensor.tucker_decompose(T, ranks=ranks)
        for U, r in zip(factors, ranks, strict=True):
            UtU = U.T @ U
            np.testing.assert_allclose(UtU.numpy(), np.eye(r), atol=1e-5)


class TestCPEdgeCases:
    def test_zero_tensor(self):
        """CP of an all-zero tensor reconstructs to zero without blowing up."""
        T = torch.zeros(4, 5, 3)
        factors, weights = trntensor.cp_decompose(T, rank=2, max_iter=10)
        reconstructed = trntensor.cp_reconstruct(factors, weights)
        np.testing.assert_allclose(reconstructed.numpy(), T.numpy(), atol=1e-5)

    def test_rank_exceeds_min_dim(self):
        """Asking for rank > min(shape) must still return valid factor shapes."""
        T = torch.randn(3, 4, 5)
        rank = 6  # exceeds min dim (3)
        factors, weights = trntensor.cp_decompose(T, rank=rank, max_iter=5)
        assert weights.shape == (rank,)
        for f, s in zip(factors, T.shape, strict=True):
            assert f.shape == (s, rank)
            assert torch.isfinite(f).all()

    def test_rank_deficient_input(self):
        """A rank-2 tensor decomposed at rank 2 should reconstruct well."""
        torch.manual_seed(7)
        rank = 2
        a = torch.randn(6, rank)
        b = torch.randn(5, rank)
        c = torch.randn(4, rank)
        T = torch.zeros(6, 5, 4)
        for r in range(rank):
            T += torch.einsum("i,j,k->ijk", a[:, r], b[:, r], c[:, r])

        factors, weights = trntensor.cp_decompose(T, rank=rank, max_iter=300, tol=1e-5)
        reconstructed = trntensor.cp_reconstruct(factors, weights)
        err = torch.linalg.norm(T - reconstructed) / torch.linalg.norm(T)
        assert err.item() < 0.05


class TestTT:
    """Tests for tt_decompose / tt_reconstruct (#23)."""

    def test_core_shapes(self):
        """Boundary bond dims are 1; adjacent cores share their bond dimension."""
        T = torch.randn(4, 5, 3)
        cores = trntensor.tt_decompose(T, max_rank=10)
        assert len(cores) == 3
        assert cores[0].shape[0] == 1  # left boundary
        assert cores[-1].shape[2] == 1  # right boundary
        for k in range(len(cores) - 1):
            assert cores[k].shape[2] == cores[k + 1].shape[0]

    def test_bond_dim_capped(self):
        """Bond dimensions do not exceed max_rank."""
        T = torch.randn(5, 6, 4)
        cores = trntensor.tt_decompose(T, max_rank=2)
        for core in cores:
            assert core.shape[0] <= 2
            assert core.shape[2] <= 2

    def test_roundtrip_random(self):
        """tt_decompose followed by tt_reconstruct recovers the tensor (large rank)."""
        torch.manual_seed(0)
        T = torch.randn(4, 5, 3)
        cores = trntensor.tt_decompose(T, max_rank=20)
        T_hat = trntensor.tt_reconstruct(cores)
        assert T_hat.shape == T.shape
        err = torch.linalg.norm(T - T_hat) / torch.linalg.norm(T)
        assert err.item() < 0.01

    def test_low_rank_exact(self):
        """Rank-1 tensor decomposes exactly at max_rank=1."""
        a = torch.randn(4)
        b = torch.randn(5)
        c = torch.randn(3)
        T = torch.einsum("i,j,k->ijk", a, b, c)
        cores = trntensor.tt_decompose(T, max_rank=1)
        T_hat = trntensor.tt_reconstruct(cores)
        err = torch.linalg.norm(T - T_hat) / torch.linalg.norm(T)
        assert err.item() < 1e-4

    def test_four_mode_shapes(self):
        """4D tensor produces 4 cores with correct boundary dims."""
        T = torch.randn(3, 4, 5, 2)
        cores = trntensor.tt_decompose(T, max_rank=6)
        assert len(cores) == 4
        assert cores[0].shape[0] == 1
        assert cores[-1].shape[2] == 1
        for k in range(len(cores) - 1):
            assert cores[k].shape[2] == cores[k + 1].shape[0]


class TestCPExtended:
    """Tests for non-negative CP and warm-start CP (#24)."""

    def test_nonneg_factors_positive(self):
        """Non-negative CP: all factor entries are ≥ 0 after decomposition."""
        torch.manual_seed(1)
        T = torch.rand(4, 5, 3)  # non-negative input
        factors, _ = trntensor.cp_decompose(T, rank=3, max_iter=50, nonneg=True)
        for f in factors:
            assert (f >= 0).all(), "negative factor entry found in nonneg CP"

    def test_nonneg_reconstructs_reasonably(self):
        """Non-negative CP approximation has bounded relative error."""
        torch.manual_seed(2)
        T = torch.rand(5, 4, 3)
        factors, weights = trntensor.cp_decompose(T, rank=4, max_iter=200, nonneg=True)
        T_hat = trntensor.cp_reconstruct(factors, weights)
        err = torch.linalg.norm(T - T_hat) / torch.linalg.norm(T)
        assert err.item() < 0.5  # loose bound — NTF converges slower than ALS

    def test_warmstart_accepts_previous_factors(self):
        """Passing prior factors as warm-start produces valid shapes and no error."""
        T = torch.randn(4, 5, 3)
        factors0, weights0 = trntensor.cp_decompose(T, rank=2, max_iter=5)
        factors1, weights1 = trntensor.cp_decompose(T, rank=2, max_iter=5, factors=factors0)
        assert len(factors1) == 3
        for f, s in zip(factors1, T.shape, strict=True):
            assert f.shape == (s, 2)

    def test_warmstart_wrong_rank_raises(self):
        """warm-start factors with wrong rank raises ValueError."""
        T = torch.randn(4, 5, 3)
        bad_factors = [torch.randn(s, 5) for s in T.shape]  # rank 5, not 2
        with pytest.raises(ValueError, match="warm-start"):
            trntensor.cp_decompose(T, rank=2, factors=bad_factors)

    def test_nonneg_and_warmstart_combined(self):
        """nonneg=True + warm-start: initial factors are clamped, output stays positive."""
        torch.manual_seed(3)
        T = torch.rand(4, 5, 3)
        init = [torch.randn(s, 2) for s in T.shape]  # may have negatives
        factors, _ = trntensor.cp_decompose(T, rank=2, max_iter=30, nonneg=True, factors=init)
        for f in factors:
            assert (f >= 0).all()
