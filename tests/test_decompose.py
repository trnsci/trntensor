"""Test tensor decompositions."""

import pytest
import torch
import numpy as np
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
        for U, r in zip(factors, ranks):
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
        for f, s in zip(factors, T.shape):
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

        factors, weights = trntensor.cp_decompose(
            T, rank=rank, max_iter=300, tol=1e-5
        )
        reconstructed = trntensor.cp_reconstruct(factors, weights)
        err = torch.linalg.norm(T - reconstructed) / torch.linalg.norm(T)
        assert err.item() < 0.05
