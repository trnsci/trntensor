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


def _np_cp_als(T, init, n_iter):
    """Independent NumPy CP-ALS reference (trntensor#103).

    Each mode update is the least-squares solution of ``T_(k) ≈ A_k V^T``
    computed with ``np.linalg.lstsq`` on an einsum-built Khatri-Rao product
    (no normal equations), then columns are normalized into the weights.
    """
    A = [np.array(f, copy=True) for f in init]
    nd = T.ndim
    w = np.ones(A[0].shape[1])
    for _ in range(n_iter):
        for k in range(nd):
            rest = [A[i] for i in range(nd) if i != k]
            V = rest[0]
            for M in rest[1:]:
                V = np.einsum("ir,jr->ijr", V, M).reshape(-1, V.shape[1])
            Tk = np.moveaxis(T, k, 0).reshape(T.shape[k], -1)
            Ak = np.linalg.lstsq(V, Tk.T, rcond=None)[0].T
            w = np.linalg.norm(Ak, axis=0)
            A[k] = Ak / (w + 1e-10)
    return A, w


class TestCPInputDtype:
    """cp_decompose works in the input dtype (trntensor#103)."""

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_matches_numpy_als_fp64(self, dtype):
        gen = torch.Generator().manual_seed(103)
        shape, rank = (6, 5, 4), 3
        true = [torch.randn(n, rank, generator=gen, dtype=dtype) for n in shape]
        T = torch.einsum("ir,jr,kr->ijk", *true)
        T = T + 1e-3 * torch.randn(shape, generator=gen, dtype=dtype)
        init = [torch.randn(n, rank, generator=gen, dtype=dtype) for n in shape]

        factors, weights = trntensor.cp_decompose(T, rank, max_iter=8, tol=0.0, factors=init)
        ref_f, ref_w = _np_cp_als(T.numpy(), [f.numpy() for f in init], 8)

        assert all(f.dtype == dtype for f in factors)
        assert weights.dtype == torch.float64
        for f, r in zip(factors, ref_f, strict=True):
            np.testing.assert_allclose(f.numpy(), r, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(weights.numpy(), ref_w, rtol=1e-8)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_random_init_recovers_rank1(self, dtype):
        gen = torch.Generator().manual_seed(7)
        vs = [torch.randn(n, generator=gen, dtype=dtype) for n in (4, 5, 3)]
        T = torch.einsum("i,j,k->ijk", *vs)
        torch.manual_seed(0)
        factors, weights = trntensor.cp_decompose(T, 1, max_iter=50, tol=1e-12)
        R = torch.einsum("r,ir,jr,kr->ijk", weights.to(dtype), *factors)
        assert factors[0].dtype == dtype
        assert (torch.linalg.norm(R - T) / torch.linalg.norm(T)).item() < 1e-10

    def test_complex_nonneg_raises(self):
        T = torch.randn(3, 4, 2, dtype=torch.complex128)
        with pytest.raises(ValueError, match="real"):
            trntensor.cp_decompose(T, 2, nonneg=True)

    def test_float32_unchanged(self):
        """Real float32 results are bit-identical to the pre-#103 normal equations."""
        gen = torch.Generator().manual_seed(11)
        T = torch.randn(5, 4, 3, generator=gen)
        init = [torch.randn(n, 2, generator=gen) for n in T.shape]
        factors, weights = trntensor.cp_decompose(T, 2, max_iter=3, tol=0.0, factors=init)
        A = [f.clone() for f in init]
        for _ in range(3):
            for k in range(3):
                rest = [A[i] for i in range(3) if i != k]
                V = torch.stack([torch.kron(rest[0][:, r], rest[1][:, r]) for r in range(2)], dim=1)
                Tk = T.moveaxis(k, 0).reshape(T.shape[k], -1)
                Ak = torch.linalg.solve(V.T @ V, (Tk @ V).T).T
                w = torch.linalg.norm(Ak, dim=0)
                A[k] = Ak / (w.unsqueeze(0) + 1e-10)
        for f, r in zip(factors, A, strict=True):
            assert torch.equal(f, r)
        assert torch.equal(weights, w)


class TestCPReconstructDtype:
    """cp_reconstruct follows the factors' dtype and device (trntensor#78)."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
    def test_dtype_matches_fp64_reference(self, dtype):
        gen = torch.Generator().manual_seed(78)
        f = [torch.randn(n, 3, generator=gen, dtype=torch.float64) for n in (4, 5, 2)]
        w = torch.rand(3, generator=gen, dtype=torch.float64)
        ref = np.einsum("r,ir,jr,kr->ijk", w.numpy(), *[x.numpy() for x in f])
        lo_f, lo_w = [x.to(dtype) for x in f], w.to(dtype)
        # reference from the rounded inputs, evaluated in fp64
        ref_lo = np.einsum(
            "r,ir,jr,kr->ijk", lo_w.double().numpy(), *[x.double().numpy() for x in lo_f]
        )
        out = trntensor.cp_reconstruct(lo_f, lo_w)
        assert out.dtype == dtype
        tol = {torch.bfloat16: 5e-2, torch.float16: 5e-3, torch.float64: 1e-12}[dtype]
        np.testing.assert_allclose(out.double().numpy(), ref_lo, rtol=tol, atol=tol)
        if dtype == torch.float64:
            np.testing.assert_allclose(out.numpy(), ref, rtol=1e-12)

    def test_mixed_dtypes_promote(self):
        f = [torch.randn(3, 2, dtype=torch.bfloat16), torch.randn(4, 2, dtype=torch.float32)]
        out = trntensor.cp_reconstruct(f, torch.ones(2, dtype=torch.bfloat16))
        assert out.dtype == torch.float32

    def test_device_follows_factors(self):
        f = [torch.empty(n, 2, device="meta") for n in (3, 4, 2)]
        out = trntensor.cp_reconstruct(f, torch.empty(2, device="meta"))
        assert out.device.type == "meta" and out.shape == (3, 4, 2)


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


class TestComplexTuckerTT:
    """Complex tensors: the Tucker core must use U^H, not U^T (audit 3)."""

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_tucker_full_rank_identity(self, dtype):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 5, 3, generator=g, dtype=dtype)
        core, factors = trntensor.tucker_decompose(x, (4, 5, 3))
        tol = 1e-4 if dtype == torch.complex64 else 1e-10
        torch.testing.assert_close(
            trntensor.tucker_reconstruct(core, factors), x, rtol=tol, atol=tol
        )

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_tucker_core_is_projection(self, dtype):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(4, 5, 3, generator=g, dtype=dtype)
        core, U = trntensor.tucker_decompose(x, (2, 3, 2))
        ref = torch.einsum("abc,ai,bj,ck->ijk", x, U[0].conj(), U[1].conj(), U[2].conj())
        tol = 1e-4 if dtype == torch.complex64 else 1e-10
        torch.testing.assert_close(core, ref, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_tt_full_rank_identity(self, dtype):
        g = torch.Generator().manual_seed(2)
        x = torch.randn(3, 4, 2, 3, generator=g, dtype=dtype)
        tol = 1e-4 if dtype == torch.complex64 else 1e-10
        torch.testing.assert_close(
            trntensor.tt_reconstruct(trntensor.tt_decompose(x, 64)), x, rtol=tol, atol=tol
        )


_MATMULS = {
    torch.matmul,
    torch.Tensor.matmul,
    torch.Tensor.__matmul__,
    torch.Tensor.__rmatmul__,
    torch.mm,
    torch.bmm,
    torch.einsum,
    torch.linalg.solve,
    torch.linalg.pinv,
}


class _NoConjViews(torch.overrides.TorchFunctionMode):
    """Records every matmul / solve that receives a lazy conj view."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _MATMULS:
            for a in (*args, *kwargs.values()):
                if isinstance(a, torch.Tensor) and a.is_conj():
                    self.seen.append(getattr(func, "__name__", str(func)))
        return func(*args, **kwargs)


class TestNoLazyConjToMatmul:
    """On the neuron device matmul ignores the conj bit of a lazy view
    (aws-neuron-sdk#1455): ``A.mH @ B`` computes ``A.mT @ B``. No such view may
    reach a matmul or solve in the decompositions (#89)."""

    def test_mode_records_a_conj_view(self):
        a = torch.randn(3, 3, dtype=torch.complex64)
        with _NoConjViews() as m:
            _ = a.mH @ a
        assert m.seen

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_tucker(self, dtype):
        x = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(3), dtype=dtype)
        with _NoConjViews() as m:
            core, U = trntensor.tucker_decompose(x, (2, 3, 2))
        assert m.seen == []
        assert not core.is_conj()
        ref = torch.einsum("abc,ai,bj,ck->ijk", x, U[0].conj(), U[1].conj(), U[2].conj())
        tol = 1e-4 if dtype == torch.complex64 else 1e-10
        torch.testing.assert_close(core, ref, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_cp(self, dtype):
        x = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(4), dtype=dtype)
        torch.manual_seed(0)
        with _NoConjViews() as m:
            trntensor.cp_decompose(x, rank=2, max_iter=3)
        assert m.seen == []

    def test_real_mH_is_no_copy(self):
        from trntensor.decompose import _conj, _mH

        r = torch.randn(4, 3)
        assert _mH(r).data_ptr() == r.data_ptr()
        assert _conj(r).data_ptr() == r.data_ptr()
