"""Test einsum and contraction planning."""

import numpy as np
import pytest
import torch

import trntensor


class TestEinsum:
    def test_matmul(self):
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        result = trntensor.einsum("ij,jk->ik", A, B)
        expected = torch.matmul(A, B)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_matmul_transA(self):
        A = torch.randn(3, 4)
        B = torch.randn(3, 5)
        result = trntensor.einsum("ji,jk->ik", A, B)
        expected = A.T @ B
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_matmul_transB(self):
        A = torch.randn(4, 3)
        B = torch.randn(5, 3)
        result = trntensor.einsum("ij,kj->ik", A, B)
        expected = A @ B.T
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_dot_product(self):
        x = torch.randn(10)
        y = torch.randn(10)
        result = trntensor.einsum("i,i->", x, y)
        expected = torch.dot(x, y)
        np.testing.assert_allclose(result.item(), expected.item(), atol=1e-5)

    def test_outer_product(self):
        x = torch.randn(3)
        y = torch.randn(4)
        result = trntensor.einsum("i,j->ij", x, y)
        expected = torch.outer(x, y)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_trace(self):
        A = torch.randn(5, 5)
        result = trntensor.einsum("ii->", A)
        expected = torch.trace(A)
        np.testing.assert_allclose(result.item(), expected.item(), atol=1e-5)

    def test_batched_matmul(self):
        A = torch.randn(8, 4, 3)
        B = torch.randn(8, 3, 5)
        result = trntensor.einsum("bij,bjk->bik", A, B)
        expected = torch.bmm(A, B)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_df_mp2_contraction(self):
        """DF-MP2 energy contraction: T_ab = Σ_P B_ia^P B_jb^P"""
        nocc, nvir, naux = 3, 5, 10
        B = torch.randn(nocc, nvir, naux)
        i, j = 0, 1
        result = trntensor.einsum("ap,bp->ab", B[i], B[j])
        expected = B[i] @ B[j].T
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_4index_transform(self):
        """AO-to-MO half-transform: (iν|P) = Σ_μ C_μi (μν|P)"""
        nbasis, nocc, naux = 6, 2, 8
        C = torch.randn(nbasis, nocc)
        eri = torch.randn(nbasis, nbasis, naux)
        result = trntensor.einsum("mi,mnP->inP", C, eri)
        expected = torch.einsum("mi,mnP->inP", C, eri)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_vs_torch_einsum(self):
        """Verify against torch.einsum for a complex pattern."""
        A = torch.randn(3, 4, 5)
        B = torch.randn(4, 5, 6)
        subscripts = "ijk,jkl->il"
        result = trntensor.einsum(subscripts, A, B)
        expected = torch.einsum(subscripts, A, B)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)


class TestMultiEinsum:
    def test_multiple_contractions(self):
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        C = torch.randn(5, 2)
        results = trntensor.multi_einsum(
            ("ij,jk->ik", A, B),
            ("ij,jk->ik", B, C),
        )
        assert len(results) == 2
        np.testing.assert_allclose(results[0].numpy(), (A @ B).numpy(), atol=1e-5)
        np.testing.assert_allclose(results[1].numpy(), (B @ C).numpy(), atol=1e-5)

    def test_three_operand_chain(self):
        """A single einsum call with three operands (ABC)."""
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        C = torch.randn(5, 2)
        result = trntensor.einsum("ij,jk,kl->il", A, B, C)
        expected = A @ B @ C
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_multi_einsum_with_three_operand_chain(self):
        """multi_einsum entry that is itself a 3-operand chain."""
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        C = torch.randn(5, 2)
        results = trntensor.multi_einsum(
            ("ij,jk,kl->il", A, B, C),
            ("ij,ij->", A, A),
        )
        assert len(results) == 2
        np.testing.assert_allclose(results[0].numpy(), (A @ B @ C).numpy(), atol=1e-4)
        np.testing.assert_allclose(results[1].item(), (A * A).sum().item(), atol=1e-4)

    def test_shared_operand_result_correct(self):
        """Two contractions sharing an operand produce the same results as independent einsum calls."""
        nocc, nvir, naux = 3, 5, 8
        B = torch.randn(nocc, nvir, naux)
        i, j = 0, 1
        results = trntensor.multi_einsum(
            ("ap,bp->ab", B[i], B[j]),
            ("ap,bp->ab", B[j], B[i]),
        )
        assert len(results) == 2
        np.testing.assert_allclose(
            results[0].numpy(), trntensor.einsum("ap,bp->ab", B[i], B[j]).numpy(), atol=1e-5
        )
        np.testing.assert_allclose(
            results[1].numpy(), trntensor.einsum("ap,bp->ab", B[j], B[i]).numpy(), atol=1e-5
        )

    def test_non_shared_operands_unchanged(self):
        """multi_einsum with no shared tensors behaves identically to independent einsum calls."""
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        C = torch.randn(5, 2)
        D = torch.randn(2, 6)
        results = trntensor.multi_einsum(
            ("ij,jk->ik", A, B),
            ("ij,jk->ik", C, D),
        )
        np.testing.assert_allclose(results[0].numpy(), (A @ B).numpy(), atol=1e-5)
        np.testing.assert_allclose(results[1].numpy(), (C @ D).numpy(), atol=1e-5)


class TestPathExecution:
    """Tests for the greedy-path execution of 3+ operand einsums."""

    def test_three_operand_correctness(self):
        """einsum path result matches torch.einsum reference."""
        torch.manual_seed(0)
        A, B, C = torch.randn(5, 8), torch.randn(8, 6), torch.randn(6, 4)
        result = trntensor.einsum("ij,jk,kl->il", A, B, C)
        expected = torch.einsum("ij,jk,kl->il", A, B, C)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_four_operand_correctness(self):
        """Chain of four operands produces correct result."""
        torch.manual_seed(1)
        A, B, C, D = torch.randn(3, 5), torch.randn(5, 4), torch.randn(4, 6), torch.randn(6, 2)
        result = trntensor.einsum("ij,jk,kl,lm->im", A, B, C, D)
        expected = torch.einsum("ij,jk,kl,lm->im", A, B, C, D)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_three_operand_non_chain(self):
        """Non-chain 3-operand contraction (shared index in all three)."""
        torch.manual_seed(2)
        # "ij,ik,il->jkl" — outer product-like, all share index i
        A = torch.randn(4, 3)
        B = torch.randn(4, 5)
        C = torch.randn(4, 2)
        result = trntensor.einsum("ij,ik,il->jkl", A, B, C)
        expected = torch.einsum("ij,ik,il->jkl", A, B, C)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_three_operand_output_shape(self):
        """Output shape is correct for a 3-operand contraction."""
        A, B, C = torch.randn(7, 3), torch.randn(3, 5), torch.randn(5, 11)
        result = trntensor.einsum("ij,jk,kl->il", A, B, C)
        assert result.shape == (7, 11)

    def test_three_operand_nki_backend(self):
        """Large binary steps in a 3-operand chain route through trntensor dispatch.

        Verifies that _execute_path calls einsum() recursively so that matmul-shaped
        binary steps are eligible for NKI dispatch, not silently sent to torch.einsum.
        On CPU-only installs (HAS_NKI=False) this degrades to a correctness check.
        """
        from trntensor.nki import HAS_NKI
        from trntensor.nki.dispatch import _MIN_NKI_FLOPS

        # Both binary steps are 512×512×512 → well above default _MIN_NKI_FLOPS
        torch.manual_seed(3)
        A = torch.randn(512, 512)
        B = torch.randn(512, 512)
        C = torch.randn(512, 512)
        result = trntensor.einsum("ij,jk,kl->il", A, B, C)
        expected = torch.einsum("ij,jk,kl->il", A, B, C)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-4)

        # The plan for the first binary step (ij,jk->ik at 512×512×512) should
        # select "matmul" strategy and "nki" backend when NKI is available.
        trntensor.clear_plan_cache()
        step_plan = trntensor.plan_contraction("ij,jk->ik", A, B)
        assert step_plan.strategy == "matmul"
        if HAS_NKI and _MIN_NKI_FLOPS <= 512 * 512 * 512:
            assert step_plan.backend == "nki"


class TestPlan:
    def test_matmul_detected(self):
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        plan = trntensor.plan_contraction("ij,jk->ik", A, B)
        assert plan.strategy == "matmul"

    def test_bmm_detected(self):
        A = torch.randn(8, 4, 3)
        B = torch.randn(8, 3, 5)
        plan = trntensor.plan_contraction("bij,bjk->bik", A, B)
        assert plan.strategy == "bmm"

    def test_complex_falls_to_torch(self):
        A = torch.randn(3, 4, 5)
        B = torch.randn(4, 5, 6)
        plan = trntensor.plan_contraction("ijk,jkl->il", A, B)
        assert plan.strategy == "torch"

    def test_flop_estimate(self):
        A = torch.randn(10, 20)
        B = torch.randn(20, 30)
        flops = trntensor.estimate_flops("ij,jk->ik", A, B)
        assert flops == 10 * 20 * 30


class TestAlphaBeta:
    """Tests for alpha/beta scaling interface (#20)."""

    def test_alpha_scaling(self):
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, alpha=2.0)
        expected = 2.0 * (A @ B)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_beta_accumulate(self):
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        C = torch.randn(4, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, beta=0.5, out=C)
        expected = A @ B + 0.5 * C
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_alpha_beta_combined(self):
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        C = torch.randn(4, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, alpha=2.0, beta=0.5, out=C)
        expected = 2.0 * (A @ B) + 0.5 * C
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_defaults_unchanged(self):
        """Default alpha=1, beta=0, out=None is identical to plain einsum."""
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, alpha=1.0, beta=0.0, out=None)
        expected = A @ B
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)


class TestDtype:
    """Tests for the dtype= mixed-precision override (#22)."""

    def test_dtype_bf16_result(self):
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, dtype="bf16")
        assert result.dtype == torch.bfloat16

    def test_dtype_fp16_result(self):
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, dtype="fp16")
        assert result.dtype == torch.float16

    def test_dtype_torch_type(self):
        """Passing a torch.dtype directly is equivalent to the string alias."""
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, dtype=torch.bfloat16)
        assert result.dtype == torch.bfloat16

    def test_dtype_none_unchanged(self):
        """dtype=None leaves input dtypes unmodified."""
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, dtype=None)
        assert result.dtype == torch.float32

    def test_dtype_invalid_raises(self):
        with pytest.raises(ValueError, match="unknown dtype"):
            trntensor.einsum("ij,jk->ik", torch.randn(4, 3), torch.randn(3, 5), dtype="foo")

    def test_dtype_correctness_bf16(self):
        """bf16 matmul result is numerically close to fp32 reference."""
        import numpy as np

        torch.manual_seed(0)
        A = torch.randn(8, 6)
        B = torch.randn(6, 10)
        result_bf16 = trntensor.einsum("ij,jk->ik", A, B, dtype="bf16").float()
        result_fp32 = trntensor.einsum("ij,jk->ik", A, B)
        np.testing.assert_allclose(result_bf16.numpy(), result_fp32.numpy(), atol=0.05)


class TestPrecision:
    """Tests for the precision= kwarg on einsum (#28)."""

    def test_kahan_result_dtype_preserved(self):
        """fp32 inputs + precision='kahan' → result dtype is float32."""
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, precision="kahan")
        assert result.dtype == torch.float32

    def test_kahan_accuracy_ill_conditioned(self):
        """kahan result is closer to fp64 reference than plain fp32 for ill-conditioned sum."""
        import numpy as np

        torch.manual_seed(42)
        # Build vectors where plain fp32 summation loses precision:
        # large constant plus many small values that should nearly cancel.
        n = 512
        large = torch.full((1, n), 1e6)
        small = torch.randn(n, 1) * 1e-3  # small perturbations

        # Contraction: "ij,jk->ik" contracts over j, result shape (1,1)
        # fp64 reference
        ref = torch.einsum("ij,jk->ik", large.double(), small.double()).float()
        fast = trntensor.einsum("ij,jk->ik", large, small, precision="fast")
        kahan = trntensor.einsum("ij,jk->ik", large, small, precision="kahan")

        err_fast = abs(float(fast) - float(ref))
        err_kahan = abs(float(kahan) - float(ref))
        # kahan goes through fp64 internally so it is exact (or very close)
        assert err_kahan <= err_fast + 1e-3

    def test_dd_raises(self):
        """precision='dd' raises NotImplementedError pointing to 'kahan'."""
        A, B = torch.randn(4, 3), torch.randn(3, 5)
        with pytest.raises(NotImplementedError, match="kahan"):
            trntensor.einsum("ij,jk->ik", A, B, precision="dd")

    def test_fast_unchanged(self):
        """precision='fast' produces the same result as the default (no kwarg)."""
        import numpy as np

        torch.manual_seed(0)
        A, B = torch.randn(8, 6), torch.randn(6, 10)
        default = trntensor.einsum("ij,jk->ik", A, B)
        fast = trntensor.einsum("ij,jk->ik", A, B, precision="fast")
        np.testing.assert_array_equal(default.numpy(), fast.numpy())


class TestPrecisionSR:
    """precision='sr' dispatch and statistical-correctness tests."""

    def test_sr_plan_stored(self):
        A = torch.randn(32, 32)
        B = torch.randn(32, 32)
        plan = trntensor.plan_contraction("ij,jk->ik", A, B, precision="sr")
        assert plan.precision == "sr"

    def test_sr_output_shape_and_dtype(self):
        A = torch.randn(32, 32, dtype=torch.bfloat16)
        B = torch.randn(32, 32, dtype=torch.bfloat16)
        result = trntensor.einsum("ij,jk->ik", A, B, precision="sr")
        assert result.shape == (32, 32)
        assert result.dtype == torch.bfloat16

    def test_sr_result_close_to_fast(self):
        """SR result agrees with fast rounding to within ~sqrt(K)*u."""
        torch.manual_seed(42)
        A = torch.randn(64, 128, dtype=torch.bfloat16)
        B = torch.randn(128, 64, dtype=torch.bfloat16)
        fast = trntensor.einsum("ij,jk->ik", A, B, precision="fast")
        sr = trntensor.einsum("ij,jk->ik", A, B, precision="sr")
        # Tolerance: sqrt(K) * u for BF16, K=128, u≈0.008 → ~0.09
        torch.testing.assert_close(sr.float(), fast.float(), atol=0.2, rtol=0.0)

    def test_sr_mean_zero_bias(self):
        """Mean rounding error over many SR samples should be ~0."""
        torch.manual_seed(0)
        # Large enough to measure bias reliably
        A = torch.ones(128, 512, dtype=torch.bfloat16)
        B = torch.ones(512, 128, dtype=torch.bfloat16)
        expected = torch.full((128, 128), 512.0)  # exact integer, SR shouldn't matter
        errors = []
        for _ in range(20):
            result = trntensor.einsum("ij,jk->ik", A, B, precision="sr")
            errors.append((result.float() - expected).mean().item())
        mean_err = sum(errors) / len(errors)
        assert abs(mean_err) < 1.0, f"Mean SR bias {mean_err:.4f} unexpectedly large"

    def test_sr_invalid_precision_raises(self):
        A = torch.randn(8, 8)
        B = torch.randn(8, 8)
        with pytest.raises(ValueError, match="precision"):
            trntensor.einsum("ij,jk->ik", A, B, precision="invalid")

    def test_sr_non_matmul_uses_fast(self):
        """Non-matmul strategies accept precision='sr' without error (fast fallback)."""
        A = torch.randn(4, 4, 4)
        B = torch.randn(4, 4)
        # Falls to torch strategy
        result = trntensor.einsum("ijk,jl->ilk", A, B, precision="sr")
        expected = torch.einsum("ijk,jl->ilk", A, B)
        torch.testing.assert_close(result, expected)

    def test_sr_bmm_shape_and_dtype(self):
        """precision='sr' is honored for batched matmul (bmm strategy)."""
        A = torch.randn(4, 16, 32, dtype=torch.bfloat16)
        B = torch.randn(4, 32, 16, dtype=torch.bfloat16)
        result = trntensor.einsum("bij,bjk->bik", A, B, precision="sr")
        assert result.shape == (4, 16, 16)
        assert result.dtype == torch.bfloat16

    def test_sr_bmm_close_to_fast(self):
        """SR bmm result agrees with fast rounding to within ~sqrt(K)*u."""
        torch.manual_seed(42)
        A = torch.randn(4, 32, 64, dtype=torch.bfloat16)
        B = torch.randn(4, 64, 32, dtype=torch.bfloat16)
        fast = trntensor.einsum("bij,bjk->bik", A, B, precision="fast")
        sr = trntensor.einsum("bij,bjk->bik", A, B, precision="sr")
        torch.testing.assert_close(sr.float(), fast.float(), atol=0.3, rtol=0.0)

    def test_sr_multi_einsum_threads_precision(self):
        """multi_einsum precision='sr' threads through to each contraction."""
        torch.manual_seed(99)
        A = torch.randn(32, 64, dtype=torch.bfloat16)
        B = torch.randn(64, 32, dtype=torch.bfloat16)
        results_sr = trntensor.multi_einsum(("ij,jk->ik", A, B), precision="sr")
        results_fast = trntensor.multi_einsum(("ij,jk->ik", A, B))
        assert results_sr[0].dtype == torch.bfloat16
        torch.testing.assert_close(
            results_sr[0].float(), results_fast[0].float(), atol=0.2, rtol=0.0
        )


class TestMultiEinsumBatched:
    """Tests for homogeneous batching in multi_einsum (#33 Phase 3)."""

    def test_homogeneous_batching_result(self):
        """N same-shape matmul contractions via multi_einsum match individual einsum results."""
        torch.manual_seed(0)
        nvir, naux = 8, 12
        # Four independent "ap,bp->ab" contractions with different operands but same shape
        A0, A1, A2, A3 = (torch.randn(nvir, naux) for _ in range(4))
        B0, B1, B2, B3 = (torch.randn(nvir, naux) for _ in range(4))

        results = trntensor.multi_einsum(
            ("ap,bp->ab", A0, B0),
            ("ap,bp->ab", A1, B1),
            ("ap,bp->ab", A2, B2),
            ("ap,bp->ab", A3, B3),
        )
        assert len(results) == 4
        for i, (A, B) in enumerate([(A0, B0), (A1, B1), (A2, B2), (A3, B3)]):
            expected = trntensor.einsum("ap,bp->ab", A, B)
            np.testing.assert_allclose(results[i].numpy(), expected.numpy(), atol=1e-5)

    def test_homogeneous_batching_mixed_shapes_fallback(self):
        """Contractions with different shapes fall through to per-contraction path without error."""
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        C = torch.randn(5, 2)
        D = torch.randn(2, 6)
        # Same subscript, different shapes → should NOT batch (fallback)
        results = trntensor.multi_einsum(
            ("ij,jk->ik", A, B),
            ("ij,jk->ik", C, D),
        )
        assert len(results) == 2
        np.testing.assert_allclose(results[0].numpy(), (A @ B).numpy(), atol=1e-5)
        np.testing.assert_allclose(results[1].numpy(), (C @ D).numpy(), atol=1e-5)

    def test_heterogeneous_subscripts_fallback(self):
        """Different subscripts fall through to per-contraction path without error."""
        torch.manual_seed(1)
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        results = trntensor.multi_einsum(
            ("ij,jk->ik", A, B),  # matmul → (4, 5)
            ("ij,ij->", A, A),  # frobenius inner product → scalar
        )
        assert len(results) == 2
        np.testing.assert_allclose(results[0].numpy(), (A @ B).numpy(), atol=1e-5)
        np.testing.assert_allclose(results[1].item(), (A * A).sum().item(), atol=1e-5)
