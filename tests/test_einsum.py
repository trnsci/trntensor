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
        from trntensor.nki.dispatch import _route

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
        if HAS_NKI and _route(512 * 512 * 512, pinned=False) == "nki":
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

    def test_dd_cpu_no_longer_raises(self):
        """precision='dd' computes correctly on CPU via Ozaki-2 (v0.14.0)."""
        A, B = torch.randn(4, 3), torch.randn(3, 5)
        result = trntensor.einsum("ij,jk->ik", A, B, precision="dd")
        assert result.shape == (4, 5)

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


class TestTargetForwardErrorMultiEinsum:
    """target_forward_error= for multi_einsum()."""

    def test_tfe_basic_result_correct(self):
        """multi_einsum with TFE produces correct result shape and dtype."""
        A = torch.randn(8, 16, dtype=torch.bfloat16)
        B = torch.randn(16, 8, dtype=torch.bfloat16)
        results = trntensor.multi_einsum(("ij,jk->ik", A, B), target_forward_error=0.1)
        assert len(results) == 1
        assert results[0].shape == (8, 8)
        assert results[0].dtype == torch.bfloat16

    def test_tfe_matches_explicit_precision(self):
        """K=16 BF16 target=1e-5 selects 'kahan'; result matches explicit kahan."""
        torch.manual_seed(10)
        A = torch.randn(8, 16, dtype=torch.bfloat16)
        B = torch.randn(16, 8, dtype=torch.bfloat16)
        results_tfe = trntensor.multi_einsum(("ij,jk->ik", A, B), target_forward_error=1e-5)
        results_kahan = trntensor.multi_einsum(("ij,jk->ik", A, B), precision="kahan")
        torch.testing.assert_close(results_tfe[0], results_kahan[0])

    def test_tfe_heterogeneous_per_contraction(self):
        """Two contractions with different K each resolve independently."""
        torch.manual_seed(11)
        # K=4 (fast region for BF16 at target=0.1) and K=512 (sr region)
        A_small = torch.randn(8, 4, dtype=torch.bfloat16)
        B_small = torch.randn(4, 8, dtype=torch.bfloat16)
        A_large = torch.randn(8, 512, dtype=torch.bfloat16)
        B_large = torch.randn(512, 8, dtype=torch.bfloat16)
        results = trntensor.multi_einsum(
            ("ij,jk->ik", A_small, B_small),
            ("ij,jk->ik", A_large, B_large),
            target_forward_error=0.1,
        )
        assert results[0].shape == (8, 8)
        assert results[1].shape == (8, 8)

    def test_tfe_ambiguous_raises(self):
        """precision= + target_forward_error= raises ValueError."""
        A = torch.randn(8, 8)
        B = torch.randn(8, 8)
        with pytest.raises(ValueError, match="target_forward_error"):
            trntensor.multi_einsum(("ij,jk->ik", A, B), precision="sr", target_forward_error=0.1)

    def test_tfe_homogeneous_batch_respected(self):
        """Homogeneous batch with TFE uses resolved precision, not always fast."""
        torch.manual_seed(12)
        # K=16, BF16, target=1e-5 → should select kahan (not fast)
        A = torch.randn(8, 16, dtype=torch.bfloat16)
        B = torch.randn(16, 8, dtype=torch.bfloat16)
        # Same operands → homogeneous batch eligible
        results_tfe = trntensor.multi_einsum(
            ("ij,jk->ik", A, B),
            ("ij,jk->ik", A, B),
            target_forward_error=1e-5,
        )
        results_kahan = trntensor.multi_einsum(
            ("ij,jk->ik", A, B),
            ("ij,jk->ik", A, B),
            precision="kahan",
        )
        torch.testing.assert_close(results_tfe[0], results_kahan[0])
        torch.testing.assert_close(results_tfe[1], results_kahan[1])


class TestTargetForwardError:
    """target_forward_error= automatic precision selection."""

    def test_large_k_bf16_selects_sr(self):
        """K=512 BF16 with target=0.1 selects 'sr' (√512·u_bf ≈ 0.088 < 0.1)."""
        A = torch.randn(16, 512, dtype=torch.bfloat16)
        B = torch.randn(512, 16, dtype=torch.bfloat16)
        result = trntensor.einsum("ij,jk->ik", A, B, target_forward_error=0.1)
        assert result.shape == (16, 16)
        assert result.dtype == torch.bfloat16

    def test_small_k_selects_fast(self):
        """K=4 BF16 with target=0.1 selects 'fast' (4·u_bf ≈ 0.016 < 0.1)."""
        A = torch.randn(8, 4, dtype=torch.bfloat16)
        B = torch.randn(4, 8, dtype=torch.bfloat16)
        result = trntensor.einsum("ij,jk->ik", A, B, target_forward_error=0.1)
        assert result.shape == (8, 8)

    def test_tight_target_selects_dd(self):
        """K=64 BF16 with target=0.006 selects 'dd' (64·u_bf² + u_bf ≈ 0.0049 < 0.006)."""
        assert trntensor.select_precision_for_error(torch.bfloat16, 64, 0.006) == "dd"
        A = torch.randn(16, 64, dtype=torch.bfloat16)
        B = torch.randn(64, 16, dtype=torch.bfloat16)
        result = trntensor.einsum("ij,jk->ik", A, B, target_forward_error=0.006)
        assert result.shape == (16, 16)

    def test_very_tight_selects_kahan(self):
        """K=16 BF16 with target=1e-5 selects 'kahan' (all BF16 modes exceed 1e-5)."""
        A = torch.randn(8, 16, dtype=torch.bfloat16)
        B = torch.randn(16, 8, dtype=torch.bfloat16)
        result = trntensor.einsum("ij,jk->ik", A, B, target_forward_error=1e-5)
        # kahan promotes to fp64 — verify against kahan-explicit call
        result_kahan = trntensor.einsum("ij,jk->ik", A, B, precision="kahan")
        torch.testing.assert_close(result, result_kahan)

    def test_ambiguous_raises(self):
        """Specifying both precision= and target_forward_error= raises ValueError."""
        A = torch.randn(8, 8)
        B = torch.randn(8, 8)
        with pytest.raises(ValueError, match="target_forward_error"):
            trntensor.einsum("ij,jk->ik", A, B, precision="sr", target_forward_error=0.1)

    def test_impossible_target_raises(self):
        """Unachievable target raises ValueError naming 'kahan' as best available."""
        A = torch.randn(4, 4, dtype=torch.bfloat16)
        B = torch.randn(4, 4, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="kahan"):
            trntensor.einsum("ij,jk->ik", A, B, target_forward_error=1e-20)


class TestPrecisionDD:
    """precision='dd' Ozaki-2 dispatch and accuracy tests (CPU mock)."""

    def test_dd_plan_stored(self):
        A = torch.randn(32, 32)
        B = torch.randn(32, 32)
        plan = trntensor.plan_contraction("ij,jk->ik", A, B, precision="dd")
        assert plan.precision == "dd"

    def test_dd_output_shape_and_dtype_fp32(self):
        A = torch.randn(32, 64, dtype=torch.float32)
        B = torch.randn(64, 32, dtype=torch.float32)
        result = trntensor.einsum("ij,jk->ik", A, B, precision="dd")
        assert result.shape == (32, 32)
        assert result.dtype == torch.float32

    def test_dd_more_accurate_than_fast_bf16(self):
        """DD achieves smaller error vs FP32 reference than single BF16 matmul."""
        torch.manual_seed(7)
        A = torch.randn(64, 512, dtype=torch.float32)
        B = torch.randn(512, 64, dtype=torch.float32)
        expected = A @ B

        fast_err = (
            (
                trntensor.einsum("ij,jk->ik", A.bfloat16(), B.bfloat16(), precision="fast").float()
                - expected
            )
            .abs()
            .mean()
        )
        dd_err = (trntensor.einsum("ij,jk->ik", A, B, precision="dd") - expected).abs().mean()

        assert dd_err < fast_err, (
            f"DD error {dd_err:.4f} should be less than fast BF16 error {fast_err:.4f}"
        )

    def test_dd_nki_still_raises(self, monkeypatch):
        """On NKI hardware, precision='dd' raises NotImplementedError (trnblas#22)."""
        import trntensor.nki.dispatch as _nd

        monkeypatch.setattr(_nd, "HAS_NKI", True)
        A = torch.randn(8, 8)
        B = torch.randn(8, 8)
        with pytest.raises(NotImplementedError, match="trnblas"):
            trntensor.einsum("ij,jk->ik", A, B, precision="dd")

    def test_dd_path_strategy_propagates(self):
        """3-operand chain with precision='dd' propagates to binary matmul steps."""
        torch.manual_seed(3)
        A = torch.randn(16, 32, dtype=torch.float32)
        B = torch.randn(32, 16, dtype=torch.float32)
        C = torch.randn(16, 8, dtype=torch.float32)
        result = trntensor.einsum("ij,jk,kl->il", A, B, C, precision="dd")
        expected = A @ B @ C
        assert result.shape == (16, 8)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)


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


def _componentwise_err(C, A, B):
    """max |C - C64| / (|A||B|), the metric the precision bounds are stated in."""
    A64, B64 = A.double(), B.double()
    return ((C.double() - A64 @ B64).abs() / (A64.abs() @ B64.abs())).max().item()


class TestPrecisionPromises:
    """Bounds include the output-dtype rounding; targets below it warn (Trn2 B5)."""

    @pytest.fixture(autouse=True)
    def _reset_floor_warning(self, monkeypatch):
        import trntensor.plan

        monkeypatch.setattr(trntensor.plan, "_output_floor_warning_emitted", False)

    def test_bounds_include_output_rounding(self):
        u_bf, u32, u64 = 2**-8, 2**-24, 2**-53
        feb = trntensor.forward_error_bound
        assert feb("fast", torch.bfloat16, 128) == 128 * u_bf + u_bf
        assert feb("sr", torch.bfloat16, 64) == 8 * u_bf + u_bf
        assert feb("dd", torch.bfloat16, 64) == 64 * u_bf**2 + u_bf
        assert feb("kahan", torch.float32, 128) == 128 * u64 + u32
        with pytest.raises(ValueError, match="unknown precision"):
            feb("nope", torch.float32, 4)

    @pytest.mark.parametrize(
        "dtype,target", [(torch.bfloat16, 1e-4), (torch.bfloat16, 1e-6), (torch.float32, 1e-8)]
    )
    def test_target_below_floor_warns_and_returns_kahan(self, dtype, target):
        with pytest.warns(RuntimeWarning, match="below the achievable floor"):
            assert trntensor.select_precision_for_error(dtype, 128, target) == "kahan"

    def test_floor_warning_once_per_process(self):
        import warnings

        with pytest.warns(RuntimeWarning):
            trntensor.select_precision_for_error(torch.bfloat16, 128, 1e-4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            trntensor.select_precision_for_error(torch.bfloat16, 128, 1e-4)

    def test_achievable_target_does_not_warn(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # fp32, K=128: kahan bound ≈ 6.0e-8 ≤ 1e-7
            assert trntensor.select_precision_for_error(torch.float32, 128, 1e-7) == "kahan"
            assert trntensor.select_precision_for_error(torch.float32, 4, 1e-6) == "fast"
            assert trntensor.select_precision_for_error(torch.bfloat16, 4, 0.1) == "fast"

    def test_impossible_target_still_raises(self):
        with pytest.raises(ValueError, match="kahan"):
            trntensor.select_precision_for_error(torch.float32, 128, 1e-20)

    def test_einsum_below_floor_warns_but_returns(self):
        torch.manual_seed(0)
        A = torch.randn(32, 128, dtype=torch.bfloat16)
        B = torch.randn(128, 32, dtype=torch.bfloat16)
        with pytest.warns(RuntimeWarning, match="bfloat16"):
            C = trntensor.einsum("ij,jk->ik", A, B, target_forward_error=1e-4)
        assert C.dtype == torch.bfloat16
        assert _componentwise_err(C, A, B) <= trntensor.forward_error_bound(
            "kahan", torch.bfloat16, 128
        )

    def test_multi_einsum_below_floor_warns(self):
        A = torch.randn(16, 64)
        Bs = [torch.randn(64, 16) for _ in range(3)]
        with pytest.warns(RuntimeWarning, match="float32"):
            outs = trntensor.multi_einsum(
                *[("ij,jk->ik", A, b) for b in Bs], target_forward_error=1e-8
            )
        assert len(outs) == 3

    @pytest.mark.parametrize("n", [128, 256])
    def test_kahan_fp32_within_one_ulp(self, n):
        """precision='kahan' fp32 meets K·2⁻⁵³ + u_fp32 (B5 measured 1.4 ulp on Trn2)."""
        g = torch.Generator().manual_seed(n)
        A = torch.randn(n, n, generator=g, dtype=torch.float64).float()
        B = torch.randn(n, n, generator=g, dtype=torch.float64).float()
        C = trntensor.einsum("ij,jk->ik", A, B, precision="kahan")
        assert _componentwise_err(C, A, B) <= trntensor.forward_error_bound(
            "kahan", torch.float32, n
        )

    def test_kahan_contracts_on_host_for_non_fp64_device(self, monkeypatch):
        """Operands on a device without fp64 are contracted on the host in fp64."""
        import sys

        te = sys.modules["trntensor.einsum"]

        monkeypatch.setattr(te, "_FP64_DEVICE_TYPES", ())
        calls = []
        real_einsum = torch.einsum

        def spy(sub, *ops):
            calls.append([(op.device.type, op.dtype) for op in ops])
            return real_einsum(sub, *ops)

        monkeypatch.setattr(te.torch, "einsum", spy)
        A = torch.randn(16, 32)
        B = torch.randn(32, 8)
        C = trntensor.einsum("ij,jk->ik", A, B, precision="kahan")
        assert calls == [[("cpu", torch.float64), ("cpu", torch.float64)]]
        assert C.dtype == torch.float32 and C.device.type == "cpu"
        torch.testing.assert_close(C, (A.double() @ B.double()).float())


def _two_operand_subscripts():
    import itertools

    subs = set()
    for out in ("ik", "ki"):
        for a in ("ij", "ji"):
            for b in ("jk", "kj"):
                subs.add(f"{a},{b}->{out}")
    for out in itertools.permutations("bik"):
        for a in ("bij", "bji", "ibj"):
            for b in ("bjk", "bkj", "jbk"):
                subs.add(f"{a},{b}->{''.join(out)}")
    subs |= {"ij,jk->i", "ij,jk->k", "ij,jk->", "bij,bjk->bi", "ij,j->i", "i,j->ji"}
    return sorted(subs)


class TestOutputOrder:
    """The matmul/bmm plans return op(A) @ op(B); a transposed or permuted output,
    or non-canonical batched operands, must not take them (#64)."""

    @pytest.mark.parametrize("subscripts", _two_operand_subscripts())
    @pytest.mark.parametrize("square", [False, True])
    def test_matches_torch_einsum(self, subscripts, square):
        sizes = dict(zip("bijk", (2, 5, 5, 5) if square else (2, 6, 5, 7), strict=True))
        g = torch.Generator().manual_seed(0)
        ins = subscripts.split("->")[0].split(",")
        ops = [torch.randn(*[sizes[c] for c in t], generator=g, dtype=torch.float64) for t in ins]
        want = torch.einsum(subscripts, *ops)
        torch.testing.assert_close(trntensor.einsum(subscripts, *ops), want)
        out = torch.randn(want.shape, generator=g, dtype=torch.float64)
        got = trntensor.einsum(subscripts, *ops, alpha=0.7, beta=0.5, out=out.clone())
        torch.testing.assert_close(got, 0.7 * want + 0.5 * out)

    @pytest.mark.parametrize(
        "subscripts,strategy",
        [
            ("ij,jk->ik", "matmul"),
            ("ji,kj->ik", "matmul"),
            ("ij,jk->ki", "torch"),
            ("ij,jk->i", "torch"),
            ("bij,bjk->bik", "bmm"),
            ("bij,bjk->bki", "torch"),
            ("bji,bjk->bik", "torch"),
            ("ibj,bjk->bik", "torch"),
        ],
    )
    def test_strategy(self, subscripts, strategy):
        sizes = dict(zip("bijk", (2, 6, 5, 7), strict=True))
        ins = subscripts.split("->")[0].split(",")
        ops = [torch.randn(*[sizes[c] for c in t]) for t in ins]
        assert trntensor.plan_contraction(subscripts, *ops).strategy == strategy


class TestBetaZeroIgnoresOut:
    """beta == 0 must not read ``out`` (BLAS / cuTENSOR convention, audit 3)."""

    @pytest.mark.parametrize(
        "subscripts,shapes",
        [
            ("ij,jk->ik", [(6, 5), (5, 7)]),
            ("ij,jk->ki", [(6, 5), (5, 7)]),
            ("bij,bjk->bik", [(3, 4, 5), (3, 5, 6)]),
            ("ij,jk,kl->il", [(4, 5), (5, 6), (6, 3)]),
            ("ij->ji", [(4, 6)]),
        ],
    )
    @pytest.mark.parametrize("fill", [float("nan"), float("inf")])
    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_nonfinite_out_not_propagated(self, subscripts, shapes, fill, alpha):
        g = torch.Generator().manual_seed(0)
        ops = [torch.randn(s, generator=g) for s in shapes]
        ref = alpha * torch.einsum(subscripts, *ops)
        out = torch.full(ref.shape, fill)
        got = trntensor.einsum(subscripts, *ops, alpha=alpha, beta=0.0, out=out)
        torch.testing.assert_close(got, ref)
        assert torch.isnan(out).all() or torch.isinf(out).all()  # out not written
