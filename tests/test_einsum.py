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
