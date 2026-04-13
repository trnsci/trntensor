"""Test einsum and contraction planning."""

import pytest
import torch
import numpy as np
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
        D = torch.randn(3, 2)
        results = trntensor.multi_einsum(
            ("ij,jk,kl->il", A, B, C),
            ("ij,ij->", A, A),
        )
        assert len(results) == 2
        np.testing.assert_allclose(results[0].numpy(), (A @ B @ C).numpy(), atol=1e-4)
        np.testing.assert_allclose(results[1].item(), (A * A).sum().item(), atol=1e-4)


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
