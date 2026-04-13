"""Planner-focused tests: subscript parsing, strategy selection, FLOP estimates."""

import pytest
import torch

import trntensor
from trntensor.plan import _parse_subscripts


class TestParseSubscripts:
    def test_explicit_output(self):
        assert _parse_subscripts("ij,jk->ik") == ("ij,jk", "ik")

    def test_implicit_output_matmul(self):
        # Implicit output = indices appearing exactly once, sorted
        input_str, output_str = _parse_subscripts("ij,jk")
        assert input_str == "ij,jk"
        assert output_str == "ik"

    def test_implicit_output_dot(self):
        # "i,i" → shared index drops out, output is empty
        _, output_str = _parse_subscripts("i,i")
        assert output_str == ""

    def test_implicit_output_outer(self):
        _, output_str = _parse_subscripts("i,j")
        assert output_str == "ij"


class TestPlanStrategy:
    def test_matmul_both_transposed(self):
        """'ji,kj->ik' — contracted 'j' is first in A and second in B → both transposed."""
        A = torch.randn(3, 4)
        B = torch.randn(5, 3)
        plan = trntensor.plan_contraction("ji,kj->ik", A, B)
        assert plan.strategy == "matmul"
        assert plan.transA is True
        assert plan.transB is True

    def test_matmul_neither_transposed(self):
        plan = trntensor.plan_contraction("ij,jk->ik", torch.randn(4, 3), torch.randn(3, 5))
        assert plan.strategy == "matmul"
        assert plan.transA is False
        assert plan.transB is False

    def test_three_operands_falls_to_torch(self):
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        C = torch.randn(5, 2)
        plan = trntensor.plan_contraction("ij,jk,kl->il", A, B, C)
        assert plan.strategy == "torch"

    def test_batched_matmul_metadata(self):
        plan = trntensor.plan_contraction(
            "bij,bjk->bik", torch.randn(2, 3, 4), torch.randn(2, 4, 5)
        )
        assert plan.strategy == "bmm"
        assert "j" in plan.contraction_indices
        assert "b" in plan.batch_indices

    def test_backend_reports_nki_or_pytorch(self):
        """`plan.backend` reports the executor that will run the contraction.

        On Neuron hosts, matmul/bmm strategies large enough to clear the
        dispatch-overhead threshold report `"nki"`; smaller matmuls and
        the torch fallback always report `"pytorch"`.
        """
        from trntensor.nki.dispatch import HAS_NKI

        # Large matmul — above _MIN_NKI_FLOPS (2 GFLOPs) on Neuron.
        plan_big = trntensor.plan_contraction(
            "ij,jk->ik", torch.randn(2048, 2048), torch.randn(2048, 2048)
        )
        # Small matmul — below threshold, falls back to PyTorch even on Neuron.
        plan_small = trntensor.plan_contraction("ij,jk->ik", torch.randn(4, 3), torch.randn(3, 5))
        # 3-operand contraction — never a NKI strategy.
        plan_torch = trntensor.plan_contraction(
            "ij,jk,kl->il", torch.randn(3, 4), torch.randn(4, 5), torch.randn(5, 2)
        )
        assert plan_big.backend == ("nki" if HAS_NKI else "pytorch")
        assert plan_small.backend == "pytorch"
        assert plan_torch.backend == "pytorch"


class TestEstimateFlops:
    def test_matmul_flops(self):
        A = torch.randn(10, 20)
        B = torch.randn(20, 30)
        assert trntensor.estimate_flops("ij,jk->ik", A, B) == 10 * 20 * 30

    def test_flops_scale_with_dims(self):
        """Doubling a contracted dim should double the FLOP count."""
        small = trntensor.estimate_flops("ij,jk->ik", torch.randn(4, 5), torch.randn(5, 6))
        big = trntensor.estimate_flops("ij,jk->ik", torch.randn(4, 10), torch.randn(10, 6))
        assert big == 2 * small

    def test_flops_batched(self):
        """Batched contraction: product includes batch dim."""
        flops = trntensor.estimate_flops("bij,bjk->bik", torch.randn(7, 3, 4), torch.randn(7, 4, 5))
        assert flops == 7 * 3 * 4 * 5


class TestExecute:
    def test_both_transposed_matmul(self):
        """Exercises the transA+transB branch of _execute_matmul."""
        import numpy as np

        A = torch.randn(3, 4)
        B = torch.randn(5, 3)
        result = trntensor.einsum("ji,kj->ik", A, B)
        expected = A.T @ B.T
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_degenerate_shared_index_length_one(self):
        """Contracted index of length 1: output shape preserved, values sensible."""
        import numpy as np

        A = torch.randn(4, 1)
        B = torch.randn(1, 5)
        result = trntensor.einsum("ij,jk->ik", A, B)
        expected = A @ B
        assert result.shape == (4, 5)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)
