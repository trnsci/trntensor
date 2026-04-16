"""Planner-focused tests: subscript parsing, strategy selection, FLOP estimates."""

import pytest
import torch

import trntensor
from trntensor.plan import _greedy_path_search, _parse_subscripts, _validate_subscripts


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

    def test_three_operands_uses_path_strategy(self):
        """3+ operand einsums now use greedy path search."""
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        C = torch.randn(5, 2)
        plan = trntensor.plan_contraction("ij,jk,kl->il", A, B, C)
        assert plan.strategy == "path"
        assert len(plan.contraction_path) == 2  # two binary steps for 3 operands

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
        # 3-operand contraction — "path" strategy; individual binary steps pick
        # their own backends, but the top-level plan.backend is "pytorch".
        plan_path = trntensor.plan_contraction(
            "ij,jk,kl->il", torch.randn(3, 4), torch.randn(4, 5), torch.randn(5, 2)
        )
        assert plan_big.backend == ("nki" if HAS_NKI else "pytorch")
        assert plan_small.backend == "pytorch"
        assert plan_path.backend == "pytorch"


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


class TestGreedyPathSearch:
    """Unit tests for _greedy_path_search and 3+ operand plan_contraction."""

    def test_three_operand_plan_has_path_strategy(self):
        A, B, C = torch.randn(3, 4), torch.randn(4, 5), torch.randn(5, 2)
        plan = trntensor.plan_contraction("ij,jk,kl->il", A, B, C)
        assert plan.strategy == "path"
        # Three operands → two binary steps
        assert len(plan.contraction_path) == 2

    def test_four_operand_plan_has_three_steps(self):
        A, B, C, D = (torch.randn(3, 4), torch.randn(4, 5), torch.randn(5, 6), torch.randn(6, 2))
        plan = trntensor.plan_contraction("ij,jk,kl,lm->im", A, B, C, D)
        assert plan.strategy == "path"
        assert len(plan.contraction_path) == 3

    def test_greedy_picks_cheaper_pair(self):
        """A(100,200) × B(200,5) × C(5,50):
        - Contract B@C first: 200*5*50 = 50k FLOPs → intermediate (200,50)
          then A@(BC): 100*200*50 = 1M → total 1.05M
        - Contract A@B first: 100*200*5 = 100k FLOPs → intermediate (100,5)
          then (100,5)@C: 100*5*50 = 25k → total 125k
        Per-step greedy compares: pair(0,1)=100k vs pair(1,2)=50k → picks (1,2) = B@C.
        """
        size_map = {"i": 100, "j": 200, "k": 5, "l": 50}
        path = _greedy_path_search(["ij", "jk", "kl"], "il", size_map)
        # Cheapest first step is (1,2) = B@C (50k FLOPs vs A@B's 100k)
        assert path[0] == (1, 2), f"expected (1,2) got {path[0]}"

    def test_greedy_picks_cheapest_when_rightmost_cheaper(self):
        """C(5,50) × D(50,3) is the cheapest pair in A(100,200)B(200,5)C(5,50)D(50,3).
        A@B = 100*200*5 = 100k, B@C = 200*5*50 = 50k, C@D = 5*50*3 = 750.
        Greedy should contract C@D first.
        """
        size_map = {"i": 100, "j": 200, "k": 5, "l": 50, "m": 3}
        path = _greedy_path_search(["ij", "jk", "kl", "lm"], "im", size_map)
        # Cheapest first pair should be (2,3) = C@D
        assert path[0] == (2, 3), f"expected (2,3) got {path[0]}"

    def test_path_length_matches_operand_count(self):
        """N operands → N-1 binary steps."""
        for n in range(3, 7):
            inputs = [f"i{k}i{k+1}" for k in range(n)]
            size_map = {f"i{k}": 4 for k in range(n + 1)}
            output = f"i0i{n}"
            path = _greedy_path_search(inputs, output, size_map)
            assert len(path) == n - 1


class TestValidation:
    """Tests for _validate_subscripts error messages (#26)."""

    def test_wrong_operand_count(self):
        """2 subscript terms but 3 operands → ValueError naming the mismatch."""
        with pytest.raises(ValueError, match="2 operand term"):
            _validate_subscripts(
                "ij,jk->ik", (torch.randn(4, 3), torch.randn(3, 5), torch.randn(5, 2))
            )

    def test_wrong_rank(self):
        """3-index term applied to a 2D operand → ValueError naming operand and term."""
        with pytest.raises(ValueError, match="operand 1"):
            _validate_subscripts("ij,jkl->ikl", (torch.randn(4, 3), torch.randn(3, 5)))

    def test_inconsistent_index_size(self):
        """Index 'j' is size 3 in A but size 5 in B → ValueError naming the index."""
        with pytest.raises(ValueError, match="index 'j'"):
            _validate_subscripts("ij,jk->ik", (torch.randn(4, 3), torch.randn(5, 6)))

    def test_invalid_characters(self):
        """Non-letter characters in subscript (other than , and ->) → ValueError."""
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_subscripts("i j,jk->ik", (torch.randn(4, 3), torch.randn(3, 5)))

    def test_valid_subscript_passes(self):
        """Correct subscript and shapes pass without error."""
        _validate_subscripts("ij,jk->ik", (torch.randn(4, 3), torch.randn(3, 5)))  # no raise

    def test_plan_contraction_raises_on_shape_mismatch(self):
        """plan_contraction integrates validation — shape error surfaces from there."""
        with pytest.raises(ValueError, match="index 'j'"):
            trntensor.plan_contraction("ij,jk->ik", torch.randn(4, 3), torch.randn(5, 6))
