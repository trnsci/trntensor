"""Tests for trntensor.parallel — ShardedTensor and output-parallel einsum.

All tests run on CPU (no hardware or NKI simulator required). The sharded
dispatch path is a pure Python loop over dense chunk tensors, so correctness
is verifiable against torch.einsum references.
"""

import numpy as np
import pytest
import torch

import trntensor
from trntensor.parallel import ShardedTensor, gather, scatter


class TestScatterGather:
    def test_roundtrip_even(self):
        """scatter then gather recovers the original tensor."""
        A = torch.randn(16, 8)
        st = scatter(A, dim=0, n_shards=4)
        assert len(st.chunks) == 4
        assert st.partition_dim == 0
        torch.testing.assert_close(gather(st), A)

    def test_roundtrip_dim1(self):
        """scatter along dim 1 round-trips."""
        A = torch.randn(5, 12)
        st = scatter(A, dim=1, n_shards=3)
        assert st.partition_dim == 1
        torch.testing.assert_close(gather(st), A)

    def test_uneven_split(self):
        """Size not divisible by n_shards — last shard absorbs remainder."""
        A = torch.randn(10, 4)
        st = scatter(A, dim=0, n_shards=3)
        sizes = [c.shape[0] for c in st.chunks]
        assert sum(sizes) == 10
        torch.testing.assert_close(gather(st), A)

    def test_single_shard(self):
        """n_shards=1 is a no-op split."""
        A = torch.randn(8, 4)
        st = scatter(A, dim=0, n_shards=1)
        assert len(st.chunks) == 1
        torch.testing.assert_close(gather(st), A)

    def test_logical_shape(self):
        """ShardedTensor.shape returns full logical shape."""
        A = torch.randn(16, 8)
        st = scatter(A, dim=0, n_shards=4)
        assert st.shape == A.shape

    def test_scatter_too_many_shards_raises(self):
        A = torch.randn(3, 4)
        with pytest.raises(ValueError, match="n_shards"):
            scatter(A, dim=0, n_shards=10)

    def test_scatter_invalid_dim_raises(self):
        A = torch.randn(4, 4)
        with pytest.raises(ValueError, match="dim"):
            scatter(A, dim=5, n_shards=2)


class TestShardedEinsum:
    """Output-parallel einsum via ShardedTensor operands."""

    def test_output_parallel_matmul(self):
        """'ij,jk->ik' sharded along i: row-parallel matmul."""
        torch.manual_seed(0)
        A = torch.randn(16, 8)
        B = torch.randn(8, 6)
        A_sharded = scatter(A, dim=0, n_shards=4)

        result = trntensor.einsum("ij,jk->ik", A_sharded, B)

        assert isinstance(result, ShardedTensor)
        assert result.partition_dim == 0
        torch.testing.assert_close(gather(result), A @ B, atol=1e-5, rtol=1e-5)

    def test_output_parallel_matmul_dim1(self):
        """'ij,jk->ik' sharded along k (B's free dim): column-parallel matmul."""
        torch.manual_seed(1)
        A = torch.randn(8, 6)
        B = torch.randn(6, 16)
        B_sharded = scatter(B, dim=1, n_shards=4)

        result = trntensor.einsum("ij,jk->ik", A, B_sharded)

        assert isinstance(result, ShardedTensor)
        assert result.partition_dim == 1  # 'k' is at output index 1 in "ik"
        torch.testing.assert_close(gather(result), A @ B, atol=1e-5, rtol=1e-5)

    def test_output_parallel_three_operand(self):
        """3-operand chain with A sharded along output dim."""
        torch.manual_seed(2)
        A = torch.randn(16, 8)
        B = torch.randn(8, 6)
        C = torch.randn(6, 4)
        A_sharded = scatter(A, dim=0, n_shards=4)

        result = trntensor.einsum("ij,jk,kl->il", A_sharded, B, C)
        expected = torch.einsum("ij,jk,kl->il", A, B, C)

        assert isinstance(result, ShardedTensor)
        torch.testing.assert_close(gather(result), expected, atol=1e-4, rtol=1e-4)

    def test_output_parallel_uneven_shards(self):
        """Works when tensor size is not divisible by n_shards."""
        torch.manual_seed(3)
        A = torch.randn(11, 8)
        B = torch.randn(8, 5)
        A_sharded = scatter(A, dim=0, n_shards=3)

        result = trntensor.einsum("ij,jk->ik", A_sharded, B)
        torch.testing.assert_close(gather(result), A @ B, atol=1e-5, rtol=1e-5)

    def test_alpha_passes_through(self):
        """alpha scalar is applied per shard."""
        torch.manual_seed(4)
        A = torch.randn(8, 4)
        B = torch.randn(4, 6)
        A_sharded = scatter(A, dim=0, n_shards=2)

        result_scaled = trntensor.einsum("ij,jk->ik", A_sharded, B, alpha=2.0)
        result_plain = trntensor.einsum("ij,jk->ik", A_sharded, B)

        torch.testing.assert_close(gather(result_scaled), 2.0 * gather(result_plain))

    def test_gather_result_matches_dense_einsum(self):
        """gather(sharded_einsum(...)) == dense einsum for various shapes."""
        torch.manual_seed(5)
        for M, K, N, n_shards in [(12, 8, 5, 3), (20, 10, 7, 4), (16, 16, 16, 8)]:
            A = torch.randn(M, K)
            B = torch.randn(K, N)
            A_sharded = scatter(A, dim=0, n_shards=n_shards)
            result = trntensor.einsum("ij,jk->ik", A_sharded, B)
            np.testing.assert_allclose(gather(result).numpy(), (A @ B).numpy(), atol=1e-5)


class TestReduceParallelError:
    """Reduce-parallel sharding raises NotImplementedError on NKI hardware (HAS_NKI=True).

    On CPU (HAS_NKI=False) the mock-allreduce path activates — those cases are
    covered by :class:`TestReduceParallelMock` below.
    """

    def test_contracted_dim_raises_on_nki(self, monkeypatch):
        """On NKI hardware, sharding along the contracted index raises with SDK note."""
        import trntensor.nki.dispatch as _nd

        monkeypatch.setattr(_nd, "HAS_NKI", True)
        A = torch.randn(8, 16)
        B = torch.randn(16, 6)
        A_sharded = scatter(A, dim=1, n_shards=4)  # shard along j (contracted in "ij,jk->ik")

        with pytest.raises(NotImplementedError, match="nki.collectives"):
            trntensor.einsum("ij,jk->ik", A_sharded, B)

    def test_error_message_names_dimension_on_nki(self, monkeypatch):
        """On NKI hardware, error message names the contracted dimension character."""
        import trntensor.nki.dispatch as _nd

        monkeypatch.setattr(_nd, "HAS_NKI", True)
        A = torch.randn(8, 12)
        B = torch.randn(12, 5)
        A_sharded = scatter(A, dim=1, n_shards=3)  # shard along 'j'

        with pytest.raises(NotImplementedError, match="'j'"):
            trntensor.einsum("ij,jk->ik", A_sharded, B)

    def test_out_with_sharded_raises(self):
        """The 'out' accumulation argument is not supported with ShardedTensor."""
        A = torch.randn(8, 4)
        B = torch.randn(4, 6)
        A_sharded = scatter(A, dim=0, n_shards=2)
        out = torch.zeros(8, 6)

        with pytest.raises(NotImplementedError, match="out"):
            trntensor.einsum("ij,jk->ik", A_sharded, B, out=out)


class TestReduceParallelMock:
    """CPU mock-allreduce path for reduce-parallel sharding (HAS_NKI=False).

    On CPU, contracting across a sharded dimension uses :func:`_mock_allreduce`
    (torch-based sum of per-shard partial results) in place of the
    ``nki.collectives.allreduce`` that will be used on Trainium (SDK 2.30+).
    """

    def test_basic_contracted_dim(self):
        """'ij,jk->ik' with A sharded along j (contracted) — mock allreduce."""
        torch.manual_seed(10)
        A = torch.randn(8, 16)
        B = torch.randn(16, 6)
        A_sharded = scatter(A, dim=1, n_shards=4)

        result = trntensor.einsum("ij,jk->ik", A_sharded, B)

        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, A @ B, atol=1e-5, rtol=1e-5)

    def test_b_copartitioned(self):
        """Both operands sharded along j: B is co-partitioned automatically."""
        torch.manual_seed(11)
        A = torch.randn(8, 12)
        B = torch.randn(12, 5)
        A_sharded = scatter(A, dim=1, n_shards=3)
        B_sharded = scatter(B, dim=0, n_shards=3)

        result = trntensor.einsum("ij,jk->ik", A_sharded, B_sharded)

        torch.testing.assert_close(result, A @ B, atol=1e-5, rtol=1e-5)

    def test_uneven_shards(self):
        """Size not divisible by n_shards — last shard absorbs remainder."""
        torch.manual_seed(12)
        A = torch.randn(7, 11)
        B = torch.randn(11, 5)
        A_sharded = scatter(A, dim=1, n_shards=3)

        result = trntensor.einsum("ij,jk->ik", A_sharded, B)

        torch.testing.assert_close(result, A @ B, atol=1e-5, rtol=1e-5)

    def test_three_operand_chain(self):
        """'ij,jk,kl->il' with A sharded along j — B is co-partitioned, C is passed full."""
        torch.manual_seed(13)
        A = torch.randn(8, 12)
        B = torch.randn(12, 6)
        C = torch.randn(6, 4)
        A_sharded = scatter(A, dim=1, n_shards=3)

        result = trntensor.einsum("ij,jk,kl->il", A_sharded, B, C)
        expected = torch.einsum("ij,jk,kl->il", A, B, C)

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_nki_path_still_raises(self, monkeypatch):
        """When HAS_NKI=True, reduce-parallel raises NotImplementedError (not mock)."""
        import trntensor.nki.dispatch as _nd

        monkeypatch.setattr(_nd, "HAS_NKI", True)
        A = torch.randn(8, 16)
        B = torch.randn(16, 6)
        A_sharded = scatter(A, dim=1, n_shards=4)

        with pytest.raises(NotImplementedError, match="nki.collectives"):
            trntensor.einsum("ij,jk->ik", A_sharded, B)

    def test_mixed_cpu_no_longer_raises(self):
        """Mixed output+reduce-parallel computes correctly on CPU (v0.12.0)."""
        torch.manual_seed(14)
        A = torch.randn(12, 6)
        B = torch.randn(6, 8)
        C = torch.randn(8, 4)
        A_sharded = scatter(A, dim=0, n_shards=3)  # output-parallel: 'i' in output
        C_sharded = scatter(C, dim=0, n_shards=2)  # reduce-parallel: 'k' contracted

        result = trntensor.einsum("ij,jk,kl->il", A_sharded, B, C_sharded)
        expected = torch.einsum("ij,jk,kl->il", A, B, C)
        torch.testing.assert_close(gather(result), expected, atol=1e-4, rtol=1e-4)


class TestMixedParallel:
    """CPU mock path for mixed output-parallel + reduce-parallel sharding (v0.12.0).

    When an einsum has both a ShardedTensor partitioned along an output dimension
    (output-parallel) and a ShardedTensor partitioned along a contracted dimension
    (reduce-parallel), the dispatch uses a nested loop: for each output shard,
    accumulate partial contractions over all reduce shards via :func:`_mock_allreduce`.

    On Trainium (``HAS_NKI=True``) the reduce step raises ``NotImplementedError``
    — the same gate as pure reduce-parallel (``nki.collectives``, SDK 2.30+).
    """

    def test_basic_correctness(self):
        """'ij,jk->ik': A sharded along i (output), B sharded along j (reduce)."""
        torch.manual_seed(20)
        A = torch.randn(12, 8)
        B = torch.randn(8, 6)
        A_sh = scatter(A, dim=0, n_shards=3)  # output-parallel: 'i'
        B_sh = scatter(B, dim=0, n_shards=4)  # reduce-parallel: 'j'

        result = trntensor.einsum("ij,jk->ik", A_sh, B_sh)

        assert isinstance(result, ShardedTensor)
        assert result.partition_dim == 0
        torch.testing.assert_close(gather(result), A @ B, atol=1e-5, rtol=1e-5)

    def test_three_operand_case(self):
        """'ij,jk,kl->il': A sharded along i (output), C sharded along k (reduce)."""
        torch.manual_seed(21)
        A = torch.randn(12, 6)
        B = torch.randn(6, 8)
        C = torch.randn(8, 4)
        A_sh = scatter(A, dim=0, n_shards=3)  # output-parallel: 'i'
        C_sh = scatter(C, dim=0, n_shards=2)  # reduce-parallel: 'k'

        result = trntensor.einsum("ij,jk,kl->il", A_sh, B, C_sh)
        expected = torch.einsum("ij,jk,kl->il", A, B, C)

        assert isinstance(result, ShardedTensor)
        torch.testing.assert_close(gather(result), expected, atol=1e-4, rtol=1e-4)

    def test_output_shard_count(self):
        """result.n_shards matches the output-parallel operand's n_shards."""
        A_sh = scatter(torch.randn(12, 6), dim=0, n_shards=4)
        B_sh = scatter(torch.randn(6, 8), dim=0, n_shards=3)
        result = trntensor.einsum("ij,jk->ik", A_sh, B_sh)
        assert result.n_shards == 4

    def test_uneven_shards(self):
        """Non-divisible sizes work for both output and reduce shards."""
        torch.manual_seed(22)
        A = torch.randn(11, 7)
        B = torch.randn(7, 5)
        result = trntensor.einsum(
            "ij,jk->ik",
            scatter(A, dim=0, n_shards=3),
            scatter(B, dim=0, n_shards=2),
        )
        torch.testing.assert_close(gather(result), A @ B, atol=1e-5, rtol=1e-5)

    def test_nki_raises(self, monkeypatch):
        """On NKI hardware, mixed sharding raises NotImplementedError (nki.collectives)."""
        import trntensor.nki.dispatch as _nd

        monkeypatch.setattr(_nd, "HAS_NKI", True)
        A_sh = scatter(torch.randn(8, 6), dim=0, n_shards=2)
        B_sh = scatter(torch.randn(6, 4), dim=0, n_shards=2)

        with pytest.raises(NotImplementedError, match="nki.collectives"):
            trntensor.einsum("ij,jk->ik", A_sh, B_sh)

    def test_alpha_threads_through(self):
        """alpha kwarg is applied correctly in the mixed path."""
        torch.manual_seed(23)
        A = torch.randn(8, 6)
        B = torch.randn(6, 4)
        A_sh = scatter(A, dim=0, n_shards=2)
        B_sh = scatter(B, dim=0, n_shards=2)

        result_scaled = trntensor.einsum("ij,jk->ik", A_sh, B_sh, alpha=3.0)
        result_plain = trntensor.einsum("ij,jk->ik", A_sh, B_sh)

        torch.testing.assert_close(gather(result_scaled), 3.0 * gather(result_plain))
