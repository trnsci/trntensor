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
    """Reduce-parallel sharding must raise NotImplementedError."""

    def test_contracted_dim_raises(self):
        """Sharding along the contracted index requires all-reduce → error."""
        A = torch.randn(8, 16)
        B = torch.randn(16, 6)
        A_sharded = scatter(A, dim=1, n_shards=4)  # shard along j (contracted in "ij,jk->ik")

        with pytest.raises(NotImplementedError, match="nki.collectives"):
            trntensor.einsum("ij,jk->ik", A_sharded, B)

    def test_error_message_names_dimension(self):
        """Error message names the contracted dimension character."""
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
