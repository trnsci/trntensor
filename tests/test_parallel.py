"""Tests for trntensor.parallel — ShardedTensor and output-parallel einsum.

All tests run on CPU (no hardware or NKI simulator required). The sharded
dispatch path is a pure Python loop over dense chunk tensors, so correctness
is verifiable against torch.einsum references.
"""

import numpy as np
import pytest
import torch

import trntensor
from trntensor import parallel
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
    """Argument errors on the sharded path."""

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

    def test_single_process_on_nki_host_uses_mock(self, monkeypatch):
        """HAS_NKI=True without a process group: mock all-reduce, correct result."""
        import trntensor.nki.dispatch as _nd

        monkeypatch.setattr(_nd, "HAS_NKI", True)
        monkeypatch.setenv("TRNTENSOR_FORCE_BACKEND", "pytorch")
        torch.manual_seed(15)
        A = torch.randn(8, 16)
        B = torch.randn(16, 6)
        A_sharded = scatter(A, dim=1, n_shards=4)

        result = trntensor.einsum("ij,jk->ik", A_sharded, B)
        torch.testing.assert_close(result, A @ B, atol=1e-5, rtol=1e-5)

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

    The inner reduce step uses the same backend selection as pure
    reduce-parallel (:func:`trntensor.parallel.allreduce_backend`).
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

    def test_single_process_on_nki_host_uses_mock(self, monkeypatch):
        """HAS_NKI=True without a process group: mixed path computes via mock."""
        import trntensor.nki.dispatch as _nd

        monkeypatch.setattr(_nd, "HAS_NKI", True)
        monkeypatch.setenv("TRNTENSOR_FORCE_BACKEND", "pytorch")
        A = torch.randn(8, 6)
        B = torch.randn(6, 4)
        result = trntensor.einsum(
            "ij,jk->ik", scatter(A, dim=0, n_shards=2), scatter(B, dim=0, n_shards=2)
        )
        torch.testing.assert_close(gather(result), A @ B, atol=1e-5, rtol=1e-5)

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


class TestAllreduceBackendSelection:
    """``allreduce_backend``: automatic choice plus TRNTENSOR_ALLREDUCE override."""

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.delenv("TRNTENSOR_ALLREDUCE", raising=False)
        parallel.set_allreduce_backend(None)
        yield
        parallel.set_allreduce_backend(None)

    def _pg(self, monkeypatch, pg):
        monkeypatch.setattr(parallel, "_process_group", lambda: pg)

    def test_auto_no_group_is_mock(self, monkeypatch):
        self._pg(monkeypatch, None)
        assert parallel.allreduce_backend(4) == "mock"

    def test_auto_world1_is_mock(self, monkeypatch):
        self._pg(monkeypatch, (0, 1, "neuron"))
        assert parallel.allreduce_backend(1) == "mock"

    def test_auto_neuron_group_is_nki(self, monkeypatch):
        self._pg(monkeypatch, (1, 4, "neuron"))
        assert parallel.allreduce_backend(4) == "nki"

    def test_auto_gloo_group_is_torch(self, monkeypatch):
        self._pg(monkeypatch, (0, 2, "gloo"))
        assert parallel.allreduce_backend(2) == "torch"

    def test_auto_shard_count_mismatch_is_mock(self, monkeypatch):
        self._pg(monkeypatch, (0, 2, "neuron"))
        assert parallel.allreduce_backend(3) == "mock"

    def test_env_override(self, monkeypatch):
        self._pg(monkeypatch, (0, 2, "neuron"))
        monkeypatch.setenv("TRNTENSOR_ALLREDUCE", "torch")
        assert parallel.allreduce_backend(2) == "torch"
        monkeypatch.setenv("TRNTENSOR_ALLREDUCE", "mock")
        assert parallel.allreduce_backend(2) == "mock"

    def test_setter_beats_env(self, monkeypatch):
        self._pg(monkeypatch, (0, 2, "neuron"))
        monkeypatch.setenv("TRNTENSOR_ALLREDUCE", "torch")
        parallel.set_allreduce_backend("mock")
        assert parallel.allreduce_backend(2) == "mock"

    def test_explicit_without_group_raises(self, monkeypatch):
        self._pg(monkeypatch, None)
        monkeypatch.setenv("TRNTENSOR_ALLREDUCE", "torch")
        with pytest.raises(RuntimeError, match="process group"):
            parallel.allreduce_backend(2)

    def test_explicit_mismatch_raises(self, monkeypatch):
        self._pg(monkeypatch, (0, 2, "gloo"))
        monkeypatch.setenv("TRNTENSOR_ALLREDUCE", "torch")
        with pytest.raises(RuntimeError, match="n_shards=4"):
            parallel.allreduce_backend(4)

    def test_nki_needs_neuron_group(self, monkeypatch):
        self._pg(monkeypatch, (0, 2, "gloo"))
        monkeypatch.setenv("TRNTENSOR_ALLREDUCE", "nki")
        with pytest.raises(RuntimeError, match="neuron"):
            parallel.allreduce_backend(2)

    def test_bad_env_value_raises(self, monkeypatch):
        monkeypatch.setenv("TRNTENSOR_ALLREDUCE", "nccl")
        with pytest.raises(ValueError, match="TRNTENSOR_ALLREDUCE"):
            parallel.allreduce_backend(2)

    def test_bad_setter_value_raises(self):
        with pytest.raises(ValueError):
            parallel.set_allreduce_backend("ring")

    def test_rank_local_computes_only_own_partial(self, monkeypatch):
        """Under a rank-local backend, reduce_partials computes shard ``rank`` only."""
        import torch.distributed as dist

        self._pg(monkeypatch, (1, 3, "gloo"))
        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        monkeypatch.setattr(parallel, "_dist_allreduce", lambda t: t * 10)
        calls = []

        def partial(i):
            calls.append(i)
            return torch.full((2, 2), float(i))

        out = parallel.reduce_partials(partial, 3)
        assert calls == [1]
        torch.testing.assert_close(out, torch.full((2, 2), 10.0))


class TestMultiRankGloo:
    """Real 2-rank reduce-parallel einsum on CPU (gloo) via the multi-rank runner."""

    def test_runner_world2_cpu(self, tmp_path):
        import os
        import subprocess
        import sys

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(root, "scripts", "run_reduce_parallel_multirank.py")
        proc = subprocess.run(
            [
                sys.executable,
                script,
                "--cpu",
                "--world",
                "2",
                "--sizes",
                "64",
                "--reps",
                "1",
                "--modes",
                "auto",
                "--job-timeout",
                "120",
                "--logdir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        assert "RESULT [w2/auto/n=64]: PASS" in proc.stdout
        assert "backend=torch" in proc.stdout
