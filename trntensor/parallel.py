"""
Sharded tensor abstraction for output-parallel contractions.

Phase 4 foundation: ``ShardedTensor`` wraps a list of dense chunk tensors
split along a single partition dimension. ``einsum()`` detects ShardedTensor
operands and dispatches each shard independently, enabling output-parallel
contractions where each shard's result is fully independent (no cross-shard
communication required).

Reduce-parallel sharding (contracting the sharded dimension across cores)
requires all-reduce collective operations (``nki.collectives``). This is
blocked on Neuron SDK 2.30+; those patterns raise ``NotImplementedError``
until then with a clear upgrade note.

Usage::

    from trntensor import einsum
    from trntensor.parallel import ShardedTensor, scatter, gather

    # Scatter a dense tensor across 4 shards along dim 0
    A_sharded = scatter(A, dim=0, n_shards=4)

    # Einsum over a sharded operand (output-parallel — no collectives)
    C_sharded = einsum("ij,jk->ik", A_sharded, B)

    # Reconstruct the dense result
    C = gather(C_sharded)
"""

from __future__ import annotations

import torch


class ShardedTensor:
    """A tensor logically split across ``n_shards`` along ``partition_dim``.

    Wraps a list of dense chunk tensors, each being a contiguous slice of
    the original along ``partition_dim``. The full (gathered) tensor is
    recovered by ``torch.cat(chunks, dim=partition_dim)``.

    The class carries enough metadata for ``einsum()`` to:
    - Detect that an operand is sharded
    - Identify which subscript index corresponds to the partition dimension
    - Classify the sharding as output-parallel or reduce-parallel
    - Dispatch each shard's contraction independently

    Use :func:`scatter` to create from a dense tensor and :func:`gather`
    to reconstruct.
    """

    def __init__(self, chunks: list[torch.Tensor], partition_dim: int) -> None:
        if not chunks:
            raise ValueError("chunks must be non-empty")
        if partition_dim < 0 or partition_dim >= chunks[0].ndim:
            raise ValueError(
                f"partition_dim {partition_dim} out of range for chunks with ndim {chunks[0].ndim}"
            )
        self.chunks = list(chunks)
        self.partition_dim = partition_dim

    @property
    def n_shards(self) -> int:
        """Number of shards."""
        return len(self.chunks)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype shared by all chunks."""
        return self.chunks[0].dtype

    @property
    def device(self) -> torch.device:
        """Device shared by all chunks."""
        return self.chunks[0].device

    @property
    def ndim(self) -> int:
        """Number of dimensions (same as a gathered dense tensor)."""
        return self.chunks[0].ndim

    @property
    def shape(self) -> torch.Size:
        """Full logical shape (as if gathered)."""
        full = list(self.chunks[0].shape)
        full[self.partition_dim] = sum(c.shape[self.partition_dim] for c in self.chunks)
        return torch.Size(full)

    def __repr__(self) -> str:
        return (
            f"ShardedTensor(shape={tuple(self.shape)}, "
            f"partition_dim={self.partition_dim}, "
            f"n_shards={self.n_shards}, "
            f"dtype={self.dtype})"
        )


def scatter(tensor: torch.Tensor, dim: int, n_shards: int) -> ShardedTensor:
    """Split a dense tensor into ``n_shards`` along ``dim``.

    Divides ``tensor.shape[dim]`` as evenly as possible. The last shard
    absorbs any remainder when ``tensor.shape[dim]`` is not divisible by
    ``n_shards``.

    Args:
        tensor: Dense tensor to split.
        dim: Dimension along which to split.
        n_shards: Number of shards.

    Returns:
        :class:`ShardedTensor` with ``n_shards`` chunks.

    Raises:
        ValueError: If ``n_shards < 1`` or ``dim`` is out of range, or
            ``n_shards`` exceeds the tensor's size along ``dim``.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError(f"dim {dim} out of range for tensor with ndim {tensor.ndim}")
    size = tensor.shape[dim]
    if n_shards > size:
        raise ValueError(f"n_shards ({n_shards}) exceeds tensor size along dim {dim} ({size})")
    base = size // n_shards
    chunks = []
    start = 0
    for i in range(n_shards):
        end = start + base + (1 if i < size % n_shards else 0)
        idx = [slice(None)] * tensor.ndim
        idx[dim] = slice(start, end)
        chunks.append(tensor[tuple(idx)])
        start = end
    return ShardedTensor(chunks, partition_dim=dim)


def gather(st: ShardedTensor) -> torch.Tensor:
    """Reconstruct a dense tensor by concatenating shards along ``partition_dim``.

    Args:
        st: :class:`ShardedTensor` to gather.

    Returns:
        Dense tensor of full logical shape.
    """
    return torch.cat(st.chunks, dim=st.partition_dim)


def shard_boundaries(st: ShardedTensor) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs for each shard along ``partition_dim``.

    Used internally by the reduce-parallel dispatch path to co-partition
    non-sharded operands that share the contracted dimension.

    Args:
        st: :class:`ShardedTensor` to inspect.

    Returns:
        List of ``(start, end)`` pairs, one per shard, where ``end - start``
        equals ``chunk.shape[partition_dim]`` for that chunk.
    """
    bounds = []
    pos = 0
    for chunk in st.chunks:
        size = chunk.shape[st.partition_dim]
        bounds.append((pos, pos + size))
        pos += size
    return bounds


def _mock_allreduce(partials: list[torch.Tensor]) -> torch.Tensor:
    """Mock all-reduce for CPU testing: sum partial contractions across simulated shards.

    On Trainium, this operation maps to ``nki.collectives.allreduce`` (arriving
    in Neuron SDK 2.30+). This torch-based implementation enables reduce-parallel
    sharding to be tested on CPU without hardware.

    Args:
        partials: Per-shard partial contraction results, all of identical shape.

    Returns:
        Element-wise sum of all partials — the all-reduced result.
    """
    return torch.stack(partials).sum(dim=0)
