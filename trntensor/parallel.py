"""
Sharded tensor abstraction for output-parallel contractions.

Phase 4 foundation: ``ShardedTensor`` wraps a list of dense chunk tensors
split along a single partition dimension. ``einsum()`` detects ShardedTensor
operands and dispatches each shard independently, enabling output-parallel
contractions where each shard's result is fully independent (no cross-shard
communication required).

Reduce-parallel sharding (contracting the sharded dimension) produces one
partial result per shard; the partials are summed by an all-reduce. The
backend is chosen by :func:`allreduce_backend`:

- ``"mock"``: single process. Every shard's partial is computed locally and
  summed with :func:`_mock_allreduce` (a torch sum). This is the CPU path and
  the fallback when no multi-rank process group is active.
- ``"torch"``: multi-rank. ``torch.distributed`` is initialized with
  ``world_size > 1`` and ``n_shards == world_size``. Rank ``r`` computes only
  shard ``r``'s partial on its own device and the partials are combined with
  ``torch.distributed.all_reduce(SUM)``. Works for any c10d backend (gloo,
  TorchNeuron's ``"neuron"``).
- ``"nki"``: same rank-local compute, combined by an NKI kernel calling
  ``nki.collectives.all_reduce(op=add)`` on ``shared_hbm`` buffers (see
  :mod:`trntensor.nki.collectives`). Needs TorchNeuron and a ``"neuron"``
  process group.

``auto`` (the default) picks ``"nki"`` under a ``"neuron"`` process group,
``"torch"`` under any other multi-rank group, and ``"mock"`` otherwise. Set
``TRNTENSOR_ALLREDUCE=mock|torch|nki|auto`` to override, or call
:func:`set_allreduce_backend`. A rank-local backend still accepts a
ShardedTensor holding every chunk; only chunk ``rank`` is read, so the other
chunks may stay on the host.

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

import os
from collections.abc import Callable

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

    Used when no multi-rank process group is active (see
    :func:`allreduce_backend`). All partials live in this process; the sum is
    a single ``torch.stack(...).sum(0)``.

    Args:
        partials: Per-shard partial contraction results, all of identical shape.

    Returns:
        Element-wise sum of all partials — the all-reduced result.
    """
    return torch.stack(partials).sum(dim=0)


# --------------------------------------------------------------------------- #
# All-reduce backend selection for reduce-parallel contractions                 #
# --------------------------------------------------------------------------- #

ALLREDUCE_ENV_VAR = "TRNTENSOR_ALLREDUCE"
_ALLREDUCE_CHOICES = ("auto", "mock", "torch", "nki")
_allreduce_override: str | None = None


def set_allreduce_backend(backend: str | None) -> None:
    """Force the reduce-parallel all-reduce backend for this process.

    Args:
        backend: ``"auto"``, ``"mock"``, ``"torch"``, ``"nki"``, or ``None``
            to clear the override (then ``TRNTENSOR_ALLREDUCE`` or ``auto``
            applies). Takes precedence over the environment variable.
    """
    global _allreduce_override
    if backend is not None and backend not in _ALLREDUCE_CHOICES:
        raise ValueError(f"allreduce backend must be one of {_ALLREDUCE_CHOICES}, got {backend!r}")
    _allreduce_override = backend


def _requested_allreduce() -> str:
    if _allreduce_override is not None:
        return _allreduce_override
    req = os.environ.get(ALLREDUCE_ENV_VAR, "auto").strip().lower() or "auto"
    if req not in _ALLREDUCE_CHOICES:
        raise ValueError(f"{ALLREDUCE_ENV_VAR}={req!r}; expected one of {_ALLREDUCE_CHOICES}")
    return req


def _process_group() -> tuple[int, int, str] | None:
    """``(rank, world_size, backend)`` of the default process group, or None."""
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch built without distributed
        return None
    if not (dist.is_available() and dist.is_initialized()):
        return None
    return dist.get_rank(), dist.get_world_size(), str(dist.get_backend()).lower()


def allreduce_backend(n_shards: int) -> str:
    """Resolve the all-reduce backend for a reduce-parallel contraction.

    Rank-local backends (``"torch"``, ``"nki"``) need an initialized
    ``torch.distributed`` process group with ``world_size > 1`` equal to
    ``n_shards``: shard ``r`` belongs to rank ``r``. Under ``auto``, any other
    situation (no group, ``world_size == 1``, or a shard count that does not
    match the world) falls back to ``"mock"``, where each rank computes the
    full sum itself. An explicit ``"torch"``/``"nki"`` request that cannot be
    honored raises ``RuntimeError`` instead of silently running replicated.

    Args:
        n_shards: Number of reduce-parallel shards in the contraction.

    Returns:
        ``"mock"``, ``"torch"``, or ``"nki"``.
    """
    req = _requested_allreduce()
    if req == "mock":
        return "mock"
    pg = _process_group()
    usable = pg is not None and pg[1] > 1 and pg[1] == n_shards
    if req == "auto":
        if not usable:
            return "mock"
        return "nki" if pg[2] == "neuron" else "torch"
    if not usable:
        state = (
            "no torch.distributed process group is initialized"
            if pg is None
            else f"world_size={pg[1]} but the contraction has n_shards={n_shards}"
        )
        raise RuntimeError(
            f"{ALLREDUCE_ENV_VAR}={req!r} needs a multi-rank process group with "
            f"world_size == n_shards > 1; {state}. Use 'auto' or 'mock' for a "
            "single-process run."
        )
    if req == "nki" and pg[2] != "neuron":
        raise RuntimeError(
            f"{ALLREDUCE_ENV_VAR}='nki' needs a 'neuron' process group "
            f"(TorchNeuron); the active backend is {pg[2]!r}."
        )
    return req


def _dist_allreduce(t: torch.Tensor) -> torch.Tensor:
    """Sum ``t`` across all ranks with ``torch.distributed.all_reduce``."""
    import torch.distributed as dist

    y = t.contiguous().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def reduce_partials(compute_partial: Callable[[int], torch.Tensor], n_shards: int) -> torch.Tensor:
    """Compute and all-reduce the per-shard partials of a reduce-parallel contraction.

    Args:
        compute_partial: ``compute_partial(i)`` returns shard ``i``'s partial
            contraction (full output shape).
        n_shards: Number of reduce-parallel shards.

    Returns:
        The summed result. Under a rank-local backend every rank returns the
        same tensor on its own device, having computed only its own partial.
    """
    backend = allreduce_backend(n_shards)
    if backend == "mock":
        return _mock_allreduce([compute_partial(i) for i in range(n_shards)])
    import torch.distributed as dist

    local = compute_partial(dist.get_rank())
    if backend == "torch":
        return _dist_allreduce(local)
    from .nki.collectives import nki_all_reduce_sum

    return nki_all_reduce_sum(local)
