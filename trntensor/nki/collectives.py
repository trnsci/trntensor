"""NKI collective all-reduce for reduce-parallel contractions (Phase 4).

``nki_all_reduce_sum(t)`` sums a tensor across every rank of the active
``"neuron"`` process group with an ``@nki.jit`` kernel that calls
``nki.collectives.all_reduce(op=nl.add)``. Each rank is a separate process
pinned to one logical NeuronCore (``NEURON_RT_VISIBLE_CORES``); the tensor
lives on that rank's ``neuron`` device.

The kernel follows the pattern measured bit-exact on Trn2 (SDK 2.33, NKI
0.7.0, LNC=2, 2 and 4 ranks; trnsci spike B4,
``scripts/spike_collectives_multirank.py``):

- Collectives cannot read or write kernel IO tensors. The input is
  ``dma_copy``'d into a named internal HBM ``src``, reduced into a named
  ``dst``, and ``dma_copy``'d to the returned output.
- ``src``/``dst`` are on ``shared_hbm``. ``private_hbm`` gave wrong values at
  LNC=2.
- ``name=`` is required on collective buffers (otherwise NCC_IBIR440).
- The kernel is invoked as
  ``torch_neuronx.wrap_nki(kernel)[(lnc,)](x, ReplicaGroup(...))``.
  ``wrap_nki`` treats the ``ReplicaGroup`` as a compile-time constant.
- Every rank must compile the same NEFF, so every rank must pass the same
  shape and dtype. A reduce-parallel einsum guarantees this: each partial
  has the full output shape.
- The replica group is the whole world, ``[[0, ..., world-1]]``. The ranks'
  physical cores must form an aligned pair or quad (cores 0-1, 2-3, 0-3
  work; 1-2 fails with "no_hier no_mesh"). That is set by the launcher, not
  here.

Importing this module does not import ``nki``. The kernel is built on first
use so CPU-only installs can import :mod:`trntensor.parallel` freely.
"""

from __future__ import annotations

import torch

_kernel = None


def _build_kernel():
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def trntensor_all_reduce_add(x, replica_group):
        src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="trntensor_ar_src")
        dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="trntensor_ar_dst")
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=x)
        ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=replica_group, op=nl.add)
        nisa.dma_copy(dst=out, src=dst)
        return out

    return trntensor_all_reduce_add


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    """View ``t`` as 2-D: the (rows, cols) layout the measured kernel used."""
    if t.ndim == 0:
        return t.reshape(1, 1)
    if t.ndim == 1:
        return t.reshape(1, t.shape[0])
    return t.reshape(t.shape[0], -1)


def nki_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """Sum ``t`` across all ranks with ``nki.collectives.all_reduce(add)``.

    Args:
        t: This rank's partial, on the rank's ``neuron`` device. Same shape
            and dtype on every rank.

    Returns:
        The rank-summed tensor, same shape as ``t``, on the same device.

    Raises:
        RuntimeError: if no ``"neuron"`` process group is initialized.
    """
    global _kernel
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("nki_all_reduce_sum needs an initialized 'neuron' process group")
    from nki.collectives import ReplicaGroup
    from torch_neuronx import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    if _kernel is None:
        _kernel = _build_kernel()
    world = dist.get_world_size()
    lnc = int(get_logical_neuron_cores())
    x = _as_2d(t.contiguous())
    y = wrap_nki(_kernel)[(lnc,)](x, ReplicaGroup([list(range(world))]))
    return y.reshape(t.shape)
