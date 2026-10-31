"""
Native TorchNeuron registration of the trntensor NKI kernels.

On the TorchNeuron lineage (Neuron SDK 2.33+, torch-neuronx 2.14,
PyTorch 2.14, Trn2/Trn3), Trainium is a native PyTorch `neuron` device
and an `@nki.jit` kernel is exposed to PyTorch as a custom operator via
`torch_neuronx.nki_op`. Each wrapper below calls an existing kernel from
`_kernels.py` unchanged; the op schema is inferred from the annotations.

Importing this module registers the ops, so `dispatch.py` imports it only
after selecting the native path. CPU-only installs and the Trn1 / XLA
lineage (whose `torch_neuronx` has no `nki_op`) never load it.

This module deliberately does not use `from __future__ import annotations`:
`nki_op` builds the op schema from the live annotation objects.

The kernels are looked up on `_kernels` at call time rather than imported
by name, so registration itself does not trace or compile anything; the
first call per input shape (and per `use_sr`) does.

The reduce-parallel all-reduce kernel (`collectives.py`) is not registered
here: its `ReplicaGroup` argument is a compile-time constant that a custom
op schema cannot carry, so it keeps its own `torch_neuronx.wrap_nki` launch,
which is already native-only.
"""

import torch
from torch_neuronx import nki_op

from . import _kernels as _k

# Fake (meta) implementations: output shape/dtype only, no kernel trace.
# Used by torch.compile tracing and for meta-device inputs.


@nki_op(
    "trntensor::matmul",
    mutates_args={},
    fake_fn=lambda a, b, use_sr=False: a.new_empty((a.shape[0], b.shape[1])),
)
def matmul(a: torch.Tensor, b: torch.Tensor, use_sr: bool = False) -> torch.Tensor:
    return _k.matmul_kernel(a, b, use_sr=use_sr)


@nki_op(
    "trntensor::batched_matmul",
    mutates_args={},
    fake_fn=lambda a, b, use_sr=False: a.new_empty((a.shape[0], a.shape[1], b.shape[2])),
)
def batched_matmul(a: torch.Tensor, b: torch.Tensor, use_sr: bool = False) -> torch.Tensor:
    return _k.batched_matmul_kernel(a, b, use_sr=use_sr)


@nki_op(
    "trntensor::mp2_energy",
    mutates_args={},
    fake_fn=lambda B_T, eps_occ_b, eps_vir, eps_vir_b: B_T.new_empty(
        (B_T.shape[0], B_T.shape[0]), dtype=torch.float32
    ),
)
def mp2_energy(
    B_T: torch.Tensor,
    eps_occ_b: torch.Tensor,
    eps_vir: torch.Tensor,
    eps_vir_b: torch.Tensor,
) -> torch.Tensor:
    return _k.mp2_energy_kernel(B_T, eps_occ_b, eps_vir, eps_vir_b)


@nki_op(
    "trntensor::ao_to_mo_transform",
    mutates_args={},
    fake_fn=lambda eri, C_occ, C_vir: eri.new_empty((C_occ.shape[1], C_vir.shape[1], eri.shape[2])),
)
def ao_to_mo_transform(eri: torch.Tensor, C_occ: torch.Tensor, C_vir: torch.Tensor) -> torch.Tensor:
    return _k.ao_to_mo_transform_kernel(eri, C_occ, C_vir)


@nki_op(
    "trntensor::mp2_energy_row",
    mutates_args={},
    fake_fn=lambda bt_i, Bw, eps_occ_i, eps_occ_w, eps_vir_col, eps_vir_row: bt_i.new_empty(
        (128, Bw.shape[0]), dtype=torch.float32
    ),
)
def mp2_energy_row(
    bt_i: torch.Tensor,
    Bw: torch.Tensor,
    eps_occ_i: torch.Tensor,
    eps_occ_w: torch.Tensor,
    eps_vir_col: torch.Tensor,
    eps_vir_row: torch.Tensor,
) -> torch.Tensor:
    return _k.mp2_energy_row_kernel(bt_i, Bw, eps_occ_i, eps_occ_w, eps_vir_col, eps_vir_row)


@nki_op(
    "trntensor::ao_to_mo_trn2",
    mutates_args={},
    fake_fn=lambda eri, C_occ, C_vir: eri.new_empty((C_occ.shape[1], C_vir.shape[1], eri.shape[2])),
)
def ao_to_mo_trn2(eri: torch.Tensor, C_occ: torch.Tensor, C_vir: torch.Tensor) -> torch.Tensor:
    return _k.ao_to_mo_trn2_kernel(eri, C_occ, C_vir)


# Keyed by the short op name used in `dispatch._launch`.
OPS = {
    "matmul": matmul,
    "batched_matmul": batched_matmul,
    "mp2_energy": mp2_energy,
    "ao_to_mo_transform": ao_to_mo_transform,
    "mp2_energy_row": mp2_energy_row,
    "ao_to_mo_trn2": ao_to_mo_trn2,
}
