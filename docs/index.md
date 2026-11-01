# trntensor

Tensor contractions for AWS Trainium via NKI.

A cuTENSOR-equivalent for Trainium: Einstein summation with contraction planning, and CP/Tucker decompositions. Expresses tensor workloads naturally instead of decomposing manually to GEMM.

For DF-MP2, the natural expression `einsum("ap,bp->ab", B[i], B[j])` is clearer than `B[i] @ B[j].T` and enables the planner to fuse operations (e.g., folding the energy denominator division into the accumulation).

## Install

```bash
pip install trntensor

# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trntensor[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trntensor[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

## Quick example

```python
import torch
import trntensor

A = torch.randn(32, 64)
B = torch.randn(48, 64)
C = trntensor.einsum("ap,bp->ab", A, B)

# With the planner
plan = trntensor.plan_contraction("ap,bp->ab", A, B)
print(f"strategy: {plan.strategy}, backend: {plan.backend}")
print(trntensor.estimate_flops("ap,bp->ab", A, B))
```

## Status

Alpha (0.x). einsum, the planner and the CP / Tucker / TT decompositions run on any PyTorch host. The fused chemistry kernels (`mp2_energy`, `ao_to_mo_transform`, `df_exchange`, `df_coulomb`) are validated on Trn2 with native TorchNeuron (Neuron SDK 2.33); on the native path dense matmul/bmm run on aten, which beat the NKI matmul kernels at every measured size. The PyTorch/XLA path (Trn1) is in maintenance. See the [changelog](https://github.com/trnsci/trntensor/blob/main/CHANGELOG.md) for current numbers.

Part of the [trnsci](https://github.com/trnsci/trnsci) scientific computing suite.
