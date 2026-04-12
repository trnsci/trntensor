# trntensor

Tensor contractions for AWS Trainium via NKI.

A cuTENSOR-equivalent for Trainium: Einstein summation with contraction planning, and CP/Tucker decompositions. Expresses tensor workloads naturally instead of decomposing manually to GEMM.

For DF-MP2, the natural expression `einsum("ap,bp->ab", B[i], B[j])` is clearer than `B[i] @ B[j].T` and enables the planner to fuse operations (e.g., folding the energy denominator division into the accumulation).

## Install

```bash
pip install trntensor
pip install trntensor[neuron]   # on Neuron hardware
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
print(f"dispatch: {plan.dispatch}, flops: {plan.flops}")
```

## Status

Alpha. einsum and CP / Tucker decompositions are functional via PyTorch fallback. NKI fused-contraction kernels are scaffolded; on-hardware validation is the next milestone.

Part of the [trnsci](https://github.com/trnsci/trnsci) scientific computing suite.
