# Quickstart

## einsum

```python
import torch
import trntensor

# 2-index contraction (dispatches to matmul)
A = torch.randn(32, 64)
B = torch.randn(48, 64)
C = trntensor.einsum("ap,bp->ab", A, B)  # (32, 48)

# Batched (dispatches to bmm; only the exact form bij,bjk->bik does)
Ab = torch.randn(16, 32, 64)
Bb = torch.randn(16, 64, 48)
Cb = trntensor.einsum("bij,bjk->bik", Ab, Bb)  # (16, 32, 48)

# Other index orders run through torch.einsum (#64)
Bt = torch.randn(16, 48, 64)
Ct = trntensor.einsum("iap,ibp->iab", Ab, Bt)  # (16, 32, 48)
```

## Contraction planning

```python
plan = trntensor.plan_contraction("ap,bp->ab", A, B)
print(plan.strategy)  # "matmul" | "bmm" | "torch" | "path"
print(plan.backend)  # "nki" | "pytorch"

flops = trntensor.estimate_flops("ap,bp->ab", A, B)
```

## Decompositions

```python
# CP (CANDECOMP/PARAFAC)
X = torch.randn(16, 16, 16)
factors, weights = trntensor.cp_decompose(X, rank=8)
reconstructed = trntensor.cp_reconstruct(factors, weights)

# Tucker (HOSVD)
core, factors = trntensor.tucker_decompose(X, ranks=(4, 4, 4))
reconstructed = trntensor.tucker_reconstruct(core, factors)
```

## Backend selection

```python
trntensor.set_backend("auto")  # default: measured routing per call
trntensor.set_backend("pytorch")  # force PyTorch fallback
trntensor.set_backend("nki")  # force NKI (requires Neuron hardware)
```
