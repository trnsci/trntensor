# Quickstart

## einsum

```python
import torch
import trntensor

# 2-index contraction (dispatches to matmul)
A = torch.randn(32, 64)
B = torch.randn(48, 64)
C = trntensor.einsum("ap,bp->ab", A, B)   # (32, 48)

# Batched (dispatches to bmm)
Ab = torch.randn(16, 32, 64)
Bb = torch.randn(16, 48, 64)
Cb = trntensor.einsum("iap,ibp->iab", Ab, Bb)   # (16, 32, 48)
```

## Contraction planning

```python
plan = trntensor.plan_contraction("ap,bp->ab", A, B)
print(plan.dispatch)   # "matmul" | "bmm" | "torch" | "nki"
print(plan.flops)      # estimated FLOP count

flops = trntensor.estimate_flops("ap,bp->ab", A, B)
```

## Decompositions

```python
# CP (CANDECOMP/PARAFAC)
X = torch.randn(16, 16, 16)
factors = trntensor.cp_decompose(X, rank=8)
reconstructed = trntensor.cp_reconstruct(factors)

# Tucker (HOSVD)
core, factors = trntensor.tucker_decompose(X, ranks=(4, 4, 4))
reconstructed = trntensor.tucker_reconstruct(core, factors)
```

## Backend selection

```python
trntensor.set_backend("auto")     # default
trntensor.set_backend("pytorch")  # force PyTorch fallback
trntensor.set_backend("nki")      # force NKI (requires Neuron hardware)
```
