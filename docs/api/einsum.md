# einsum

Einstein summation with contraction planning. Mirrors the subscript
notation used by `torch.einsum` and `numpy.einsum`.

## `einsum(subscripts, *operands) -> torch.Tensor`

Execute a single contraction. The planner selects one of `matmul`,
`bmm`, or `torch.einsum` based on the subscript pattern and operand
shapes. When a NKI backend is active and the pattern maps to a 2-index
matmul, the kernel in `trntensor.nki._kernels.matmul_kernel` is used.

```python
import trntensor

# Matrix multiply
C = trntensor.einsum("ij,jk->ik", A, B)

# Batched matmul
C = trntensor.einsum("bij,bjk->bik", A, B)

# DF-MP2 pair contraction
T_ab = trntensor.einsum("ap,bp->ab", B_i, B_j)

# 4-index transform
Inu = trntensor.einsum("mi,mnP->inP", C, eri)
```

## `multi_einsum(*contractions) -> list[torch.Tensor]`

Execute several contractions in one call. Each contraction is a tuple
of `(subscripts, *operands)`. The result preserves input order. In
future versions the planner will reuse intermediates across
contractions when they share operands (tracked in [#19][19]).

```python
results = trntensor.multi_einsum(
    ("ij,jk->ik", A, B),
    ("ij,jk->ik", B, C),
)
```

[19]: https://github.com/trnsci/trntensor/issues/19
