# einsum

Einstein summation with contraction planning. Mirrors the subscript
notation used by `torch.einsum` and `numpy.einsum`, with additional
control over scaling, accumulation precision, and mixed-precision compute.

## `einsum(subscripts, *operands, *, alpha, beta, out, dtype, precision) -> torch.Tensor`

Execute a single contraction.

```python
einsum(
    subscripts: str,
    *operands: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    out: torch.Tensor | None = None,
    dtype: str | torch.dtype | None = None,
    precision: str = "fast",
) -> torch.Tensor
```

The planner selects `matmul`, `bmm`, `path`, or `torch.einsum` based on
the subscript pattern and operand shapes. When a NKI backend is active and
the pattern maps to a 2-index matmul, the kernel in
`trntensor.nki._kernels.matmul_kernel` is used.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `subscripts` | — | Einstein summation string, e.g. `"ij,jk->ik"` |
| `*operands` | — | Input tensors |
| `alpha` | `1.0` | Scalar multiplier on the contraction result |
| `beta` | `0.0` | Scalar multiplier on `out` before accumulation |
| `out` | `None` | Accumulation tensor; result is `α·contract + β·out` |
| `dtype` | `None` | Cast all operands to this dtype before contracting; result is returned in this dtype. Accepts `torch.dtype` or strings: `"bf16"`, `"bfloat16"`, `"fp16"`, `"float16"`, `"f32"`, `"float32"` |
| `precision` | `"fast"` | Accumulation precision: `"fast"`, `"kahan"`, or `"dd"` (see below) |

### Basic examples

```python
import trntensor

# Matrix multiply
C = trntensor.einsum("ij,jk->ik", A, B)

# Batched matmul
C = trntensor.einsum("bij,bjk->bik", A, B)

# DF-MP2 pair contraction
T_ab = trntensor.einsum("ap,bp->ab", B_i, B_j)

# 4-index AO→MO transform step
Inu = trntensor.einsum("mi,mnP->inP", C, eri)
```

### `alpha` / `beta` scaling

Matches cuTENSOR's GEMM-style interface:
`alpha·contract(A, B) + beta·C`.

```python
# Scaled GEMM: 2·A@B + 0.5·C
result = trntensor.einsum("ij,jk->ik", A, B, alpha=2.0, beta=0.5, out=C)

# In-place accumulation (beta=1, alpha=1): result += A@B
trntensor.einsum("ij,jk->ik", A, B, beta=1.0, out=acc)
```

### `dtype` — mixed-precision compute

Cast operands to a lower dtype to hit the NKI bf16 matmul path without
changing the model's weight dtype. The result is returned in the requested
dtype.

```python
# Route through NKI bf16 kernel even if weights are fp32
result_bf16 = trntensor.einsum("ij,jk->ik", A, B, dtype="bf16")

# Use torch.dtype directly
result = trntensor.einsum("ij,jk->ik", A, B, dtype=torch.float16)
```

### `precision` — accumulation precision

| Value | Behavior |
|-------|----------|
| `"fast"` | Native operand dtype; NKI kernels accumulate in fp32 via PSUM |
| `"kahan"` | Promotes operands to fp64 before contracting, casts back to original dtype; ~15.9 significant digits; bypasses NKI dispatch (runs on CPU) |
| `"dd"` | Double-double accumulation via trnblas Phase 2 — raises `NotImplementedError` until trnblas#22 lands |

```python
# High-precision DF-MP2 energy accumulation
E_corr = trntensor.einsum("ijab,ijab->", T, T2, precision="kahan")
```

---

## `multi_einsum(*contractions) -> list[torch.Tensor]`

Execute several contractions in one call. Each contraction is a tuple
of `(subscripts, *operands)`. Results are returned in input order.

When NKI dispatch is active, operand tensors that appear in more than one
contraction (by object identity) are pre-pinned to the XLA device once
before executing the loop, eliminating redundant host↔device transfers.

```python
results = trntensor.multi_einsum(
    ("ap,bp->ab", B_i, B_j),   # Coulomb term
    ("ap,bp->ab", B_i, B_k),   # exchange term — B_i pinned once
)
```

### Plan cache helpers

These are documented on the [Planning](plan.md) page but are also
relevant to einsum performance:

- `trntensor.clear_plan_cache()` — discard all cached plans
- `trntensor.plan_cache_info()` — return `{"size": N}` cache statistics
