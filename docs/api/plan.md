# Planning

The planner analyzes einsum subscripts to select a concrete execution
strategy, estimate cost, and cache the decision so repeated calls are cheap.

## `plan_contraction(subscripts, *operands, *, precision="fast") -> ContractionPlan`

Inspect the dispatch decision without executing.

```python
plan = trntensor.plan_contraction("ij,jk->ik", A, B)
plan.strategy             # "matmul" | "bmm" | "torch" | "path"
plan.contraction_indices  # list of indices being summed over
plan.batch_indices        # list of batch indices
plan.output_indices       # list of output indices in order
plan.transA, plan.transB  # whether to pre-transpose before matmul
plan.contraction_path     # [(i,j), ...] for "path" strategy
plan.precision            # "fast" | "kahan" | "dd"
```

Results are cached by `(subscripts, operand shapes, precision)`. Repeated
calls with the same subscript, shapes, and precision skip replanning entirely.
Call `clear_plan_cache()` to invalidate (e.g. after a backend change).

### `precision` values

| Value | Effect on planning |
|-------|--------------------|
| `"fast"` | Default; NKI kernels eligible for matmul/bmm strategies |
| `"kahan"` | Signals fp64 promotion at execution time; plan is still computed but `einsum` will use `torch.einsum` in fp64 regardless of strategy |
| `"dd"` | Raises `NotImplementedError` at execution time (trnblas#22 pending) |

```python
plan_fast  = trntensor.plan_contraction("ij,jk->ik", A, B, precision="fast")
plan_kahan = trntensor.plan_contraction("ij,jk->ik", A, B, precision="kahan")
# Two distinct cache entries — plan_fast is not plan_kahan
```

## `ContractionPlan`

Dataclass returned by `plan_contraction`. All fields are read-only after
the planner returns.

| Field | Type | Description |
|-------|------|-------------|
| `subscripts` | `str` | Original subscript string |
| `strategy` | `str` | `"matmul"` \| `"bmm"` \| `"torch"` \| `"path"` |
| `backend` | `str` | `"nki"` or `"pytorch"` |
| `transA` | `bool` | Pre-transpose first operand (matmul only) |
| `transB` | `bool` | Pre-transpose second operand (matmul only) |
| `contraction_indices` | `list[str]` | Indices summed over |
| `batch_indices` | `list[str]` | Shared batch indices |
| `output_indices` | `list[str]` | Output index order |
| `estimated_flops` | `int` | Multiply-add estimate |
| `contraction_path` | `list[tuple[int,int]]` | Greedy pair order for `"path"` strategy (opt_einsum convention) |
| `precision` | `str` | `"fast"` \| `"kahan"` \| `"dd"` |

### Strategy selection

- **`"matmul"`** — 2-operand, 2D tensors, single contracted index → `torch.matmul` (or NKI kernel when backend is `"nki"`)
- **`"bmm"`** — 2-operand, 3D tensors, single contracted index + single batch index → `torch.bmm` (or NKI batched kernel)
- **`"torch"`** — 2-operand fallback for patterns that don't fit matmul/bmm → `torch.einsum`
- **`"path"`** — 3+ operands → greedy binary contraction ordering; each binary step is dispatched through the full backend-selection stack so large sub-contractions still reach NKI

## `estimate_flops(subscripts, *operands) -> int`

Estimate multiply-add operations for a contraction as the product of
all distinct index sizes.

```python
trntensor.estimate_flops("ij,jk->ik", A, B)  # M*K*N
trntensor.estimate_flops("iap,jbp->ijab", B, B)  # nocc² * nvir² * naux
```

## Plan cache helpers

### `clear_plan_cache() -> None`

Discard all cached contraction plans. Call after changing the active
backend or when memory is a concern.

```python
trntensor.set_backend("nki")
trntensor.clear_plan_cache()  # flush plans that may have chosen "pytorch"
```

### `plan_cache_info() -> dict[str, int]`

Return cache statistics. Currently returns `{"size": N}` where `N` is
the number of cached plans.

```python
info = trntensor.plan_cache_info()
print(f"cached plans: {info['size']}")
```
