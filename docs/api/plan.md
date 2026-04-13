# Planning

The planner analyzes einsum subscripts to select a concrete execution
strategy and estimate cost.

## `plan_contraction(subscripts, *operands) -> ContractionPlan`

Inspect the dispatch decision without executing.

```python
plan = trntensor.plan_contraction("ij,jk->ik", A, B)
plan.strategy          # "matmul" | "bmm" | "torch"
plan.contraction_indices  # list of indices being summed over
plan.batch_indices        # list of batch indices
plan.output_indices       # list of output indices in order
plan.transA, plan.transB  # whether to pre-transpose before matmul
```

## `ContractionPlan`

Dataclass with the fields above plus:

- `subscripts` — the original subscript string
- `backend` — executor that will run the contraction: `"nki"` (when
  the strategy is `matmul`/`bmm` and `neuronxcc` is importable) or
  `"pytorch"`
- `estimated_flops: int`

## `estimate_flops(subscripts, *operands) -> int`

Estimate multiply-add operations for a contraction as the product of
all distinct index sizes.

```python
trntensor.estimate_flops("ij,jk->ik", A, B)  # M*K*N
```
