# API Reference

## einsum

- `einsum(subscripts, *operands)` — single contraction; mirrors `torch.einsum` with dispatch optimization
- `multi_einsum(expressions, operands_map)` — execute a sequence of contractions reusing intermediates

## Planning

- `plan_contraction(subscripts, *operands) -> ContractionPlan`
  - `.dispatch` — `"matmul" | "bmm" | "torch" | "nki"`
  - `.flops` — estimated FLOP count
  - `.execute(*operands)` — run the plan on new operands with the same shape
- `estimate_flops(subscripts, *operands) -> int`

## Decompositions

- `cp_decompose(X, rank, *, max_iter=100, tol=1e-6) -> list[Tensor]`
- `cp_reconstruct(factors) -> Tensor`
- `tucker_decompose(X, ranks, *, max_iter=100, tol=1e-6) -> (core, factors)`
- `tucker_reconstruct(core, factors) -> Tensor`

## Dispatch

- `set_backend("auto" | "pytorch" | "nki")`
- `get_backend()`
- `HAS_NKI` — module-level flag set at import time
