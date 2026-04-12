# Architecture

```
trntensor/
├── trntensor/
│   ├── __init__.py
│   ├── einsum.py        # einsum(), multi_einsum()
│   ├── plan.py          # ContractionPlan, plan_contraction(), estimate_flops()
│   ├── decompose.py     # cp_decompose, tucker_decompose (HOSVD)
│   └── nki/
│       ├── __init__.py
│       └── dispatch.py  # Fused contraction kernels
├── tests/
├── examples/
│   └── df_mp2_einsum.py # DF-MP2 energy via einsum
```

## Contraction planning

The planner analyzes einsum subscripts and selects a dispatch target:

- **matmul**: 2D contraction over a single shared index → `torch.matmul`
- **bmm**: batched 2D contraction → `torch.bmm`
- **torch**: complex patterns → `torch.einsum`
- **nki** (future): fused multi-index contractions on the Tensor Engine

`ContractionPlan` carries the dispatch decision, the FLOPs estimate, and any reshape/transpose preamble so the same plan can be re-executed cheaply.

## Decompositions for quantum chemistry

- **CP (CANDECOMP/PARAFAC)** — Tensor hypercontraction (THC) of two-electron integrals. Reduces $O(N^4)$ storage to $O(N^2 R)$.
- **Tucker (HOSVD)** — Low-rank approximation of DF coefficient tensors. Reduces memory for large auxiliary basis sets.

Both use alternating-least-squares on top of `trnblas`-style GEMM primitives. The decompositions can be fed back through `einsum` at evaluation time — e.g., contracting Tucker factors directly against a state vector without materializing the full tensor.
