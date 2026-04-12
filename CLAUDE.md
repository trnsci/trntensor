# trntensor

Tensor contractions for AWS Trainium via NKI.
Part of the trn-* scientific computing suite by Playground Logic.

## What This Is

A cuTENSOR-equivalent for Trainium. Einstein summation with contraction
planning, and CP/Tucker decompositions. Expresses tensor workloads
naturally instead of decomposing to GEMM.

For DF-MP2, the natural expression `einsum("ap,bp->ab", B[i], B[j])` is
clearer than `B[i] @ B[j].T` and enables the planner to fuse operations
(e.g., folding the energy denominator division into the accumulation).

## Architecture

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
│   ├── test_einsum.py   # Matmul, bmm, 4-index transform, DF-MP2 pattern
│   └── test_decompose.py # CP rank-1/low-rank, Tucker orthogonality
├── examples/
│   └── df_mp2_einsum.py # DF-MP2 energy via einsum
```

## Contraction Planning

The planner analyzes einsum subscripts and selects:
- **matmul**: 2D contraction over single index → `torch.matmul`
- **bmm**: batched 2D contraction → `torch.bmm`
- **torch**: complex patterns → `torch.einsum`
- **(future) nki**: fused multi-index contractions on Tensor Engine

## Decompositions for Quantum Chemistry

- **CP (CANDECOMP/PARAFAC)**: Tensor hypercontraction (THC) of
  two-electron integrals. Reduces O(N⁴) storage to O(N²R).
- **Tucker (HOSVD)**: Low-rank approximation of DF coefficient tensors.
  Reduces memory for large auxiliary basis sets.

## Dependencies

- `torch>=2.1`, `numpy>=1.24`
- `neuronxcc` (optional)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
python examples/df_mp2_einsum.py --demo
```
