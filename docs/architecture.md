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

## Fused contraction-reduction: the architectural core

What makes trntensor different from `torch.einsum` or `trnblas.gemm` is **fusion across the boundary between contraction and reduction**. On Trainium:

- the Tensor Engine does the contraction, accumulating into PSUM
- the Vector Engine does the elementwise reshaping, reading from SBUF
- a small SBUF scalar accumulates the reduction
- nothing lands in HBM until the whole program finishes

A cuTENSOR clone would compose these three phases as separate ops with HBM round-trips between them. NKI lets us write them as one `@nki.jit` program with the same data living in PSUM → SBUF → accumulator across phases. That's the architectural lever.

The canonical demonstration is `trntensor.mp2_energy` — the DF-MP2 correlation energy is

```
T_{i,j,a,b} = Σ_P B_{i,a,P} B_{j,b,P}              (contraction)
term        = T * (2T - T^T) / Δ_{i,j,a,b}         (elementwise)
E           = Σ_{i,j,a,b} term                      (reduction)
```

The fused kernel in `trntensor/nki/_kernels.py::mp2_energy_kernel` does all three in one program, iterating over `(i, j)` pairs in `affine_range`. No intermediate `T` tensor is ever materialized to HBM.

This pattern — fused contract + elementwise + reduce with SBUF-resident intermediates — is what trntensor extends across the suite. Generic `einsum` dispatch to `matmul` / `bmm` kernels is the foundation; primitives like `mp2_energy` are the architectural flagships.

**Performance honest note.** The fused kernel is architecturally correct but currently carries ~15–40 ms of XLA dispatch + compile overhead per call, which means at small chemistry sizes the CPU fallback path still wins. The fusion work pays off once per-call overhead is amortized (tracked in #33/#34). See [Benchmarks](benchmarks.md) for the current numbers.

## Decompositions for quantum chemistry

- **CP (CANDECOMP/PARAFAC)** — Tensor hypercontraction (THC) of two-electron integrals. Reduces $O(N^4)$ storage to $O(N^2 R)$.
- **Tucker (HOSVD)** — Low-rank approximation of DF coefficient tensors. Reduces memory for large auxiliary basis sets.

Both use alternating-least-squares on top of `trnblas`-style GEMM primitives. The decompositions can be fed back through `einsum` at evaluation time — e.g., contracting Tucker factors directly against a state vector without materializing the full tensor.
