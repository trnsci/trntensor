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

A second pattern — **fused multi-contraction** with a SBUF-resident shared operand — is demonstrated by `ao_to_mo_transform`. The classical AO→MO integral transform chains two matmuls sharing the MO coefficient tensors; a cuTENSOR equivalent would issue two dispatches with the intermediate four-index tensor materialized to HBM between them. Our NKI kernel loads `C_occ` and `C_vir` once SBUF-resident across every auxiliary index P, does both matmuls in one program, and the per-P intermediate never appears as a user-visible tensor. The general case of "multi-step contraction DAG compiled to one NKI program" is the architectural superset cuTENSOR can't express; `ao_to_mo_transform` is the first concrete instance.

A third pattern — **user-level operand residency** — exposes that same data-locality idea at the Python API level. Inside a kernel, intermediates live in PSUM/SBUF across steps. At the program level, operands can live on the XLA device across successive trntensor calls. `trntensor.to_xla(tensor)` pins an operand; subsequent trntensor calls with XLA inputs skip the per-dispatch host↔device transfer entirely. The DF-MP2 pipeline becomes "transfer once, run two fused kernels, pull back the scalar" — matching at the whole-program level what `mp2_energy_kernel` does at the per-kernel level. Users who understand residency get the architecture's full benefit; users who don't still get a correct (if slower) result.

**Performance honest note.** Fused kernels are architecturally correct. Without residency, per-dispatch overhead dominates at small chemistry sizes and the CPU fallback wins on benchmarks. With residency (`to_xla` / `from_xla`), the overhead is paid once per pipeline instead of once per call, and the fusion work starts paying for itself. See [Benchmarks](benchmarks.md) for the current numbers.

## Decompositions for quantum chemistry

- **CP (CANDECOMP/PARAFAC)** — Tensor hypercontraction (THC) of two-electron integrals. Reduces $O(N^4)$ storage to $O(N^2 R)$.
- **Tucker (HOSVD)** — Low-rank approximation of DF coefficient tensors. Reduces memory for large auxiliary basis sets.

Both use alternating-least-squares on top of `trnblas`-style GEMM primitives. The decompositions can be fed back through `einsum` at evaluation time — e.g., contracting Tucker factors directly against a state vector without materializing the full tensor.
