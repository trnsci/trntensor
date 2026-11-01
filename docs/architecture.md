# Architecture

```
trntensor/
├── trntensor/
│   ├── __init__.py
│   ├── einsum.py        # einsum(), multi_einsum()
│   ├── plan.py          # ContractionPlan, plan_contraction(), estimate_flops()
│   ├── decompose.py     # cp_decompose, tucker_decompose (HOSVD), tt_decompose
│   ├── quantum.py       # mp2_energy, ao_to_mo_transform, df_exchange, df_coulomb
│   ├── parallel.py      # ShardedTensor, scatter, gather
│   └── nki/
│       ├── __init__.py
│       ├── _kernels.py  # @nki.jit kernels
│       ├── _native.py   # TorchNeuron custom ops (nki_op), loaded lazily
│       ├── collectives.py  # nki.collectives all-reduce (reduce-parallel)
│       └── dispatch.py  # host wrappers, dispatch-path selection
├── tests/
├── examples/
│   └── df_mp2_einsum.py # DF-MP2 energy via einsum
```

## Contraction planning

The planner analyzes einsum subscripts and selects a dispatch target:

- **matmul**: 2D contraction over a single shared index whose output lists A's free index then B's → `torch.matmul`
- **bmm**: exactly `bij,bjk->bik` → `torch.bmm`
- **torch**: complex patterns → `torch.einsum`
- **path**: 3+ operands → greedy binary contraction ordering; each binary step re-enters the full backend-selection stack
- **nki** (future): fused multi-index contractions on the Tensor Engine

`ContractionPlan` carries the dispatch decision, the FLOPs estimate, any reshape/transpose preamble, the greedy `contraction_path` for multi-operand contractions, and the `precision` setting — so the same plan can be re-executed cheaply without re-analyzing subscripts. Plans are cached by `(subscripts, shapes, precision)`; call `clear_plan_cache()` to invalidate after a backend change.

### Greedy path search

For 3+ operand contractions, `_greedy_path_search` selects the cheapest binary contraction order by minimizing per-step FLOP cost (product of all index sizes in the pair union). This is the same criterion as `opt_einsum`'s greedy mode. The resulting `contraction_path` drives `_execute_path`, which routes each binary step through the full backend-selection stack — a large sub-contraction in the middle of a 4-operand chain gets the same routing as a standalone call (on the native path that is aten on the device; on the XLA path, NKI above the matmul threshold).

Before 0.22.0 the planner chose `matmul` / `bmm` without checking the output index order, so transposed or permuted outputs (`ij,jk->ki`, `bij,bjk->bki`) came back in the wrong layout on every backend (#64). Outputs in any other order now run through `torch.einsum`.

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

On the native TorchNeuron path (Trn2/Trn3), `mp2_energy_row_kernel` does all three per occupied row, using the pair symmetry `e_ij = e_ji`; on the XLA path (Trn1), `mp2_energy_kernel` does them in one program over `(i, j)` pairs. Neither materializes the intermediate `T` tensor to HBM. See [Quantum-chemistry primitives](api/quantum.md).

This pattern — fused contract + elementwise + reduce with SBUF-resident intermediates — is what trntensor extends across the suite. Generic `einsum` dispatch is the foundation (on the native path dense matmul/bmm run on aten, which beat the NKI matmul kernels at every measured size, so those kernels are deprecated there); primitives like `mp2_energy`, `ao_to_mo_transform`, `df_exchange` and `df_coulomb` are the architectural flagships.

A second pattern — **fused multi-contraction** with a SBUF-resident shared operand — is demonstrated by `ao_to_mo_transform`. The classical AO→MO integral transform chains two matmuls sharing the MO coefficient tensors; a cuTENSOR equivalent would issue two dispatches with the intermediate four-index tensor materialized to HBM between them. Our NKI kernels do both stages without the intermediate ever appearing as a user-visible tensor (it goes through kernel-scratch HBM between the stages), and on the native path they can read packed (μ≥ν) integrals in place. `df_exchange` and `df_coulomb` apply the same idea to the DF-SCF J/K build. The general case of "multi-step contraction DAG compiled to one NKI program" is the architectural superset cuTENSOR can't express; `ao_to_mo_transform` is the first concrete instance.

A third pattern — **user-level operand residency** — exposes that same data-locality idea at the Python API level. Inside a kernel, intermediates live in PSUM/SBUF across steps. At the program level, operands can live on the device (`neuron` on TorchNeuron, the XLA device on Trn1) across successive trntensor calls. `trntensor.to_device(tensor)` pins an operand (`to_xla` is the older name); subsequent trntensor calls with device inputs skip the per-dispatch host↔device transfer entirely. The DF-MP2 pipeline becomes "transfer once, run two fused kernels, pull back the scalar" — matching at the whole-program level what `mp2_energy_kernel` does at the per-kernel level. Users who understand residency get the architecture's full benefit; users who don't still get a correct (if slower) result.

**Performance honest note.** At small chemistry sizes the per-launch floor dominates and `"auto"` keeps aten or the host. On Trn2 (native, device-resident) the row kernel is 1.25–1.9× faster than aten once each row launch carries at least 3 GFLOP, the Trn2 AO→MO kernel 1.3–1.6× at nbasis 256–768, and `df_exchange` takes K at n_ao = 970 from 304 to 94 ms. With residency (`to_device` / `from_device`) the transfer is paid once per pipeline; under `"auto"`, large host-input calls are uploaded above measured thresholds. See [Quantum-chemistry primitives](api/quantum.md) and [NKI dispatch](api/nki.md) for the tables.

## Precision control

`einsum` and `plan_contraction` accept `precision=`:

- **`"fast"`** (default) — native operand dtype; NKI kernels accumulate in fp32
  via PSUM, result returned in operand dtype. No overhead.
- **`"kahan"`** — promotes all operands to fp64 before contracting (via
  `torch.einsum`), casts result back to original dtype. The accumulation is
  fp64-accurate (error K·2⁻⁵³), but the result carries one final rounding to
  its dtype (u_out), so an fp32 result is good to ~1 ulp of fp32, not 16
  digits. Bypasses NKI dispatch; operands on the Neuron device are
  contracted on the host, since Trainium has no fp64 datapath. Useful for
  DF-MP2/CCSD reductions (e.g. a large sum into a scalar) where fp32
  accumulation error, K·u, would dominate.
- **`"dd"`** — double-double accumulation via trnblas Phase 2 fused GEMM kernel.
  On Trainium hosts it raises `NotImplementedError` until trnblas#22 lands. On
  CPU, bf16/fp16 operands run a software Ozaki-2 emulation for `matmul` plans
  (`"kahan"` for `bmm` / `torch` plans); fp32 operands run as `"kahan"`
  (exact products, fp64 accumulation, inside K·u²); fp64 operands use exact
  TwoProduct products with double-word accumulation on `matmul` / `bmm` plans
  and `path` steps, and plain fp64 accumulation on generic `torch` plans.

```python
# Accumulate DF-MP2 energy with ~15-digit precision
E_corr = trntensor.einsum("ijab,ijab->", T, T2, precision="kahan")
```

`precision` is included in the plan cache key so `"fast"` and `"kahan"` plans
cache independently without cross-contaminating the execution path.

## Decompositions for quantum chemistry

- **CP (CANDECOMP/PARAFAC)** — Tensor hypercontraction (THC) of two-electron integrals. Reduces $O(N^4)$ storage to $O(N^2 R)$. Supports `nonneg=True` (multiplicative ALS updates) and `factors=` warm-start.
- **Tucker (HOSVD)** — Low-rank approximation of DF coefficient tensors. Reduces memory for large auxiliary basis sets.
- **Tensor Train (TT-SVD)** — Chain-of-cores decomposition for high-dimensional tensors (Oseledets 2011). Bond dimension capped at `max_rank`. Reduces $O(n^d)$ storage to $O(d \cdot n \cdot R^2)$ at the cost of truncation error. Useful for DMRG-style state-vector compression.

CP and Tucker use alternating-least-squares on top of `trnblas`-style GEMM primitives. TT-SVD uses successive truncated SVD. All decompositions can be fed back through `einsum` at evaluation time — e.g., contracting Tucker factors directly against a state vector without materializing the full tensor.
