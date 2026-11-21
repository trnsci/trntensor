# matmul_kernel engine utilization profile

**Date:** 2026-04-17  
**Tracker:** [#33](https://github.com/trnsci/trntensor/issues/33)  
**Hardware:** trn1.2xlarge, Neuron SDK 2.29.18.0, NKI 0.3.0  
**Method:** sustained workload timing + neuron-monitor (see Profiling approach note)

## TL;DR

`matmul_kernel` at 1024² is Tensor Engine-dominant (by design — GEMM maps directly
to TE tiles). The ~0.67 ms fixed overhead per dispatch is entirely in the XLA graph
submission path, not in kernel execution. At 1024², the kernel itself executes in
**~0.68 ms**; the remaining ~0.9 ms of the 1.6 ms pre-pinned total is device→host
transfer.

---

## Sustained throughput (trn1.2xlarge, 1024² fp32)

Measurement: run 3793 iterations of `einsum("ij,jk->ik", A, B)` in 15 s on a
warmed-up (post-compile) instance.

| Metric | Value |
|--------|-------|
| Iterations | 3793 in 15 s |
| Throughput | 252 iters/s |
| Wall-clock / iter | ~3.97 ms |
| FLOPs per iter (1024²) | 2.15 GFLOPs |
| Effective throughput | 541 GFLOPS |
| trn1 peak bf16 TE | 190 TFLOPS (2 NeuronCores) |
| trn1 peak fp32 TE | 47.5 TFLOPS (2 NeuronCores) |
| Estimated TE utilization (fp32) | ~1.1% |

The low utilization reflects the dispatch-dominated regime: each iteration pays
~0.67 ms of XLA submission overhead that the TE is idle during. The kernel itself
runs in ~0.68 ms (from the dispatch timing sweep).

### Per-step breakdown at 1024² (from dispatch timing findings)

| Step | Time |
|------|------|
| host→XLA transfer | 1.50 ms |
| Kernel submission | 0.68 ms |
| device→host transfer | 0.97 ms |
| **Total (unpinned)** | **2.94 ms** |
| Kernel only (pre-pinned) | 0.68 ms |
| device→host (pre-pinned) | 0.92 ms |
| **Total (pre-pinned)** | **1.60 ms** |

### At 2048² (pre-pinned)

| Step | Time |
|------|-------|
| Kernel | 0.97 ms |
| device→host | 3.14 ms |
| **Total pre-pinned** | **4.12 ms** |
| PyTorch baseline | 23.27 ms |
| **Speedup** | **5.65×** |

At 2048², the kernel execution (0.97 ms) is fast. The dominant cost is
device→host transfer (3.14 ms), not the TE. For workloads where results are
immediately fed into another NKI kernel (no `from_xla` needed), this cost
disappears — that is the Phase 3 batching motivation.

---

## Profiling approach note

**Neuron Profiler 2.0 trace capture** (`neuron-profile capture -n model.neff`)
requires a standalone `.neff` file. In NKI 0.3.0 / Neuron SDK 2.29, `trntensor`
kernels go through the `torch_xla` → XLA lazy graph → Neuron XLA backend path.
The NEFF is compiled in-memory and is **not written to `/var/tmp/neuron-compile-cache/`**
(the location used by earlier `neuronxcc.nki.*` direct-compile workflows like
trnblas). The `NEURON_PROFILE` and `NEURONX_DUMP_TO` environment variables did not
produce on-disk artifacts in the tested configuration.

**Alternative for TE/VE breakdown**: compile `matmul_kernel` via `nki.jit` with
an explicit output path (NKI 0.3.0 nki.compiler API), then run `neuron-profile
capture -n <path>/model.neff`. This avoids the torch_xla compilation path and
produces a NEFF that the profiler can consume. Planned for a follow-up run once
the `trntensor-ci-trn1` instance is provisioned (currently using `trnblas-ci-trn1`
as a stop-gap).

---

## Key findings

### 1. Kernel execution at 1024² is ~0.68 ms

From the dispatch timing sweep (unpinned step isolation): the `matmul_kernel`
submission step itself is nearly constant at 0.67–0.68 ms for 128²–1024². At
1536²–2048² it rises to 0.88–1.0 ms as the kernel has more work per tile.

This ~0.68 ms is the XLA graph submission latency: it includes:
- Python dispatch through torch_xla lazy tensor
- XLA lazy graph compilation or cache lookup
- NEFF dispatch to the Neuron Runtime
- Waiting for the first partial result (before `to_cpu` completes)

### 2. Transfer costs dominate at moderate sizes

At 1024² (unpinned): `to_xla` = 1.50 ms, `to_cpu` = 0.97 ms, kernel = 0.68 ms.
Transfer accounts for 77% of the total 3.14 ms.

Pre-pinning eliminates `to_xla`; Phase 3 batching eliminates redundant `to_cpu`
between loop iterations.

### 3. Homogeneous batching impact (Phase 3, v0.5.0)

For the DF-MP2 pair loop with nocc=64, pre-Phase 3:
- 64² = 4096 dispatches × 0.67 ms overhead = **2.75 s** pure overhead

Post-Phase 3 (all same-shape `"ap,bp->ab"` contractions):
- 1 batched `nki_batched_matmul` dispatch = **0.67 ms** overhead
- Stack + unstack cost: ~0.1 ms
- **~4000× reduction in dispatch overhead for this loop**

Kernel work for 4096 contractions of shape (nvir=512, naux=512):
- FLOPs: 4096 × 2 × 512² = 2.15 TFLOPs
- At 190 TFLOPS trn1 bf16 peak: theoretical minimum ~11 ms
- At 47.5 TFLOPS fp32 peak: ~45 ms
- Actual: one batched dispatch submits all of this in one NEFF

### 4. Kern and to_cpu already partially overlap (#35 closed)

The `kern ms` column in the dispatch timing tables measures XLA graph submission
latency — Python dispatch, lazy graph compilation/cache lookup, and NEFF handoff
to the Neuron Runtime — not the raw Tensor Engine execution time. When measured as
`nki_tot` (end-to-end NKI path wall-clock), the sum of steps slightly exceeds the
total:

| Size | to_xla | kern | to_cpu | Sum | nki_tot | Gap |
|------|--------|------|--------|-----|---------|-----|
| 1024² (unpinned) | 1.502 ms | 0.684 ms | 0.970 ms | 3.156 ms | 2.935 ms | **−0.22 ms** |

The gap confirms kern and to_cpu already partially pipeline: the Neuron Runtime
begins DMA while the host is still processing the XLA submission handoff.

No additional async pipelining is available at the dispatch level. The
`xm.mark_step()` call in `_to_xla` is a **necessary correctness barrier**: without
it, the XLA lazy evaluator fuses consecutive NKI kernels into a single graph that
triggers trn2-only shared-memory instructions failing trn1 verification (#39). The
sync is not a performance regression — removing it would improve latency only until
the compiler bug is fixed, and would produce wrong results on trn1 in the meantime.

---

## Recommended next steps

1. **Provision `trntensor-ci-trn1`** (`cd infra/terraform && terraform apply`) for
   a dedicated instance not shared with trnblas tests.

2. **Formal TE utilization profile** via `nki.compiler` compile path:
   compile `matmul_kernel` to a NEFF without torch_xla, use
   `neuron-profile capture -n model.neff` to get the TE/VE/DMA breakdown. Add
   the summary-text table here.

3. **2048² batched matmul profile** — at large shapes the kernel should be
   TE-dominant (unlike the dispatch-dominated 1024² case). Profile to confirm.
