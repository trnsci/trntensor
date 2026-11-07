# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Device matmuls with 32 MiB or longer rows were silently wrong (#123,
  trnblas#112, aws-neuron-sdk#1457).** On the native TorchNeuron path (Trn2,
  SDK 2.33), a device matmul returns wrong values with no error (or aborts
  with NRT status 1203) once an operand's 128 DMA partitions span 4 GiB or
  more, i.e. `min(rows, 128) * row_bytes >= 2**32`, and that includes a
  contiguous left operand. Example: fp32 `(96, 128) @ (128, 8,389,632)` gave
  rel err 0.8. Affected: the contiguous-LHS `ao_to_mo_transform` aten route
  of #98 (0.23.0–0.25.0) when `nbasis * naux >= 8,388,608` (fp32), and
  device `einsum` matmul / bmm plans (`nki_matmul`, `nki_batched_matmul`)
  with such operands (0.17.0–0.25.0). These products are now cut into
  contiguous row blocks of the long operand, each block spanning at most 2 GiB:
  K is split (partial sums), or M / N (concatenated results). No operand is
  copied. Splitting K changes the fp32 summation order of the affected products
  only. Every other product is the same aten call as before (bitwise
  unchanged). Standard DF ratios (naux about 3 × nbasis) on one core do not
  reach it. The `_ATEN_TRANSPOSED_LHS_MAX_BYTES` comment said exactly 2**32
  bytes was correct. That was a lucky measurement (the operand sat above an
  identical copy), and the comment is corrected; the guard already started
  at 2**32.

## [0.25.0] — 2026-11-04

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: the blocked packed AO→MO kernel (#112, #113): `ao_to_mo_transform` on packed fp32 integrals stays on the device past the old SBUF budget (n_ao 3508 at 85 ms per chunk vs 8.6 s on the host), and `auto` picks it where the resident kernel does not fit.

### Fixed

- **Packed `ao_to_mo_transform` on the device past n_ao ≈ 1100–1300 (#112,
  trnsci/trnsci#115).** `ao_to_mo_trn2_packed_kernel` keeps C_occ, C_vir, one
  chunk's gather indices and a whole μ column of eri in SBUF. At n_ao 3508
  (alkane60 / cc-pVTZ, 87-column aux chunks) that needs 979 KiB per
  partition against a 176 KiB budget, and at n_ao 1658 it needs 282 KiB. Both
  cases fell back to aten, which unpacks a full square per chunk (4.3 GB at
  3508). That first call did not finish compiling in 19 minutes. The new
  `ao_to_mo_trn2_packed_blocked_kernel` streams every operand over blocks of
  up to 8 μ/ν tiles, and its PSUM accumulators live across the blocks. Its
  SBUF footprint (71–78 KiB at the measured shapes) does not depend on n_ao.
  The aux chunk width is the chunk itself up to 512 columns, otherwise a
  divisor in [64, 512]. So an 87-column chunk goes in without the
  128-column padding copy (3.1 GB at n_ao 3508). Under `"auto"`, fp32 packed
  integrals take it wherever the resident kernel does not fit, and
  `TRNTENSOR_AO_TO_MO_PACKED=auto|resident|blocked` overrides. Measured on
  Trn2 with one core, fp32 and synthetic packed chunks:
  - At n_ao 3508 / nocc 241 / 87 columns the kernel takes 85 ms per chunk at
    LNC=2 and 94 ms at LNC=1. The host fp32 route on 2 threads takes 8.6 s.
    The cold compile takes 162 s, and 26 s with the NEFF cache.
  - At n_ao 1658 / 161 / 384 it takes 56 ms against 6.1 s on the host, with
    a 128 s cold compile. At 2522 / 215 / 168 it takes 73 ms against 7.9 s,
    with a 195 s cold compile.
  - Its maximum error against an fp64 host reference is 3.2–4.4e-7 relative
    (3.9e-7 at 3508). The host fp32 route gets 5.4–6.6e-7.
  - A full alkane60-shaped pass through `AuxStream.ao_to_mo` took 43 s warm
    and 102 s including the first-call compiles (NEFF cache warm from the
    single-chunk runs). That is 97 chunks, 8400 aux and 207 GB uploaded. The
    96 full chunks reuse one host tensor, so no disk reads are included.
    The trnsci#115 run did this pass on the host in 673 s, reading the store
    from NVMe.
  - Where the resident kernel fits, the two kernels give bit-identical
    results, because they run the same per-element PSUM chains.

## [0.24.0] — 2026-11-04

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: trntensor.stream.AuxStream for DF integrals streamed over the aux index (#102), cached eager NKI handles, and the trnsci float64 policy with a dtype guard at the kernel boundary.

### Fixed

- **`mp2_energy`'s aten route chunks j as well as i (trnsci/trnsci#115).** It
  chunked only the occupied index i. Once one i row of `T` (`nocc · nvir²`)
  exceeded `_MP2_CHUNK_ELEMS`, every step held that whole row, 1.37 GB at
  nocc 161 / nvir 1461 / naux 6064 (C40H82 / aug-cc-pVDZ), plus its temporaries.
  On a 24 GiB Trn2 core, with only the 5.7 GB of `B` resident, the step ran out
  of device memory. Blocks are now at most `_MP2_CHUNK_ELEMS` elements, and the
  same case ran on one core in 222 s.
### Changed

- **`AuxStream` uploads a memory-mapped chunk to a device straight from the
  mapping (trnsci/trnsci#115).** The sequential read into RAM, added so that
  host row gathers stop faulting pages at random, now runs only for chunks
  computed on the host. On Trn2, a page-cached 1.6 GB chunk went from 2.9 GB/s
  (copy, then upload) to 12–14 GB/s (upload from the mapping). For water61 /
  aug-cc-pVDZ (114.6 GB of B, 49.7 GB streamed per SCF iteration over 4
  cores), the per-iteration upload wait went from 12.5 s to 4.3 s.
### Performance

- **Eager NKI kernel calls skip the per-call handle rebuild and the custom-op
  layer (same fix as trnblas#92).** On the native path a `trntensor::`
  custom-op call cost about 0.1 ms of host time before the kernel was queued,
  against 0.016 ms for aten. Most of it went to nki's `Kernel.__call__`,
  which rebuilds the torch handle on every call, plus the `torch.library`
  layer. `dispatch._launch` and the `mp2_energy_row` loop now call a
  `torch_neuronx.wrap_nki` handle built once per kernel; compile-time keyword
  arguments (`use_sr`, `nub`, `tp`, `tail`, `banks`) pass through unchanged.
  On trn2 (LNC=2), a 256x128x512 `matmul` launch drops from 0.143 to
  0.036 ms of issue time (e2e 0.222 to 0.128 ms), `_nki_mp2_energy_rows` with
  nocc=8 from 2.94 to 2.34 ms, and the `df_exchange` kernel pair's issue time
  from 0.78 to 0.36 ms. Results are bit-identical. `torch.compile`,
  requires-grad, int64 and tensor-subclass inputs keep the custom op.
  `TRNTENSOR_NKI_EAGER_HANDLE=0` turns this off;
  `scripts/bench_eager_handle.py` measures it.

### Added

- **`trntensor.stream`: DF integrals streamed over the aux index**
  (trnsolver#15). At nbasis 1658 the packed fp32 integrals were 33.4 GB and
  only fit split over a Trn2 chip's 4 logical cores; at nbasis ≈ 3300 they
  are about 270 GB.
  - `AuxStream(chunks, nbasis, devices=..., resident_bytes=..., prefetch=...)`
    holds `B` as `(npair, nq)` aux chunks (tensors, numpy memmaps or lazy
    loaders), keeps chunks resident up to a per-device budget and uploads the
    rest on every call from a loader thread, one chunk ahead per device, so
    uploads overlap compute. `jk()` builds packed J and K in one pass
    (`df_coulomb` / `df_exchange` per chunk, summed per device, then on the
    host); `ao_to_mo()` writes `B_ia^P` chunk by chunk into a host tensor or
    memmap. Chunks are dealt round-robin over `devices` (`neuron:0..3`).
  - `aux_chunk_width` / `split_aux`: 128-multiple chunks under 2 GiB by
    default, below the 4 GiB transposed-LHS limit (aws-neuron-sdk#1457, #97).
  - `pack_density` / `unpack_packed`: the packed-density helpers, for use on
    host tensors at large nbasis (the device gathers compile for > 40 min at
    nbasis 1658, trnsolver#87).
  - `mp2_energy_blocked`: DF-MP2 over occupied blocks of a host `B_ia^P`,
    for `B` larger than one device.
  - Memory-mapped chunks are read into RAM with one sequential read when
    loaded. Left mapped, the row gathers of `df_exchange` fault pages in at
    random, which was disk-bound on a disk-backed store.
  - CPU-validated against dense references; device overlap and scaling are
    still to be measured.

### Changed

- **float64 follows the trnsci float64 policy (#92, trnsci/trnsci#35).**
  Trainium computes float64 at fp32 precision, so a float64 operand on a
  kernel route cannot get float64 accuracy there. This applies to the matmul /
  batched-matmul kernels (`einsum`, `multi_einsum`) and to the chemistry
  kernels (`mp2_energy`, `ao_to_mo_transform`, `df_exchange`, `df_coulomb`):
  - **`"auto"`:** float64 under `auto` now computes in torch at float64
    accuracy (previously cast to fp32 on the NKI path). For the chemistry
    kernels the cast came in with #88 and was in 0.23.0. Through 0.22.0
    float64 reached them unchanged and returned inf. torch runs on the
    operands' own device; float64 host operands stay on the host.
  - **Explicit `set_backend("nki")` (or `TRNTENSOR_FORCE_BACKEND=nki`):**
    float64 is cast to fp32, the kernel runs, and the result comes back as
    float64. A one-time `UserWarning` per op says the precision is fp32.
    Through 0.23.0 the chemistry kernels cast without a warning; the matmul
    kernels got float64 operands unchanged.
  - **`TRNTENSOR_REQUIRE_NKI=1`:** float64 on a kernel route raises
    `NotImplementedError` (trnblas#69's convention), under explicit `"nki"`
    and `"auto"` alike. **Callers who ran float64 chemistry inputs under
    `TRNTENSOR_REQUIRE_NKI=1` (0.23.0: the kernel ran in fp32) now get an
    error.** Pass float32 operands to run the kernels, or unset the variable.
  - **Result dtype:** always the pytorch backend's dtype for the same inputs,
    including bf16 / fp16 and mixed-dtype promotion (`torch.promote_types`
    of the operands).

  - **float64 already on the `neuron` device:** under `"auto"`, "torch on
    the operands' device" would be device aten, which computes float64 at
    fp32 precision (measured on Trn2). float64 / complex128 operands on the
    `neuron` device are now copied to the host, computed in host torch at
    float64 accuracy, and the result is returned on the original device in
    float64 / complex128. This covers `einsum`, `multi_einsum` (which returns
    host tensors, as before), `gather` (device `torch.cat` rounds float64 to
    fp32), `einsum(dtype=float64)` on fp32 / bf16 device operands, the
    chemistry calls and the CP / Tucker / TT
    decompositions and reconstructions. One `UserWarning` per op per process:
    "float64 on the neuron device is computed on the host for float64
    accuracy (device arithmetic is fp32); pass float32 to stay on device".
    **New: `TRNTENSOR_FP64_ON_DEVICE=1`** keeps device aten (fp32 precision,
    no warning) for callers who prefer speed. **Under
    `TRNTENSOR_REQUIRE_NKI=1`, float64 / complex128 on the `neuron` device
    now raises `NotImplementedError` at every guarded entry point** (backends
    `"auto"` and `"nki"`, no host copy, ahead of `TRNTENSOR_FP64_ON_DEVICE`);
    before, only the kernel routes raised, and an `einsum` with no kernel
    route (below the size thresholds, or not a matmul) or a decomposition
    ran device aten at fp32 precision with no error. Host-resident float64 is
    unchanged, and `multi_einsum` no longer pins host float64 / complex128
    operands to the device under `"auto"`; through 0.23.0 shared operands
    were pinned, so device aten computed them at fp32 precision.
    `--backend auto --conformance-device neuron` now holds float64 and
    complex128 to the float64 tolerance.
    The routes that place their own compute follow the same rule:
    `trntensor.stream.AuxStream(..., dtype=float64)` and
    `mp2_energy_blocked` with a `neuron` device in `devices` keep the chunks
    and the arithmetic on the host (results still default to the first
    device given), and `pack_tril` / `unpack_tril` / `pack_density` /
    `unpack_packed` gather float64 / complex128 on the host, because on the
    neuron device both the gather (`index_select`, fancy indexing) and
    `torch.cat` round float64 to fp32 (measured on Trn2, rel err ~4e-8).

  Affected versions: the old behaviour is 0.23.0 for the chemistry kernels
  (the silent fp32 cast) and 0.23.0 and earlier for the matmul kernels
  (float64 unchanged). The new behaviour lands in the next release (0.24.0).
  Docs: "Precision and dtypes" in `docs/api/nki.md`.
  `tests/test_conformance.py` now holds float64 to the float64 tolerance
  for host inputs under `--backend auto`, as under `pytorch`.

### Fixed

- **One dtype guard at the NKI kernel boundary: float64, integer, bool and
  complex operands no longer reach a kernel that does not take them**
  (silent, trnsci/trnsci#35; the class of #87). `nki.dispatch` now has a
  per-kernel operand-dtype table (`KERNEL_OPERAND_DTYPES`). Every launch
  checks it and raises `TypeError` if a dtype is not listed: the native op
  and the eager-handle fast path (`_native_call`), the XLA call, the
  simulator, the `mp2_energy` row loop and the reduce-parallel all-reduce. A native op without an entry cannot be loaded.
  The host wrappers decide first:
  - `einsum` / `multi_einsum` (matmul and batched-matmul kernels; fp32, bf16,
    fp16, one shared dtype): integer, bool, complex and mixed dtypes run
    torch, as the pytorch backend does; float64 follows the policy above.
    Through 0.23.0 they reached the kernels under `set_backend("nki")`, or
    under `"auto"` above the size thresholds.
  - `mp2_energy`, `ao_to_mo_transform` (fp32 / bf16 kernels) and
    `df_exchange`, `df_coulomb` (fp32 kernels): float16 operands, and
    bfloat16 operands of `df_*`, are computed in fp32 (a lossless upcast).
    The result is returned in the aten route's dtype. float64 follows the
    policy above. Integer, bool and complex operands take aten. On the kernel
    route, bf16 `df_coulomb` raised in the host tail whenever `npair` was not
    a multiple of 256, then fell back to aten. Re-allowing bf16 on the `df_*`
    kernels waits on a Trn2 bf16 run (#92).
  - reduce-parallel all-reduce: partials other than fp32 / bf16 / fp16 are
    summed by `torch.distributed` instead of the NKI collective kernel.
    float64 partials are summed there at full precision.
  - Under `TRNTENSOR_REQUIRE_NKI=1`, a dtype that would run torch instead of
    the kernel raises `NotImplementedError` (trnblas's convention).
  The `df_half_packed` op's fake (meta) output is now float32, like the
  kernel's. Test: `tests/test_kernel_dtype_guard.py`.

- **`unpack_tril` / `pack_tril` on the neuron device at LNC=1 returned
  wrong data once the gathered output reached 4 GiB (#97).** Affected
  0.20.0–0.23.0: `unpack_tril`, `pack_tril`, and the packed-`eri` aten /
  legacy routes of `ao_to_mo_transform`, which unpack with `unpack_tril`: on
  the neuron device at LNC=1, one `index_select` along dim 0 whose output
  reaches 2^32 bytes can be silently wrong (about half the elements, at
  times NaN) or hit the runtime's 30 s execution timeout. Measured on Trn2:
  exact at 2^32 − 16 KiB, wrong at exactly 2^32, and 154 M mismatched
  elements for `unpack_tril` at nbasis 736 / naux 2304 (4.99e9 bytes). It
  is exact at LNC=2 and on the host. Device gathers are now split into row blocks of at most
  2 GiB of output and concatenated in order (exact). Host gathers are one
  `index_select` as before.
- **Complex `tucker_decompose` on the neuron device used `U^T`, not `U^H`
  (#89).** Affected 0.23.0 (the release that switched the core to
  `factors[mode].mH`, #73): `.mH` of a complex tensor is a lazy conj view,
  and device matmul ignores the conj bit (aws-neuron/aws-neuron-sdk#1455),
  so the core of a device-resident complex tensor was silently computed
  with the unconjugated factors. CPU results were correct. The conjugate
  transpose is now materialized with `resolve_conj()` before the mode
  product, as trnsolver's `factor._mH` does; real input is unchanged (no
  copy). The complex `cp_decompose` added in this release (#103) had the
  same pattern (`V.mH @ V`, `T_mode @ V.conj()`) and is fixed the same way.
  A CPU `TorchFunctionMode` test fails if any lazy conj view reaches a
  matmul, einsum, solve or pinv in either decomposition.

- **`cp_decompose` works in the input dtype, including float64 and complex
  (#103).** Affected 0.1.1–0.23.0: the random init and the Khatri-Rao buffer
  were float32 on the host, so float64 or complex input raised `expected m1
  and m2 to have the same dtype` on every backend (and a float64 warm start
  was truncated to float32 inside the Khatri-Rao product). The init, the
  Khatri-Rao buffer and the weights now follow `tensor.dtype` and
  `tensor.device`, warm-start factors are cast to them, and complex input
  uses the conjugated normal equations `(V^H V) A^T = (T_(k) conj(V))^T`.
  Weights are real (`tensor.real.dtype`). `nonneg=True` with complex input
  raises `ValueError`. Real float32 results are bit-identical.
- **`cp_reconstruct` returns the factors' dtype on the factors' device
  (#78).** Affected 0.1.1–0.23.0: it accumulated into a float32 host buffer
  (`torch.zeros(shape)`), so bf16 or fp16 factors came back as float32 and
  device factors raised a device mismatch. The buffer now has the promoted
  dtype of `factors` and `weights` and the factors' device. float32 and
  float64 results are unchanged.
- **`plan_contraction` fills `ContractionPlan.estimated_flops` (#83).**
  Affected 0.1.1–0.23.0: the documented multiply-add estimate was never set
  and always read 0. `matmul`, `bmm` and `torch` plans now carry
  `estimate_flops(subscripts, *operands)` (the product of all index sizes);
  `path` plans carry the multiply-adds of the planned binary steps, which is
  what runs and is less than the dense all-index product. Plans are cached by
  shape, so the field is exact for the cached shapes.
- **`einsum(..., out=)` now writes the result into `out` and returns it
  (#77).** Affected 0.3.0–0.23.0: `out` was only read as the `beta` addend
  and never written, so the documented in-place accumulation example
  `einsum(..., beta=1.0, out=acc)` silently left `acc` unchanged. `out` now
  behaves as in `torch.einsum` and cuTENSOR: it receives
  `alpha * contract + beta * out` (old contents not read when `beta == 0`),
  must have the result's shape (`ValueError` otherwise), must accept the
  result dtype under `torch.can_cast` (`RuntimeError` otherwise), and may
  alias an operand. Like `torch.einsum(out=)`, it raises `RuntimeError` when
  grad mode is on and an operand or `out` requires grad. Callers that reused
  `C` after passing it as `out=C` now see it overwritten; pass `out=C.clone()`
  to keep the old behaviour.

## [0.23.0] — 2026-11-02

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: **correctness fixes — upgrade recommended.** `kahan`/`dd` dropped the imaginary part of complex operands and rounded large integers (#69, #90); `beta=0` read `out=` (#71); complex `tucker_decompose` used Uᵀ (#73); `multi_einsum` returned fp32 results as fp64 (#75); `einsum(precision="dd")` missed its documented bound (#84); float64 chemistry kernels returned inf on device (#87); `ao_to_mo_transform` on device with operands over 4 GiB was silently wrong and is now guarded (#97). New: native AO→MO for any nocc, 1.8–3.5× faster than aten at nocc 160–384 (#94).

### Changed

- **`ao_to_mo_transform`'s Trn2 kernels now run any `nocc` (#80).** They
  needed `nocc ≤ 128`, so larger systems fell back to aten under `"auto"`.
  - Stage 1 runs occupied blocks of ≤ 128 over each eri block, so `eri` is
    read once whatever `nocc` is. `C_occ` and `C_vir` stay resident in SBUF.
  - Both stages move 512 columns per Tensor Engine pass (`512 // TP` ν in
    stage 1, `512 // TP` orbitals in stage 2), and stage 2 fetches and
    stores each group with one strided DMA.
  - With an odd aux-chunk count and `nocc > 128`, the two programs of an
    LNC=2 core split the occupied index instead of the aux chunks (the
    dispatcher appends one zero orbital when `nocc` is odd), so both
    physical cores still work.
  - The packed kernel gets the same treatment and keeps its in-place gather.
  - Trn2, LNC=2, fp32, warm: at nbasis 512 / naux 1536 the kernel is
    1.8–2.4× faster than aten at nocc 160–384 (full `eri`) and 2.7–3.5×
    with packed `eri`. It is 1.3–1.7× faster than launching the old kernel
    once per 128 orbitals. At LNC=1 it is 1.3–1.7× (full) and 2.3–2.6×
    (packed) faster than aten. Tables on #80.
  - At `nocc ≤ 128` it is 0.99–1.08× the speed of `main` (the 512-column
    passes), and the first-call compile is shorter.
  - With an odd aux-chunk count and `nocc > 128` the occupied split is
    1.3–1.6× faster than one program (e.g. 512/257/1152: 17.8 → 11.7 ms).
- **`"auto"` routing for `ao_to_mo_transform` (#80).**
  - The `nocc ≤ 128` condition is gone.
  - A full `eri` whose nbasis or naux is not a 128 multiple now stays on
    aten. The kernel pads a copy of the whole square first, and that copy
    costs more than aten (300/100/700: 5.6 ms against 2.0 ms). This also
    changes routing at `nocc ≤ 128`. Pass packed `eri` to keep the kernel:
    it pads nbasis only in its gather indices.
  - A packed `eri` with a padded nbasis needs 2e9 multiply-adds of stage-1
    work instead of 5e8.
  - The kernel's SBUF budget goes from 144 to 176 KiB (df_exchange's). At
    n_ao 970, nocc 161, packed naux 4608 it needs 158 KiB and runs in
    118 ms on Trn2 (LNC=2); aten ran out of device memory there.

### Added

- **Guard against the NKI trace-cache closure bug ([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454),
  trnsci/trnsci#35).** `tests/test_trace_cache_safety.py` (CPU) finds every
  `@nki.jit` kernel and fails if its body, or a helper it calls, reads a
  closure variable or a module global that is not a constant expression
  (reviewed exceptions are allow-listed with a reason), and checks each
  kernel's `__closure__` at runtime against a stub `nki`.
- Tests: `conftest.py` no longer turns the NKI trace cache off. Neuron tests
  run with it on, as under `import torch_neuronx`, with a fresh
  `NKI_TRACE_CACHE_URL` directory per session (no stale editable-install
  traces). `NKI_ENABLE_TRACE_CACHE=0` still turns it off.
- README: known platform issue note for #1454.

### Fixed

- **`ao_to_mo_transform` aten route on the device: wrong results from 4 GiB of
  `eri` (#97).** On TorchNeuron (torch-neuronx 2.14.3), a matmul whose left
  operand is a transposed view returns wrong values, without an error, once the
  right operand exceeds 2^32 bytes and the contraction dim is a multiple of 128.
  If the right operand came straight from the host it aborts instead
  (aws-neuron/aws-neuron-sdk#1448). Both `torch.einsum` steps of the aten route
  lower to that matmul, so at nbasis 768 / naux 2304 the result was off by
  about 1 relative. On the device, from 2^32 bytes of `eri` or of the
  half-transformed tensor, the two steps now run as `mm` and `bmm` with
  contiguous orbital transposes. At 768/64/2304 the error against float64 is
  2.2e-7 (was 1.06), and a warm call takes 16.1 ms (the wrong route took
  25 ms). Not fixed here: at LNC=1, the device `index_select` that builds a
  square `eri` over 4 GiB is itself wrong (it unpacks packed input for the aten
  route), so that case still returns wrong values or NaN (#97).

- **`einsum(precision="kahan")` rounded int32 values above 2**24 on the
  neuron device** (silent, #90). The device computes integer matmul in fp32.
  Since #70 integer operands contracted natively on any device. They now
  contract natively only on CPU/CUDA, and on the host for other devices.
  Unreleased (`main` after #70 only). Found by the conformance suite on Trn2
  (trnsci/trnsci#35).
- **NKI chemistry kernels on Trainium: float64 operands gave inf, and kernel
  routes changed the result dtype** (silent, #87). float64 tensors reached the
  fp32 kernels as is. `df_coulomb` (≥ 256 packed rows), `ao_to_mo_transform`
  and `df_exchange` came back inf under `set_backend("nki")`, or under
  `"auto"` at the sizes where it picks the kernels. float64 operands are now
  computed in fp32 and returned as float64, the documented fp64-on-Trainium
  behaviour (`df_*` 0.21.0–0.22.0, `ao_to_mo_transform` every release with
  its kernels). `mp2_energy` (row and legacy kernels), `ao_to_mo_transform`
  and `df_exchange` also returned float32 for float64 or bfloat16 callers.
  They now return the aten route's dtype. `df_exchange` with `naux == 0` (or
  `nocc == 0`) raised `ZeroDivisionError` in the route's tiling and now takes
  aten. Found by the conformance suite on Trn2 (trnsci/trnsci#35).
- **`einsum(precision="dd")` on float32 / float64 operands was less accurate
  than `"fast"`** (silent; 0.17.0 through 0.22.0, the CPU emulation; Trainium
  raises `NotImplementedError`). The CPU route split every operand into two
  bf16-precision pieces (~16 mantissa bits) and rounded float64 operands to
  float32, so the error was ~1e-5 instead of the documented K·u² + u_out
  (6e-8 fp32, 1.1e-16 fp64). float32 operands now accumulate in fp64 (exact
  products, inside K·u²). float64 operands form exact TwoProduct products and
  sum them in double-word arithmetic on `matmul` / `bmm` plans and `path`
  steps; generic `torch` plans accumulate in fp64. bf16 / fp16 `"dd"` and
  every `"fast"` result are unchanged (#84).
- **`einsum(..., out=C)` with `beta=0` (the default) read `C`**, so NaN or
  inf in the buffer (e.g. `torch.empty`) made the result NaN (silent, every
  backend). Affects 0.3.0 through 0.22.0. `beta == 0` now ignores `out`, as
  in BLAS, cuTENSOR and trnblas (#71).
- **`tucker_decompose` returned a wrong core for complex tensors** (silent,
  every backend): it projected with `Uᵀ` instead of `Uᴴ`, so even a
  full-rank reconstruction was off by ~140 %. Affects every release through
  0.22.0. `tt_decompose` now also accepts complex tensors instead of raising
  (#73). Real tensors are unchanged.
- **`precision="kahan"` / `"dd"` / `target_forward_error` dropped the imaginary
  part of complex operands and rounded large integers** (silent, every
  backend). Affects `"kahan"` in 0.4.0 through 0.22.0, and `"dd"` and
  `target_forward_error` in 0.17.0 through 0.22.0. Complex operands now
  accumulate in complex128, and integer operands contract natively, which is
  exact (#69).
- **`multi_einsum` changed result dtypes for mixed-dtype batches** (silent,
  every backend). Same-shape contractions were stacked without checking
  dtype, so float32 results came back float64. Affects 0.17.0 through 0.22.0.
  Batching now requires matching operand dtypes and devices (#75).

## [0.22.0] — 2026-11-01

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: **correctness fix:** in every release through 0.21.0, on every backend, `einsum` planned matmul/bmm without checking index order, so transposed or permuted outputs came back in the wrong layout (`ij,jk->ki` returned `A@B`, silently for square operands) and non-canonical batched operands raised. 228 of 364 two-operand cases were wrong or raised (#65, #64). Also: `df_exchange` tiling minimises partial SYRK tiles, so K at n_ao=970 drops from 118 to 94 ms (#66).

### Fixed

- **`einsum` returned the wrong layout for transposed or permuted outputs**
  (silent). Affects **every release through 0.21.0**, on every backend
  (PyTorch included). The planner chose the `matmul` / `bmm` strategy without
  checking index order, so `einsum("ij,jk->ki", A, B)` returned `A @ B`
  (shape `(i, k)`, and for square operands the transpose with no error), the
  permuted batched outputs (`bij,bjk->bki` etc.) returned `(b, i, k)`, and
  non-canonical batched operands (`bji,...`, `...,bkj`, `ibj,...`) raised.
  `matmul` is now planned only for outputs `[A free, B free]` and `bmm` only
  for exactly `bij,bjk->bik`; the rest run through `torch.einsum` (#64).

### Changed

- **`df_exchange` tiles naux to minimise the SYRK's partial aux tiles (#66, #60).**
  K at n_ao = 970 takes 93.8 ms instead of 118 ms at LNC=2, and 187 ms
  instead of 230 ms at LNC=1. The error against float64 is 6.2e-7 (was
  8.1e-7) relative to max |K|.
  - Aux chunks can now be 128 multiples with a short last chunk of
    `naux mod width`. Before, the width had to divide naux.
    `df_half_packed_kernel` takes `tp` and `tail` and gathers from the packed
    rows viewed as `(npair·naux / W, W)`, `W = gcd(naux, tp)`. `B` is still
    not copied.
  - `df_half_tiling` first minimises the 128-row tiles over the SYRK
    launches (`df_syrk_tiles`), because each partial tile there costs a full
    one. Next it prefers a divisor of naux, then the old order.
    - At naux = 4686 that means chunks of 256 with a 78 tail, 37 tiles
      instead of 44.
    - n_ao = 250 and 490 keep their previous tilings, which already reach
      the minimum.
  - `df_groups` folds a lone short last chunk into the previous launch group.
  - The launch-1 SBUF budget is now 176 KiB (was 160 KiB), so that
    TP = 256, NUB = 8, NVC = 16 fits at n_ao = 970. This tiling is 7 ms
    faster than NVC = 8.
  - naux with no divisor in [32, 512] now runs the kernels, not aten.
  - `TRNTENSOR_DF_K_TP` forces the chunk width.

## [0.21.0] — 2026-11-01

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: new fused DF exchange and Coulomb builds, `trntensor.df_exchange` / `df_coulomb`, which read packed integrals in place without building the square: K at n_ao=970 is 2.6× faster (304 → 118 ms) at unchanged error, and DF-RHF SCF at n_ao=970 converges in 4.71 s instead of 9.68 s (#62, #60).

### Added

- **`trntensor.df_exchange(B, C_occ, alpha=1.0)`: the DF exchange matrix
  straight from packed three-centre integrals (#60).**
  `K = alpha·Σ_{i,P} X X^T` with `X[i,ν,P] = Σ_λ C_occ[λ,i] B[λν,P]`; for a
  closed-shell SCF, `alpha=2`.
  - On the native TorchNeuron path it is two NKI launches per group of aux
    chunks. `df_half_packed_kernel` gathers the packed rows in place
    (batched indirect DMA, as in `ao_to_mo_trn2_packed_kernel`) and forms
    the half-transformed `Xt[P, i, ν]` with the aux index on partitions.
    `df_exchange_syrk_kernel` accumulates K in PSUM over 128-row aux tiles,
    computes only the upper-triangle strips and mirrors them on the Tensor
    Engine. No `(n, n, naux)` square is built, `B` is not copied or padded,
    and both physical cores work at LNC=2 (`program_id` split, no grids).
  - `"auto"` uses it for device-resident packed `B` from
    `nocc·nbasis²·naux` = 1e7; `TRNTENSOR_DF_EXCHANGE=auto|trn2|aten`
    overrides. Elsewhere it is a row gather, one einsum and one GEMM.
  - Trn2, one logical core, PySCF C_kH_2k+2 cc-pVDZ / cc-pvdz-jkfit, fp32,
    against the unpack + einsum + `trnblas.syrk` K of trnsolver's DF-RHF
    example: n_ao = 250 / 490 / 970 went 3.17 → 2.18, 15.5 → 9.33 and
    304 → 118 ms at LNC=2 (1.5x / 1.7x / 2.6x), and 4.71 → 3.46, 25.9 →
    16.9 and 424 → 230 ms at LNC=1. K error against float64 is unchanged
    (8.1e-7 vs 8.3e-7 relative to max |K| at 970).
  - In that example (with `df_coulomb` below, LNC=2, time to convergence)
    the SCF went 352 → 229 ms at n_ao = 250, 756 → 587 ms at 490 and
    9.68 → 4.71 s at 970 (J/K stage 365 → 163 ms per iteration).
- **`trntensor.df_coulomb(B, D)`: the DF Coulomb matrix from the same packed
  integrals (#60).** Two NKI launches each stream `B` once in contiguous
  128-row tiles with one fused Vector Engine instruction per tile (`d = Bᵀp`,
  then `J = B d`). `TRNTENSOR_DF_COULOMB=auto|trn2|aten`. Against the two
  GEMVs per aux block it replaces: 3.81 → 2.90, 12.6 → 8.83 and 86.7 →
  46.0 ms at n_ao = 250 / 490 / 970 (LNC=2), 126 → 81.7 ms at 970 (LNC=1),
  error 2.8e-7 vs 1.9e-7 relative to max |J|. It also runs on a single
  8.8 GB block, where the aten GEMV needed a 26.7 GB graph. A packed
  `(npair,)` density gives a packed J, so blocked callers pack once.
- Kernels: `df_half_packed_kernel`, `df_exchange_syrk_kernel`,
  `df_coulomb_d_kernel`, `df_coulomb_j_kernel` (gridless, `num_programs()`
  split, MLIR-verified at lnc=1 and lnc=2). Tiling helpers
  `df_half_tiling`, `df_group_chunks`, `df_exchange_blocks`.

## [0.20.0] — 2026-10-31

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: packed ERI input for `ao_to_mo_transform` (DF-MP2 14–18% faster, #55); pinned-memory downloads (1.6–3.7×, #57); large host-input calls offload to the device above measured thresholds (umbrella DF-MP2 at xxl 29 s → 0.4 s, #58).

### Changed

- **Host inputs run on the device when that is faster (#56).** On the
  native TorchNeuron path under `"auto"`, host operands are uploaded, run on
  the device by the device-resident policy and downloaded into cached host
  memory once the work reaches a measured threshold:
  - `mp2_energy`: `nocc²·nvir²·naux` ≥ 1.5e8;
  - `ao_to_mo_transform`: `nocc·nbasis²·naux` ≥ 2e8 for a full `eri`, 3e7 for a packed one;
  - `einsum` generic and multi-operand plans: ≥ 5e8 multiply-adds and at least 0.5 multiply-adds per byte moved. A path uploads once and keeps its intermediates on the device.
  - These thresholds scale with `torch.get_num_threads() / 2`, as in trnblas#65. Matmul/bmm keep `TRNTENSOR_MIN_NKI_FLOPS_NATIVE` (5e8, re-measured).
  - Trn2, 2 host threads, warm: `mp2_energy` 646 -> 16 ms at 32/224/1024 and 15.8 s -> 135 ms at 64/448/1536. `ao_to_mo_transform` at 512/1536/64 went 574 -> 137 ms full and 979 -> 80 ms packed. The umbrella DF-MP2 profile in host-auto mode at 512/1536/64 went 29.9 s -> 0.37 s (fused pipeline) and 28.9 s -> 0.41 s (example pipeline, with trnblas 0.9 routing).
  - `TRNTENSOR_HOST_ROUTE=host|device|auto` overrides the choice. `set_backend("pytorch")` and `set_backend("nki")` keep their routes.
  - If the upload or the device call fails, the call runs on the host.
- **float64 and complex host matmul/bmm stay on the host** under native
  `"auto"`. Before, they were offloaded from 5e8 multiply-adds, and the
  device computes in at most fp32: a float64 `einsum("ij,jk->ik")` at 1024³
  came back with 1.4e-7 relative error. It is now exact float64.

- **Host results at the host-link rate (#56).** On the native TorchNeuron
  path, results of 1 MiB and up that come back to the host (`einsum` /
  `multi_einsum` with host operands on the `aten`-on-device route, kernel
  launches, `from_device`) are copied into memory from the accelerator's
  caching host allocator (`torch.empty(..., pin_memory=True)`), the fix
  trnblas#64 made. A copy into fresh pageable memory runs at about 4 GB/s on
  Trn2 because every page faults; into that memory it runs at 13-14 GB/s.
  Each call still returns a fresh tensor, and results are bit-identical.
  `TRNTENSOR_PINNED_OUTPUT=0` turns it off.
- **`einsum`'s `alpha` / `beta` / `out` epilogue runs in place** on a fresh
  result (same float ops, same bits). Out of place each step allocated and
  page-faulted another result-sized host tensor.
- Trn2, host operands, warm medians: `einsum("ij,jk->ik")` 4096³ 32 -> 20 ms,
  8192×1024×8192 80 -> 29 ms; with `alpha=2, beta=1, out=C` 69 -> 29 ms and
  230 -> 63 ms; 16×1024³ bmm 30 -> 17 ms; `from_device` of 256 MB 68 -> 19 ms.
  `ao_to_mo_transform` and `mp2_energy` with host inputs run on the host and
  are unchanged.

### Added

- **Packed (μ≥ν) three-centre integrals in `ao_to_mo_transform` (#54).**
  - A 2-D `eri` is read as the packed lower triangle: `(npair, naux)` (row `μ(μ+1)/2 + ν`, the `trnblas.parallel.pack_tril` layout), or PySCF's `_cderi` `(naux, npair)` as is. 3-D calls are unchanged.
  - New `trntensor.pack_tril` / `trntensor.unpack_tril`.
  - A host `eri` with device `C_occ` is now uploaded to that device; this used to fail on mixed devices.
  - New native kernel `ao_to_mo_trn2_packed_kernel`. It gathers its stage-1 blocks from the packed rows with batched indirect DMA, so the full square is never written to device memory. Bit-identical to `ao_to_mo_trn2_kernel`, and as fast at LNC=1 and LNC=2.
  - DF-MP2 end to end on Trn2 (umbrella `df_mp2_profile.py`, one LNC=2 core): 342 → 281 ms at ao/aux/occ 512/1536/64, 157 → 135 ms at 384/1152/48, with identical energies. The upload of `eri` halves (119 → 62 ms).
  - Pinned host memory does not speed up the upload (measured), so there is no pinned staging path.

## [0.19.0] — 2026-10-31

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: new native MP2-energy and AO→MO kernels beat aten by 1.25–1.9× and 1.3–1.6× in the `auto` region; native NKI matmul/bmm kernels deprecated (#52).

Resolves #51: on the native TorchNeuron path (Trn2/Trn3), the AO→MO and MP2-energy kernels are rewritten for Trn2 and now beat aten, and the NKI matmul/bmm kernels are retired in favour of aten.

### Added

- **`mp2_energy_row_kernel`, the native MP2-energy kernel (#51).**
  - One launch per occupied row, using the pair symmetry `e_ij = e_ji`.
  - Per pair, one GEMM fills `T` in SBUF; the `Tᵀ` blocks are Tensor Engine transposes. `T` never reaches device memory.
  - Gridless launch: at LNC=2 each physical core takes half of the `j` window (`num_programs()` / `program_id`), and it also runs at LNC=1.
  - The design follows trnblas `pair_energy_sym` (trnblas#55, #58).
  - Native `"auto"` uses it for device-resident `B` once each row launch carries at least 3 GFLOP. On Trn2 it measured 1.25–1.9× faster than aten: for example 0.69 s against 1.30 s at `nocc=96, nvir=672, naux=2304`.
  - Near a full device the kernel's input staging (about 1.5× the padded `B`) needs more memory than aten's 64 MiB chunks. `"auto"` keeps aten when live allocations plus that staging would leave less than 3 GiB of the core free, and remembers shapes whose launch failed.
  - `TRNTENSOR_MP2_ENERGY=auto|rows|aten|legacy` overrides the choice.
- **`ao_to_mo_trn2_kernel`, the native AO→MO kernel (#51).**
  - Two plain GEMM stages with the contraction index on the partition axis, so there are no DMA transposes.
  - The auxiliary index is split across both physical cores at LNC=2.
  - Native `"auto"` uses it for device-resident `eri` with `nocc ≤ 128` and stage-1 work of at least 5e8 multiply-adds (about nbasis ≥ 256). There it measured 1.3–1.6× faster than the two aten einsums on Trn2, at nbasis 256–768. Smaller problems stay on aten.
  - `TRNTENSOR_AO_TO_MO=auto|trn2|aten|legacy` overrides the choice.
- `scripts/bench_native_kernels.py`: a kernel-level A/B harness (aten against the original and the Trn2 kernels), with `--hold-gb` for the memory-pressure regime of aws-neuron-sdk#1447.

### Deprecated

- **On the native path, the NKI `matmul` / `batched_matmul` kernels and the original `mp2_energy` / `ao_to_mo_transform` kernels (#51).** `set_backend("nki")` still runs the matmul/bmm kernels there, with a one-time `FutureWarning`; a later release will route them to aten.
  - A Trn2 rewrite of matmul used both cores, Tensor Engine transposes and B-tile reuse. It was 1.2–4.3× faster than the original kernel, but still slower than aten at every size (4096³ fp32: 4.10 ms against 3.80 ms; bf16: 1.61 ms against 1.43 ms). It was not kept.
  - Native `"auto"` already used aten for these.
  - The XLA path (Trn1) keeps all four original kernels unchanged and does not warn.

### Changed

- Under `set_backend("nki")` on the native path, `mp2_energy` and `ao_to_mo_transform` run the Trn2 kernels instead of the original ones.

## [0.18.0] — 2026-10-30

Native TorchNeuron dispatch (Trn2/Trn3) alongside the PyTorch/XLA path (Trn1, maintenance); validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0) — see trnsci/trnsci#35.

Headline performance: native `"auto"` routes to aten on the device, and the planner contraction-order fix takes the DF-MP2 AO→MO transform (`mi,na,mnP->iaP`, nb=256) from 70 to 1.16 ms (#49).

### Changed

- **Native dispatch thresholds are now measured (#41).** They were measured
  on trn2.3xlarge (SDK 2.33, NKI 0.7.0) with `scripts/bench_native_einsum.py`
  and `scripts/autotune_dispatch.py`.
  - On device-resident operands, aten `torch.matmul` / `torch.bmm` beat the
    NKI matmul and batched-matmul kernels at every size (128³ to 4096³, fp32
    and bf16). The native pinned threshold now defaults to `never`, so
    device work runs as aten on `neuron`.
  - Host operands at or above `TRNTENSOR_MIN_NKI_FLOPS_NATIVE` (now
    500 M multiply-adds, previously a provisional 1 G) are offloaded to
    aten on the device instead of to the NKI kernel.
  - Thresholds accept `never` / `inf`.
  - The XLA path is unchanged.
- **Native `"auto"` runs the fused quantum primitives as aten.** This covers
  `mp2_energy` and `ao_to_mo_transform`. On Trn2 the AO→MO kernel measured
  6–20× slower than aten, and the MP2 kernel 1.0–1.5× slower.
  `set_backend("nki")` still runs both kernels, and the XLA path is
  unchanged.
- **`mp2_energy` reference path is vectorized.** Without NKI it now uses
  aten ops chunked over `i` instead of a nocc² Python loop. That is 10–40×
  faster on device. On an Apple M-series CPU it is 3× faster at small sizes
  and about even (±10%) at `nocc=32, nvir=128, naux=512`.
- **`ao_to_mo_transform` reference path** is now two explicit binary
  contractions.

### Fixed

- **Greedy contraction path no longer starts with an outer product.** For
  example, `einsum("mi,na,mnP->iaP")` used to contract `mi,na` first,
  building a 4-index intermediate. Pairs that share an index are now
  preferred.
- **`scripts/autotune_dispatch.py` now reports M·K·N**, the quantity dispatch
  compares, where it used to report 2·M·K·N. On the native path it now
  derives both thresholds: host vs device offload, and NKI vs aten on device.

## [0.17.0] — 2026-10-28

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron device) — see trnsci/trnsci#35.

### Added

- **Native PyTorch dispatch on TorchNeuron (#41, trnsci/trnsci#35).** On
  TorchNeuron (Neuron SDK 2.33+, `torch-neuronx` 2.14, PyTorch 2.14,
  Trn2/Trn3) every host-wrapper NKI kernel is a PyTorch custom op registered
  with `torch_neuronx.nki_op` and launched on `device="neuron"`, with no
  `torch_xla` import: `trntensor::matmul`, `batched_matmul`, `mp2_energy`,
  `ao_to_mo_transform`. Kernel bodies are unchanged. The PyTorch/XLA path
  (Trn1) is unchanged.
  - The dispatch path is detected on the first kernel launch, not at
    import: native when `torch_neuronx` exports `nki_op` and
    `torch_neuronx.device_count() > 0`, XLA when `torch_xla` is installed,
    otherwise none. `TRNSCI_NKI_DISPATCH=auto|native|xla` (suite-wide)
    forces a path; forcing an unavailable one raises.
  - The ops are registered lazily (`trntensor/nki/_native.py` is imported
    only on the native path). Every launch goes through one helper,
    `dispatch._launch`. That covers `einsum` (matmul/bmm/path steps),
    `multi_einsum` (batched and per-contraction), `ao_to_mo_transform`,
    `mp2_energy`, and `precision="sr"` (which passes `use_sr=True` to the op).
    `precision="kahan"` stays a host fp64 contraction, and `"dd"` still
    raises on NKI hosts (trnblas#22). The reduce-parallel `nki.collectives`
    all-reduce from Phase 4 is unchanged.
  - New `to_device` / `from_device` place tensors on the active path's
    device (`neuron` or XLA). `to_xla` / `from_xla` stay public. On the
    native path `to_xla` places on `neuron`, so existing residency code keeps
    working. `multi_einsum` pre-pins shared operands on either device.
  - FLOP dispatch thresholds are path-aware. The XLA values are unchanged
    (2 GFLOPs unpinned, 1 GFLOPs pinned). The native path uses provisional
    1 GFLOPs / 128 MFLOPs, because eager native dispatch costs about 0.2 ms
    per launch. Override them with `TRNTENSOR_MIN_NKI_FLOPS_NATIVE[_PINNED]`
    or the `trntensor_min_nki_flops_native[_pinned]` autotune cache keys.
    `scripts/autotune_dispatch.py` measures and writes the native keys on a
    TorchNeuron host.
  - Dormant deprecation hook: with `TRNSCI_XLA_DEPRECATION=1` (off by
    default), selecting the XLA path and calling `to_xla` / `from_xla` each
    emit a one-time `DeprecationWarning`. Migration: replace `to_xla` with
    `to_device` and `from_xla` with `from_device` (docs/api/nki.md).
  - The `mp2_energy` CPU fallback copies `neuron`-resident operands to the
    host before its scalar loop, then moves the result back.
  - New extras: `[torchneuron]` (`torch>=2.14`, `torch-neuronx>=2.14`,
    `nki>=0.6`) and `[neuron]` for the XLA lineage (`torch-neuronx>=2.9`,
    `nki>=0.3.0`).
  - Tests: `tests/conftest.py` sets `NKI_ENABLE_TRACE_CACHE=0` by default.
    New CPU tests in `tests/test_native_dispatch.py` cover path selection,
    launch routing for every entry point, lazy registration, op schemas,
    fake shapes, thresholds, and the deprecation hook. New
    `@pytest.mark.neuron` tests in `tests/test_nki_native.py` have not yet
    run on hardware. `TestXlaResidency::test_matmul_stays_on_xla` is now
    path-aware: it expects `neuron` on TorchNeuron and `xla` on Trn1, and
    still checks residency.

- **Reduce-parallel sharding on real multi-rank all-reduce (Phase 4 prototype).**
  Before this change, reduce-parallel and mixed output/reduce-parallel einsums
  raised `NotImplementedError` whenever `HAS_NKI` was true.
  - Under an initialized `torch.distributed` group with
    `world_size == n_shards > 1`, rank `r` now computes only shard `r`'s
    partial on its own device. The partials are combined with a real
    all-reduce. Single-process runs keep `_mock_allreduce`.
  - The backend is chosen by `trntensor.parallel.allreduce_backend`. `auto`
    selects `nki` under a `"neuron"` process group, `torch` under any other
    group, and `mock` otherwise. Override it with
    `TRNTENSOR_ALLREDUCE=mock|torch|nki|auto` or `parallel.set_allreduce_backend()`.
    An explicit `torch`/`nki` request that cannot be honored raises
    `RuntimeError`; it never falls back to replicated compute.
  - `trntensor.nki.collectives.nki_all_reduce_sum` is an `@nki.jit` kernel
    using `nki.collectives.all_reduce(op=add)` on named `shared_hbm` buffers,
    invoked via `torch_neuronx.wrap_nki(k)[(lnc,)]`. This is the pattern
    measured bit-exact on Trn2 (SDK 2.33, NKI 0.7.0, LNC=2) in trnsci spike B4.
    `private_hbm` gave wrong values at LNC=2.
  - `scripts/run_reduce_parallel_multirank.py` runs multi-rank checks: one
    process per rank, core pinning, hard timeouts, and per-rank logs. It
    checks `ij,jk->ik` with `j` sharded at n = 512–2048 against the
    single-device result (fp32 rel err ≤ 1e-5) and requires all ranks to
    agree. It also reports bit-exactness against a fixed-order sum, plus
    timings. Pass `--cpu` for a gloo pre-flight. Rank cores must be an
    aligned pair or quad (0-1, 2-3, 0-3).

### Fixed

- **Precision promises (Trn2 spike B5, #42).**
  - `precision="kahan"` on Neuron-device (or other non-fp64) operands now
    contracts on the host in true fp64 and moves the result back. Before,
    the "fp64" einsum ran on the device, which has no fp64 datapath. On Trn2
    fp32 results measured ~1.4 ulp (max componentwise 8.4e-8 at N=128),
    above the K·2⁻⁵³ + u bound that a real fp64 accumulation plus one
    rounding satisfies. The bound was right; the extra rounding was the bug.
  - Error bounds now include the final rounding of the result to its dtype
    (`u_out`): `"fast"` K·u + u_out, `"sr"` √K·u + u_out, `"dd"` K·u² + u_out,
    `"kahan"` K·2⁻⁵³ + u_out. `select_precision_for_error` selects on these
    bounds. When the target is below the floor (≈ 3.9e-3 for bf16 results,
    6.0e-8 for fp32), it emits a one-time `RuntimeWarning` and returns
    `"kahan"` instead of silently missing the target. It does not raise.
    Targets below K·2⁻⁵³ still raise `ValueError`. Because of the `u_out`
    term, some bf16 targets near 2⁻⁸ now select a more accurate mode than
    before.

- **NKI 0.7 / neuronx-cc 2.28 kernel compile on Trn2 (#41).**
  - `ao_to_mo_transform_kernel`: step 1 now computes the intermediate
    already transposed (`eri` tile as the stationary operand, ν-major
    scratch), so step 2 loads its stationary tile with a plain DMA. The
    former `nl.load_transpose2d` of a strided HBM slice was rejected by the
    BIR verifier (`checkDMATranspose`: "transposing dimensions should be
    continuous"). The ν loop is now `TILE_K`-wide, so the dispatcher pads
    `nbasis` to a multiple of 128 instead of 512.
  - `mp2_energy_kernel`: removed the `(128, 1) → (1, 128)` reshape
    ("Partition dim size must be preserved") and the cross-partition
    `nl.sum(axis=(0, 1))`. The dispatcher now passes `B` as
    `(nocc, naux, nvir)` (plain loads, no DMA transposes) and ε_occ / ε_vir
    pre-broadcast across the 128 partitions; the kernel row-reduces per
    lane and finishes with a `accᵀ · ones` matmul on the Tensor Engine.
  - `matmul_kernel` / `batched_matmul_kernel` `use_sr=True`: NKI 0.7's
    `nisa.activation` has no `out_dtype` / `round_mode` kwargs; the
    downcast now calls `nisa.activation(dst=..., op=nl.copy, data=psum)`.
    Stochastic rounding is a runtime setting on this toolchain, not a
    per-instruction one.
- `tests/test_nki_kernels.py`: replaced the stale `test_oversize_raises`
  (nvir > 128 no longer raises since M-tiling landed) with a correctness
  check at nvir = 200.

### Added

- `trntensor.forward_error_bound(mode, dtype, K)`: the componentwise error
  bound each precision mode promises, including output rounding (#42).
- **Stochastic-rounding runtime guard (#41).** On NKI 0.7 (Neuron SDK
  2.33) the `use_sr` flag no longer chooses the rounding mode. On Trn2,
  with `NEURON_RT_STOCHASTIC_ROUNDING_EN` unset or `0`, both
  `use_sr=True` and `use_sr=False` round to nearest even and are
  deterministic. With `=1`, both round stochastically. The Neuron Runtime
  sets the mode for the whole process when it initializes.
  `nki_matmul` / `nki_batched_matmul` now emit a one-time `RuntimeWarning`
  when SR is requested on the native NKI path without
  `NEURON_RT_STOCHASTIC_ROUNDING_EN=1`, then still compute
  (round-to-nearest). This covers `precision="sr"` in `einsum` /
  `multi_einsum` and `target_forward_error=` when it picks `"sr"`. The CPU
  fallback and `nki.simulate` paths are unaffected. New
  `trntensor.nki.dispatch.sr_runtime_enabled()`.
  - Docs (`docs/api/einsum.md`, `docs/api/plan.md`) and docstrings explain
    that the variable has to be set at process start, and that with it set
    *every* downcast in the process rounds stochastically, including
    `precision="fast"`.
  - Tests: `tests/test_sr_runtime.py` (CPU, native path stubbed);
    hardware `TestStochasticRoundingRuntime` in `tests/test_nki_kernels.py`
    (markers `neuron` + `neuron_sr`), gated by the `sr_runtime` fixture.
    The fixture skips unless the variable was `1` at process start.
- `tests/test_nki_compile.py` (`nki_simulator` marker): runs the NKI
  frontend (trace → MLIR → verify) on CPU for every kernel at trn1, trn2,
  and trn2 lnc=2. Catches the verifier errors `nki.simulate` misses.

## [0.16.0] — 2026-04-29

### Added

- **`target_forward_error=` for `multi_einsum()`** — extends the automatic
  precision contract from v0.15.0 to multi-contraction calls.

  For the per-contraction fallback path, each ``einsum()`` call resolves its
  own K independently, so heterogeneous contractions (different subscripts or
  operand shapes) each get the cheapest mode that satisfies the target for
  *their* K. For the homogeneous batching path (same subscript, same shapes),
  the mode is resolved once from the first contraction's K and dtype — which
  equals all others in the batch.

  Mutually exclusive with explicit ``precision=``, same as ``einsum()``.

## [0.15.0] — 2026-04-27

### Added

- **`target_forward_error=` kwarg for `einsum()`** — automatic precision
  selection. Instead of specifying a precision mode manually, callers supply
  a relative forward error bound and trntensor picks the cheapest mode whose
  theoretical error bound satisfies it:

  ```python
  einsum("ij,jk->ik", A, B, target_forward_error=1e-4)
  ```

  Selection order (cheapest first): ``"fast"`` → ``"sr"`` → ``"dd"`` →
  ``"kahan"``. Error bounds used (Higham 2002 §3, Connolly–Higham–Mary 2021):

  - ``"fast"``:  K · u (Wilkinson worst-case)
  - ``"sr"``:    √K · u (expected; mean-zero — statistical, not worst-case)
  - ``"dd"``:    K · u² (Ozaki-2 worst-case)
  - ``"kahan"``: K · u_f64 (FP64 Wilkinson worst-case)

  Where K = product of contracted dimension sizes and u = unit roundoff for
  the operand dtype (BF16: 2⁻⁸, FP16: 2⁻¹¹, FP32: 2⁻²⁴, FP64: 2⁻⁵³).
  ``"sr"`` and ``"dd"`` are only considered for BF16 and FP16 inputs.

  Raises ``ValueError`` when both ``precision=`` and ``target_forward_error=``
  are given, or when no available mode can achieve the target.

- **`select_precision_for_error(dtype, K, target_forward_error)`** — the
  selector as a public API function (``trntensor.select_precision_for_error``).
  Useful for inspecting which mode would be chosen without running the full
  contraction.

### Notes

``target_forward_error=`` is a first step toward the suite-level
``target_forward_error`` contract described in the architecture brief. The
current implementation uses static bounds (Wilkinson/Connolly–Higham–Mary);
adaptive error estimation (measuring actual residuals) is future work.

## [0.14.0] — 2026-04-24

### Added

- **`precision="dd"` on CPU — Ozaki-2 splitting** — lifts the unconditional
  ``NotImplementedError`` for double-double accumulation on CPU.

  ``_ozaki_split_cpu(A, k)`` splits a FP32 tensor into ``k`` BF16-precision
  summands using the Ogita–Rump–Oishi error-free split: at each step,
  ``s = round_bfloat16(remainder)`` and ``remainder -= s`` (exact by Sterbenz's
  lemma since ``|remainder|/|s| ∈ [0.5, 2]``). ``_ozaki_matmul_cpu(A, B, k=2)``
  computes ``k²`` FP32 matmuls of the summand pairs and sums in FP32, achieving
  forward error O(K · u_bf² · |C|) vs O(K · u_bf · |C|) for a single BF16
  matmul — substantially more accurate for large K.

  On CPU (``HAS_NKI=False``): ``precision="dd"`` routes the ``matmul`` strategy
  through ``_ozaki_matmul_cpu``; the ``path`` strategy propagates ``precision="dd"``
  recursively so each binary matmul step uses Ozaki; ``bmm`` and ``torch``
  strategies fall back to kahan (fp64 promotion).

  On Trainium (``HAS_NKI=True``): ``NotImplementedError`` preserved — the fused
  NKI version requires trnblas Phase 2 double-double GEMM kernels (trnblas#22),
  which hold both the main and correction PSUM tiles simultaneously in one NKI
  program rather than dispatching four separate kernel calls.

  Same named-mock pattern as ``_stochastic_round_cpu`` (SR, v0.11.0) and
  ``_mock_allreduce`` (reduce-parallel, v0.10.0).

### Notes

The remaining ``NotImplementedError`` for ``precision="dd"`` on Trainium gates
on trnblas#22. The CPU contract is now fully testable.

## [0.13.0] — 2026-04-22

### Fixed

- **`precision="sr"` now honored for the `bmm` strategy** — `_execute_bmm`
  was calling ``nki_batched_matmul(A, B)`` without ``use_sr``, ignoring
  ``plan.precision``. The NKI kernel already supported ``use_sr`` since
  v0.11.0; the dispatch layer just didn't thread it through.

### Added

- **`precision=` kwarg for `multi_einsum()`** — all contractions in a
  ``multi_einsum`` call now share the same precision mode. The homogeneous
  batching path (``_try_batched_multi_einsum``) passes ``use_sr`` to
  ``nki_batched_matmul`` when ``precision="sr"``; ``"kahan"`` and ``"dd"``
  fall through to the per-contraction loop which handles them correctly.

### Notes

``precision=`` is now uniformly honored across all dispatch paths: single
matmul (`einsum` → `_execute_matmul`), batched matmul (`einsum` →
`_execute_bmm`), path execution (`einsum` → `_execute_path` → recursive
`einsum`), and multi-contraction (`multi_einsum`).

## [0.12.0] — 2026-04-22

### Added

- **Mixed output-parallel + reduce-parallel sharding** — ``einsum()`` now
  supports einsums where some ``ShardedTensor`` operands are partitioned along
  an output dimension (output-parallel) and others along a contracted dimension
  (reduce-parallel) in the same call.

  The dispatch uses a nested loop: for each output shard ``s``, partial
  contractions over all reduce shards are accumulated via
  :func:`~trntensor.parallel._mock_allreduce` and the result becomes output
  chunk ``s``. The return value is a ``ShardedTensor`` partitioned along the
  output dimension.

  Operands that are neither sharded but share the contracted index are
  co-partitioned automatically (same behavior as pure reduce-parallel).
  Output-parallel operand chunks are also co-partitioned along the contracted
  index when they span it.

  On Trainium (``HAS_NKI=True``) the gate is unchanged — the inner all-reduce
  still requires ``nki.collectives`` (Neuron SDK 2.30+).

  3-operand and higher chains work naturally: the inner recursive ``einsum()``
  call receives dense operands and dispatches through the normal path.

### Notes

Mixed output-parallel + reduce-parallel with ``HAS_NKI=True`` raises
``NotImplementedError`` with the same ``nki.collectives`` message as pure
reduce-parallel. SDK 2.30+ unblocks both simultaneously.

## [0.11.0] — 2026-04-21

### Added

- **`precision="sr"` — stochastic rounding at the PSUM→SBUF boundary** —
  adds a fourth precision mode to ``einsum()`` and ``plan_contraction()``.

  On Trainium, ``matmul_kernel`` and ``batched_matmul_kernel`` replace the
  ``nisa.tensor_copy(psum → c_sbuf)`` downcast with
  ``nisa.activation(..., round_mode="stochastic")``. This rounds each FP32
  PSUM element to its upper or lower BF16 neighbor with probability
  proportional to the fractional distance — mean rounding error is zero
  (Connolly–Higham–Mary, *SIAM J. Sci. Comput.*, 2021), replacing Wilkinson's
  worst-case n·u bias with a √n·u bound.

  On CPU (``HAS_NKI=False``), ``_stochastic_round_cpu`` applies the same
  rounding to the ``torch.matmul`` output, enabling ``precision="sr"`` to be
  tested in CPU-only CI. This mirrors the ``_mock_allreduce`` pattern from
  v0.10.0: a named, documented stand-in that makes the hardware path testable
  without hardware.

  ``"sr"`` applies only to the ``matmul`` strategy. For ``"bmm"``,
  ``"torch"``, and ``"path"`` strategies, ``precision="sr"`` is accepted
  without error and uses fast (round-nearest) rounding.

  **Correctness note**: for BF16 dot products with K ≥ ~300, stochastic
  rounding is a *correctness requirement* for convergent Krylov iteration
  (without SR, BF16 Krylov stagnates). Use ``precision="sr"`` when K is
  large or accumulation bias matters.

### Notes

Mixed output-parallel + reduce-parallel operands in the same einsum remain
deferred to v0.12.0.

## [0.10.0] — 2026-04-21

### Added

- **Reduce-parallel sharding on CPU** — the ``NotImplementedError`` gate for
  contracted-dimension sharding is now conditional on ``HAS_NKI``. On CPU
  (``HAS_NKI=False``) reduce-parallel contractions dispatch through a mock
  all-reduce (torch-based sum of per-shard partial results), making the full
  sharding API testable in CI without hardware.

  On Trainium (``HAS_NKI=True``) the gate is unchanged — ``nki.collectives``
  is required for production reduce-parallel, arriving in Neuron SDK 2.30+.

- **Co-partitioning of plain-tensor operands** — when a reduce-parallel
  ``ShardedTensor`` operand partitions along a contracted index, ``einsum()``
  automatically slices any non-sharded operand that shares that index along
  the matching dimension per shard. No API change required; callers can mix
  sharded and dense operands in the same reduce-parallel einsum.

- **``shard_boundaries(st)``** in ``trntensor.parallel`` — returns the
  ``(start, end)`` slice pairs for each shard. Used internally by the dispatch
  layer; also useful for callers that need to manually co-partition ancillary
  data structures.

- **``_mock_allreduce(partials)``** in ``trntensor.parallel`` — sums a list of
  same-shape tensors element-wise. Named and documented to clearly identify the
  operation that ``nki.collectives.allreduce`` will replace on hardware.

### Notes

Mixed output-parallel + reduce-parallel operands in the same einsum are not
yet supported and raise ``NotImplementedError``. This covers an edge case not
present in the DF-MP2 patterns; deferred to v0.11.0.

## [0.9.0] — 2026-04-18

### Added

- **`trntensor.parallel` module** — Phase 4 foundation for output-parallel tensor
  contractions. Introduces ``ShardedTensor``, ``scatter``, and ``gather``:
  - ``ShardedTensor``: wraps a list of dense chunk tensors split along a single
    ``partition_dim``. Carries the partition spec so ``einsum()`` can detect it
    and classify the sharding mode.
  - ``scatter(tensor, dim, n_shards)``: splits a dense tensor evenly along
    ``dim``; last shard absorbs any remainder.
  - ``gather(st)``: reconstructs the dense tensor via ``torch.cat``.

- **Output-parallel sharding in ``einsum()``**: when any operand is a
  ``ShardedTensor``, ``einsum()`` checks whether the partition dimension maps to
  an output index (output-parallel). If so, each shard's contraction is dispatched
  independently (no cross-shard communication) and the result is returned as a
  ``ShardedTensor``. Works for all subscript patterns handled by the existing
  planner (matmul, bmm, torch, path) and propagates through multi-operand chains.

- **Reduce-parallel gate**: if the partition dimension is a contraction index
  (not in the output), ``einsum()`` raises ``NotImplementedError`` with a clear
  explanation: reduce-parallel sharding requires ``nki.collectives`` all-reduce,
  arriving in Neuron SDK 2.30+. The gate prevents silent wrong results.

### Notes

``nki.collectives`` (all-reduce, all-gather) is not available in NKI 0.3.0
Stable. Reduce-parallel sharding (contracted-dimension sharding) is gated until
SDK 2.30+. This sprint covers the output-parallel subset, which handles the
common row-parallel and column-parallel matmul patterns used in DF-MP2
AO→MO transform with large basis sets (nbasis > 3000 across chips).

## [0.8.0] — 2026-04-18

### Fixed

- **`_execute_path` now routes binary steps through `einsum()`** — multi-operand
  contractions (3+ tensors via the greedy path planner) previously called
  `torch.einsum` directly for each binary step, bypassing the entire trntensor
  dispatch layer (plan cache, FLOP thresholds, NKI routing). The fix is a
  one-line change: each step now calls `einsum(sub, a, b, precision=plan.precision)`,
  so matmul-shaped steps dispatch to NKI when operand size clears the threshold.
  All correctness tests are unaffected; top-level `ContractionPlan.backend` for
  strategy `"path"` remains `"pytorch"` (accurate: no single NKI kernel spans
  an arbitrary path). (#18 follow-up)

## [0.7.0] — 2026-04-17

### Added

- **N-tiled `ao_to_mo_transform_kernel`** — lifts `nbasis ≤ 512` and
  `nvir ≤ 512` single-N-tile constraints. Step 1 now tiles the ν (basis)
  dimension in ``TILE_N=512``-wide chunks; step 2 tiles the virtual-orbital
  a dimension similarly. Enables full DF-MP2 AO→MO transform for large basis
  sets: cc-pVTZ on naphthalene (nbasis≈572), aug-cc-pVTZ virtual spaces
  (nvir > 512). Dispatcher pads both `nbasis` and `nvir` to ``TILE_N``
  multiples (zero-fill); result sliced back to `(nocc, nvir, naux)`.

- **Simulator tests for N-tiled shapes** (`tests/test_nki_sim.py`,
  `TestAoToMoTransformSimulator`) — three new cases: `test_n_tiled_nbasis`
  (nbasis=640 > TILE_N), `test_n_tiled_nvir` (nvir=600 > TILE_N),
  `test_both_n_tiled` (nbasis=640, nvir=600). Each verifies against the
  CPU einsum reference.

### Changed

- **`ao_to_mo_transform` gates removed** — `NotImplementedError` for
  `nbasis > 512` and `nvir > 512` removed; both constraints are now handled
  by the tiled kernel + dispatcher. The `nocc > 128` gate remains (M-tiling
  not yet implemented).
- **`_nki_ao_to_mo_transform` padding** — `nbasis` now padded to
  `_round_up(nbasis, TILE_N)` instead of `_round_up(nbasis, TILE_K)`.
  TILE_N alignment is stricter (multiples of 512 vs 128), but implies
  TILE_K alignment (TILE_N = 4×TILE_K). `nvir` padding added.

## [0.6.0] — 2026-04-17

### Added

- **Tiled `mp2_energy_kernel`** — lifts `nvir ≤ 128` and `naux ≤ 128`
  single-tile constraints. The kernel now M-tiles over the virtual-orbital
  dimensions (TILE_M=128 per side) and K-tiles over the auxiliary index P
  (TILE_K=128), enabling arbitrary `nvir` and `naux` (dispatcher pads to
  tile multiples with zero-fill). Typical real-molecule sizes now supported:
  aug-cc-pVDZ (naux≈300+), cc-pVTZ (nvir≈200+). Both `nvir=256` + `naux=256`
  benchmarked at the same correctness tolerance as the single-tile path.

- **NVIR/NAUX padding in `_nki_mp2_energy` dispatch** — `_nki_mp2_energy`
  now pads `B` to `(nocc, nvir_pad, naux_pad)` and `eps_vir` to
  `(nvir_pad, 1)` before kernel entry. Zero-padding is safe: padded B
  entries are 0, so tile contributions to `T*(2T-T.T)/Δ` are 0 regardless
  of Δ.

- **Simulator tests for tiled shapes** (`tests/test_nki_sim.py`,
  `TestMp2EnergySimulator`) — three new cases: `test_naux_tiled`
  (nocc=4, nvir=32, naux=256), `test_nvir_tiled` (nocc=4, nvir=256,
  naux=64), `test_both_tiled` (nocc=4, nvir=256, naux=256). Each verifies
  against `_cpu_mp2_energy` via the NKI CPU simulator.

### Changed

- **`mp2_energy` gate removed** — the `NotImplementedError` for
  `nvir > 128 or naux > 128` is gone. The NKI path now handles all sizes
  ≥ TILE_M / TILE_K via the tiled kernel; smaller sizes (padded up) are
  handled identically to the original single-tile path.

## [0.5.0] — 2026-04-17

### Added

- **`multi_einsum` homogeneous batching** (#33 Phase 3) — when all contractions
  share the same subscript, operand shapes, and resolve to a 2D matmul, they are
  stacked and dispatched as a single `nki_batched_matmul` call instead of N
  independent ones. Amortizes the ~0.67 ms fixed XLA dispatch overhead across the
  batch (nocc²=4096 pair contractions: 2.7 s → 0.67 ms dispatch overhead). Falls
  through to the per-contraction loop for heterogeneous sets.

- **`_MIN_NKI_FLOPS_PINNED` dispatch threshold** (#33) — second FLOP gate for
  when operands are already XLA-resident (`to_xla` pre-applied). Profiler data
  (trn1.2xlarge, `scripts/run_neuron_profile.sh --dispatch-timing`) shows the
  NKI crossover drops from ~2 GFLOPs to ~900 MFLOPs when host→device transfer
  is eliminated. Default: 1 GFLOPs. Override with `TRNTENSOR_MIN_NKI_FLOPS_PINNED`.
  Dispatch automatically detects XLA-resident inputs and uses the lower threshold.

- **`docs/design/matmul_dispatch_findings.md`** — full per-step timing analysis
  documenting XLA dispatch latency (~0.67 ms fixed overhead), transfer costs, and
  the unpinned vs pre-pinned crossover. Raw JSON timing data included.

- **`docs/design/matmul_kernel_profile.md`** — Tensor Engine utilization profile
  for `matmul_kernel` on trn1.2xlarge. Captures TE/VE split, DMA transfer
  breakdown, and PSUM accumulation efficiency at medium (1024²) and large (2048²)
  matrix sizes.

- **`infra/terraform/terraform.tfvars`** — VPC/subnet/region values for
  `trntensor-ci-trn1` (us-east-1, vpc-cd49bfb0, subnet-1bd24d15).

### Changed

- **`infra/terraform/main.tf`** — default `aws_region` corrected from `us-west-2`
  to `us-east-1` to match the existing VPC/subnet. Running `terraform apply`
  without overrides now selects the correct region.

## [0.4.0] — 2026-04-16

### Added

- **`precision=` kwarg for `einsum` and `plan_contraction`** (#28 partial) —
  three modes: `"fast"` (default, no change), `"kahan"` (promotes all operands
  to `float64` before contracting via `torch.einsum`, casts result back to
  original dtype — gives ~15.9 significant digits; bypasses NKI dispatch for
  CPU compatibility), `"dd"` (raises `NotImplementedError` pending trnblas Phase
  2 double-double GEMM). `precision` is included in the plan cache key so
  `"fast"` and `"kahan"` plans cache independently. Useful for DF-MP2 / CCSD
  energy convergence where fp32's ~7.2 digits are insufficient.

- **`dtype` mixed-precision override for `einsum`** (#22) — accepts string
  aliases (`"bf16"`, `"fp16"`, `"f32"`) or `torch.dtype` instances. When set,
  all operands are cast to the requested dtype before contracting and the
  result is returned in that dtype. Matches Neuron SDK autocast recommendations;
  use `dtype="bf16"` to route fp32 models through the NKI bf16 matmul path
  without changing the model's weight dtype.

- **Neuron profiler script** (`scripts/run_neuron_profile.sh`) — captures a
  Neuron Profiler 2.0 trace of `matmul_kernel` or `batched_matmul_kernel` via
  SSM on the `trntensor-ci-trn1` instance. Supports `--probe` (API discovery),
  `--kernel matmul|bmm`, and `--shape small|medium|large`. Adapted from the
  trnblas profiler pattern (double-base64 encoding, auto start/stop).

- **Dispatch autotune script** (`scripts/autotune_dispatch.py`) (#33 partial)
  — sweeps `(M, K, N)` shapes and measures NKI vs PyTorch wall-clock time to
  find the empirical FLOP crossover on the current hardware. Reports the
  recommended `TRNTENSOR_MIN_NKI_FLOPS` value; `--write-cache` persists the
  result to `/var/tmp/trntensor-autotune/threshold.json`. The dispatch layer
  reads this cache at startup (env var `TRNTENSOR_AUTOTUNE_CACHE` overrides
  path; explicit `TRNTENSOR_MIN_NKI_FLOPS` always wins).

## [0.3.0] — 2026-04-16

### Added

- **Greedy contraction-path search for 3+ operand `einsum`** (#18) —
  `plan_contraction()` now returns `strategy="path"` for three or more
  operands. `_greedy_path_search` selects the cheapest binary contraction
  order by minimizing per-step FLOP cost; the resulting
  `ContractionPlan.contraction_path` drives `_execute_path`, which
  routes each binary step through the full backend selection stack so
  large sub-contractions still dispatch to NKI.

- **`multi_einsum` shared-operand XLA residency** (#19) — When NKI
  dispatch is active, `multi_einsum` detects operand tensors that
  appear in more than one contraction (by object identity) and
  pre-pins them to the XLA device once before executing the loop.
  Eliminates redundant host↔device transfers for workloads like DF-MP2
  where the three-center integral tensor `B` feeds many pair
  contractions. Falls back to the existing per-contraction loop on CPU.

- **Subscript and shape validation with descriptive errors** (#26) —
  `plan_contraction()` now validates subscripts up-front and raises
  `ValueError` with precise messages: wrong operand count, rank
  mismatch, and inconsistent index sizes are all caught before any
  torch operation runs. Eliminates cryptic downstream errors from
  `torch.einsum`.

- **PEP 561 `py.typed` marker** (#25) — `trntensor` now ships a
  `py.typed` file so type checkers (mypy, pyright, etc.) recognise
  the package as typed and apply inline annotations.

- **`alpha`/`beta` scaling for `einsum`** (#20) — matches cuTENSOR's
  GEMM-style interface:
  `einsum(subscripts, A, B, alpha=α, beta=β, out=C)` returns
  `α * contract(A, B) + β * C`. Defaults (`alpha=1, beta=0, out=None`)
  preserve existing behaviour exactly. Useful for accumulation patterns
  and in-place gradient updates without an extra allocation.

- **Contraction plan cache** (#29 partial) — `plan_contraction()` caches
  results by `(subscripts, operand shapes)`. Repeated calls with the same
  subscript and shapes skip replanning entirely. `clear_plan_cache()` and
  `plan_cache_info()` are exported from the top-level API.

- **Tensor Train (TT) decomposition** (#23) — `tt_decompose(tensor, max_rank)`
  decomposes a d-dimensional tensor into a chain of 3-tensor cores via TT-SVD
  (Oseledets 2011), bond dimension capped at `max_rank`. `tt_reconstruct(cores)`
  contracts the chain back. Useful for DMRG-style high-dimensional compression.

- **Non-negative CP and warm-start CP** (#24) — `cp_decompose` gains two new
  keyword arguments: `nonneg=True` switches ALS to multiplicative updates
  to enforce non-negative factors; `factors=` accepts a list of pre-computed
  factor matrices to warm-start from, skipping random initialization.
  Both options compose.

## [0.2.0] — 2026-04-15

### Added

- **`trntensor.to_xla(tensor)` / `trntensor.from_xla(tensor)`** — explicit
  operand residency on the Trainium XLA device. Pre-pinning operands
  lets repeated trntensor calls skip per-dispatch host↔device
  transfer, which otherwise dominates at current kernel sizes. The
  full DF-MP2 pipeline (`ao_to_mo_transform` → `mp2_energy`) with all
  operands pre-pinned pays transfer cost once instead of once per
  call. The dispatch layer's `_to_xla` helper takes a fast path when
  every operand is already on XLA, returning the result on XLA — the
  caller decides when to pull back via `from_xla`. Closes #34.
- **`trntensor.ao_to_mo_transform(eri, C_occ, C_vir)`** — fused 4-index
  AO→MO integral transform with K-tiling over the basis index (#37).
  One NKI program computes `B[i,a,P] = Σ_{μν} C_occ[μ,i] · C_vir[ν,a] ·
  eri[μ,ν,P]`. Tiles over μ (step 1) and ν (step 2) in TILE_K=128
  chunks so `nbasis` up to 512 is supported; dispatch pads to the
  nearest TILE_K multiple. Shape constraints: `nbasis ≤ 512`,
  `nocc ≤ 128`, `nvir ≤ 512`. Composes with `mp2_energy` for the full
  DF-MP2 pipeline from AO integrals to correlation energy. Validated
  on trn1 (hardware) and via the CPU simulator CI job.
- **NKI CPU simulator dispatch** via `TRNTENSOR_USE_SIMULATOR=1`.
  Routes kernels through `nki.simulate(kernel)(numpy_args)` on CPU,
  bypassing `torch_xla` + NEFF compile. Iteration loop drops from
  ~5 min per SSM round-trip to seconds. Correctness-only — MLIR
  verifier errors remain hardware-only.
- **`nki-simulator` CI job on `ubuntu-latest`** — runs the
  `nki_simulator`-marked suite against `nki>=0.3.0` from the AWS pip
  index on every push + PR. Zero AWS cost for the correctness gate.
- `tests/test_nki_sim.py` — simulator-backed correctness suite,
  marker `nki_simulator`. Covers matmul, batched matmul,
  `ao_to_mo_transform` (including K-tiled nbasis=256 and non-aligned
  nbasis=200), and `mp2_energy`.
- `scripts/run_simulator_tests.sh` — SSM runner for the simulator
  suite on the trn1 DLAMI.
- `docs/developing_kernels.md` — NKI kernel development guide with
  trntensor-specific env vars and file locations.

### Changed

- **Migrated to NKI 0.3.0 / Neuron SDK 2.29.** Canonical `nki.*`
  namespace; the legacy `neuronxcc.nki.*` shim is no longer used.
  Kernels updated for the NKI 0.3.0 breaking-change surface:
  `nisa.nc_matmul(dst=, stationary=, moving=, accumulate=True)` (all
  kwargs); `nl.copy(psum)` returns a view — use `nl.ndarray` +
  `nisa.tensor_copy` instead; tensor-tensor `nl.divide` dropped — use
  `multiply × reciprocal`.
- Dev workflow migrated to [uv](https://github.com/astral-sh/uv).
  `uv sync --extra dev` replaces `pip install -e ".[dev]"`; CI uses
  `astral-sh/setup-uv@v6` and `uv run pytest` / `uvx ruff`. `uv.lock`
  is committed for reproducible installs.
- Removed the `[neuron]` optional-dependencies extra. `nki` is
  installed from the AWS Neuron pip index in CI or provided by the
  Deep Learning AMI's pre-built venv on hardware.
- `CONTRIBUTING.md` updated to reflect the uv-based setup.

### Fixed

- `mp2_energy_kernel` 1D-load ambiguity on pre-pinned XLA ε inputs
  (#38). Reshape `eps_occ` / `eps_vir` to 2D `(N, 1)` at the dispatch
  boundary; partition-dim inference is unambiguous regardless of
  residency state.
- `mp2_energy_kernel` 0-D SBUF rejection (`SBUF tensors must have at
  least 2 dimensions`). Per-`(i,j)` reduction now uses a persistent
  `(1, 1)` SBUF accumulator (`nl.zeros((1, 1), ...)`) instead of a
  direct `nl.sum` store.
- `_to_xla` fast-path now calls `xm.mark_step()` when operands are
  already on XLA, forcing pending lazy computations to materialize
  before the next kernel dispatch.

### Known limitation

- The full DF-MP2 pipeline (`ao_to_mo_transform` → `mp2_energy`) with
  every operand pre-pinned exposes an NKI compiler bug on trn1: the
  combined XLA lazy graph provokes `trn2-only shared memory`
  instructions that fail verification on trn1. Workaround: `from_xla`
  the intermediate `B` between the two calls. Tracked in #39 for
  upstream AWS escalation.

## [0.1.2] — 2026-04-12

### Changed

- `set_backend("nki")` now raises `RuntimeError` on non-Neuron hosts
  instead of silently accepting the backend and failing later. Matches
  the sibling-suite pattern.
- CI actions bumped to `actions/checkout@v6` + `actions/setup-python@v6`
  (Node.js 24), ahead of GitHub's June 2026 default switch.
- pyproject metadata normalized across the trnsci suite (author email,
  URLs, classifier list).
- Standalone `docs.yml` removed — docs are now served via `trnsci.dev`
  through the umbrella's combined build. `notify-umbrella.yml` pings
  the umbrella on docs changes.
- `infra/terraform/main.tf`: user-data clone URL corrected to
  `trnsci/trntensor`.

### Added

- `benchmarks/bench_einsum.py` — pytest-benchmark cases for einsum
  dispatch and decompositions. CPU baseline numbers populated in
  `docs/benchmarks.md`.
- `tests/test_nki.py` — backend-dispatch unit tests (CPU path).

## [0.1.1] — 2026-04-12

### Added

- mkdocs site with `index`, `installation`, `quickstart`, `api`, `architecture`, `aws_setup`
- `infra/terraform/` for on-hardware CI instance provisioning
- `scripts/run_neuron_tests.sh` and benchmark helpers
- GitHub Actions `ci.yml`, `docs.yml`, `publish.yml`
- `Issues` and `Documentation` URLs in pyproject.toml
- `tests/test_plan.py` — dedicated planner unit tests (parsing,
  strategy selection, FLOP estimates); extended CP / Tucker coverage
  (all-zero tensor, rank > min dim, unequal mode ranks)

### Changed

- Bumped `neuronxcc` floor from `>=2.15` to `>=2.24` to unify with the
  rest of the trnsci suite. `torch-neuronx` floor bumped to `>=2.9`.

## [0.1.0] — 2026-04-12

### Added

- Initial scaffold: einsum with contraction planning, CP / Tucker decompositions
- NKI dispatch with fused-contraction kernel stubs
- `examples/df_mp2_einsum.py` — DF-MP2 energy via einsum
