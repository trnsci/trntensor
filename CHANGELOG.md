# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
