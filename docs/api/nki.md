# NKI dispatch

Backend selection between the PyTorch fallback and the NKI-accelerated
path on Trainium. Kernels live in `trntensor.nki._kernels`; the
public-facing dispatch is in `trntensor.nki.dispatch`.

## `set_backend(backend: str)`

- `"auto"` — the default. When `nki` is importable, each call is routed by
  measured thresholds (see [Environment variables](#environment-variables)
  and the tables below); on the native path matmul/bmm run on aten, and the
  fused chemistry kernels run where they beat it. Otherwise PyTorch
- `"pytorch"` — always use `torch.matmul` / `torch.einsum`
- `"nki"` — require NKI; raises `RuntimeError` on non-Neuron hosts

## `get_backend() -> str`

Return the current backend selection.

## `HAS_NKI: bool`

Module-level flag set at import time. `True` when `nki` (0.3.0+) is
importable, otherwise `False`.

## Dispatch path

On Neuron hardware, kernels run on one of two `torch-neuronx` lineages,
detected on the first kernel launch (never at import):

- **Native (TorchNeuron, Trn2/Trn3, Neuron SDK 2.33+)** — chosen when
  `torch_neuronx` exports `nki_op` and a `neuron` device is available
  (`torch_neuronx.device_count() > 0`). Each kernel is a PyTorch custom op
  on `device="neuron"`, also callable directly and inside
  `torch.compile(backend="neuron")`: `torch.ops.trntensor.matmul`,
  `batched_matmul`, `mp2_energy`, `ao_to_mo_transform`, `mp2_energy_row`,
  `ao_to_mo_trn2`, `ao_to_mo_trn2_packed`, `df_half_packed`,
  `df_exchange_syrk`, `df_coulomb_d`, `df_coulomb_j` (`matmul` / `batched_matmul` are
  deprecated there, see below). The ops are
  registered only once this path is selected, and this path never imports
  `torch_xla`.
- **XLA (Trn1)** — chosen when `torch_xla` is installed and the native
  path is unavailable. Kernels run on the `torch_xla` device, unchanged.
- Otherwise there is no device path: the host wrappers fall back to
  `torch` (or raise under `TRNTENSOR_REQUIRE_NKI=1`), as before.

`set_backend` is unaffected: it still chooses NKI vs PyTorch. Set the
suite-wide `TRNSCI_NKI_DISPATCH=native` or `xla` to force a path; forcing an
unavailable path raises `RuntimeError`.

The reduce-parallel all-reduce kernel (`trntensor.nki.collectives`) is
native-only already; it keeps its `torch_neuronx.wrap_nki` launch because
its `ReplicaGroup` argument is a compile-time constant that a custom-op
schema cannot carry.

Precision modes follow the same launch: `precision="sr"` passes
`use_sr=True` to the `matmul` / `batched_matmul` op (stochastic rounding
still needs `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` at process start and warns
once otherwise); `precision="kahan"` contracts in true fp64 on the host
(Trainium has no fp64 datapath) on both paths; `precision="dd"` still
raises on NKI hosts pending trnblas#22.

## `to_device(tensor) -> Tensor` / `from_device(tensor) -> Tensor`

Pin an operand on the Trainium device of the active dispatch path —
`neuron` on TorchNeuron, the XLA device on Trn1 — so repeated trntensor
calls skip the host↔device transfer that otherwise dominates dispatch
overhead.

```python
import trntensor

# One-time transfer onto the accelerator.
eri_d = trntensor.to_device(eri)
C_occ_d = trntensor.to_device(C_occ)
C_vir_d = trntensor.to_device(C_vir)
eps_occ_d = trntensor.to_device(eps_occ)
eps_vir_d = trntensor.to_device(eps_vir)

# Full DF-MP2 pipeline — B_d never leaves the device.
B_d = trntensor.ao_to_mo_transform(eri_d, C_occ_d, C_vir_d)
E_d = trntensor.mp2_energy(B_d, eps_occ_d, eps_vir_d)

# Pull the scalar back when we actually need it in Python.
E = trntensor.from_device(E_d)
```

- `to_device`: no-op when the tensor is already on a Neuron device.
  Raises `RuntimeError` on hosts without the NKI runtime.
- `from_device`: no-op when the tensor is already on CPU.

When every operand is already on the device, dispatch skips the per-call
transfer, uses the lower "pinned" FLOP threshold, and returns the result on
the device — the caller controls when to pull back.

## `to_xla(tensor) -> Tensor` / `from_xla(tensor) -> Tensor`

The original names, kept for compatibility. On Trn1 they behave exactly as
before. On TorchNeuron there is no XLA device, so `to_xla` places the
tensor on `neuron` instead (residency works the same way) and `from_xla`
moves it back to CPU.

### Migrating from `to_xla` / `from_xla`

Replace `to_xla` with `to_device` and `from_xla` with `from_device`. Same
behaviour on Trn1; on Trn2/Trn3 the new names say what they do. Checks such
as `t.device.type == "xla"` become `t.device.type in ("xla", "neuron")`.

`to_xla` / `from_xla` stay available. When the suite-wide
`TRNSCI_XLA_DEPRECATION=1` is set (off by default; trnsci turns it on at
TorchNeuron GA), each emits a one-time `DeprecationWarning` pointing at the
new names, and selecting the XLA dispatch path emits: "the PyTorch/XLA path
is deprecated on Trn2/Trn3; use TorchNeuron (native). Trn1/Inf2 remain
supported in maintenance mode."

## Environment variables

| Variable | Effect |
|---|---|
| `TRNTENSOR_REQUIRE_NKI=1` | Re-raise kernel exceptions instead of falling back to PyTorch. Useful in the validation loop to surface silent kernel breakage. |
| `TRNTENSOR_MIN_NKI_FLOPS=<int>` | XLA path: override the FLOP threshold (M·K·N) below which dispatch skips NKI and uses `torch.matmul` / `torch.bmm` directly. Default: `2_000_000_000`. Set to `0` to force NKI for kernel validation. |
| `TRNTENSOR_MIN_NKI_FLOPS_PINNED=<int>` | XLA path, operands already on the device. Default: `1_000_000_000`. |
| `TRNTENSOR_MIN_NKI_FLOPS_NATIVE=<int>` | Native (TorchNeuron) path, operands on CPU: M·K·N at which the work leaves the host for the `neuron` device. Below it, `torch.matmul` / `torch.bmm` run on the host. Above it, the operands are offloaded. The offload runs aten unless the pinned threshold below selects NKI. Default: `500_000_000` (measured on Trn2, see below). |
| `TRNTENSOR_MIN_NKI_FLOPS_NATIVE_PINNED=<int>\|never` | Native path: M·K·N at which the NKI kernel replaces aten on the device. Below it, device-resident operands run `torch.matmul` / `torch.bmm` (aten) on `neuron`. Default: `never`, because on Trn2 aten beat the NKI kernels at every measured size. |
| `TRNSCI_NKI_DISPATCH=auto\|native\|xla` | Suite-wide. Force the native or XLA dispatch path; `auto` (default) detects. Forcing an unavailable path raises. |
| `TRNSCI_XLA_DEPRECATION=1` | Suite-wide, off by default. Emit the one-time XLA-path and `to_xla` / `from_xla` deprecation warnings. |
| `NKI_ENABLE_TRACE_CACHE=0` | TorchNeuron setting (not trntensor's). Turns off the cross-process NKI kernel cache, which cannot version editable installs and may serve a stale kernel after a source edit. Set it with `pip install -e`; the test suite sets it by default. |
| `TRNTENSOR_MP2_ENERGY=auto\|rows\|aten\|legacy` | Native path: force the `mp2_energy` executor (`rows` = the Trn2 row kernel, `legacy` = the original kernel, deprecated there). |
| `TRNTENSOR_AO_TO_MO=auto\|trn2\|aten\|legacy` | Native path: force the `ao_to_mo_transform` executor (`trn2` = the Trn2 kernel, or `ao_to_mo_trn2_packed_kernel` for a packed 2-D `eri`; `legacy` = the original kernel, deprecated there; `aten` and `legacy` unpack a packed `eri` first). |
| `TRNTENSOR_DF_EXCHANGE=auto\|trn2\|aten` | Native path: force the `df_exchange` executor (`trn2` = `df_half_packed_kernel` + `df_exchange_syrk_kernel` on packed integrals; `aten` = row gather + einsum + one GEMM). `auto` uses the kernels for device-resident packed `B` from `nocc·nbasis²·naux` = 1e7 (#60). |
| `TRNTENSOR_DF_COULOMB=auto\|trn2\|aten` | Native path: force the `df_coulomb` executor (`trn2` = `df_coulomb_d_kernel` + `df_coulomb_j_kernel`; `aten` = two GEMVs). `auto` uses the kernels for device-resident packed `B` from `npair·naux` = 1e7 (#60). |
| `TRNTENSOR_HOST_ROUTE=auto\|host\|device` | Native path, `"auto"` backend, all operands on the host: where the work runs (#56). `auto` (default) uploads it once it reaches the measured thresholds below; `host` keeps it on the host; `device` always uploads. Covers `mp2_energy`, `ao_to_mo_transform`, `einsum` / `multi_einsum` (matmul, bmm, generic and multi-operand plans). Only float32 / bfloat16 / float16 operands are uploaded. `set_backend("pytorch")` and `set_backend("nki")` ignore it. |
| `TRNTENSOR_PINNED_OUTPUT=0` | Native path: download results with plain `.to("cpu")` instead of into memory from the caching host allocator (`torch.empty(..., pin_memory=True)`, results of 1 MiB and up; #56). |
| `TRNTENSOR_FORCE_BACKEND=pytorch\|nki\|auto` | Override the backend selection at runtime without calling `set_backend()`. Used by benchmarks to sweep both paths on the same machine. |
| `TRNTENSOR_USE_SIMULATOR=1` | Route kernel dispatch through `nki.simulate(kernel)(numpy_args)` on CPU instead of XLA → NEFF → hardware. Catches Python-trace-level errors without AWS round-trips. MLIR verifier errors remain hardware-only. |

### Native thresholds: measured on Trn2

These were measured on a trn2.3xlarge (Neuron SDK 2.33, NKI 0.7.0), using one
NeuronCore and two host threads, with `scripts/bench_native_einsum.py` and
`scripts/autotune_dispatch.py` (#41):

- **Pinned operands: NKI never beats aten.** On device-resident operands,
  aten `torch.matmul` / `torch.bmm` beat the trntensor NKI matmul and
  batched-matmul kernels at every size from 128³ to 4096³, in both fp32 and
  bf16. Examples:
  - 4096³ bf16: 1.38 ms with aten, 6.82 ms with NKI.
  - 1024³ fp32: 0.20 ms with aten, 0.55 ms with NKI.

  The eager launch floor is about 0.13 ms for an aten op and about 0.24 ms for
  an NKI custom op.
- **Host operands: offload to aten from about 0.5 G multiply-adds.**
  Offloading to aten (upload, run on device, download) beats offloading
  through the NKI kernel at every size. It also beats host torch:
  - fp32: from about 134 M multiply-adds (512³) when the host has two
    threads.
  - bf16: from about 0.5 to 1 G multiply-adds.

  The default of 500 M sits between these. A host with more threads moves the
  crossover up. Re-measure on your host with `scripts/autotune_dispatch.py`.

### Host operands: upload or stay (#56)

With every operand on the host, native `"auto"` uploads the work, runs it on
the device by the device-resident policy, and downloads the result into
memory from the caching host allocator, once the work reaches:

| entry point | work | offload from (2 host threads) |
|---|---|---|
| `mp2_energy` | `nocc²·nvir²·naux` | 1.5e8 |
| `ao_to_mo_transform`, full `eri` | `nocc·nbasis²·naux` | 2e8 |
| `ao_to_mo_transform`, packed `eri` | `nocc·nbasis²·naux` | 3e7 |
| `einsum`, generic or multi-operand plan | contraction multiply-adds (a path: the sum of its steps) | 5e8, and at least 0.5 multiply-adds per byte moved |
| `einsum` matmul / bmm | `M·K·N` | `TRNTENSOR_MIN_NKI_FLOPS_NATIVE` (5e8, above) |

- The first four scale with `torch.get_num_threads() / 2`, as trnblas#65
  does: the host side is compute-bound and the thresholds were measured with
  2 threads. The matmul threshold keeps its own override and is not scaled.
- Einsums with little reuse stay on the host at any size: under 0.5
  multiply-adds per byte moved, such as elementwise (0.08) and outer
  (0.25) products. They measured 1.4-2.9x faster uploaded, but only because
  2 host threads stream memory slower than the link. Host bandwidth grows
  with threads and the link's does not, and these patterns fill device
  memory.
- float64 and complex operands stay on the host, matmul included. The
  device computes in at most fp32.
- If the upload or the device call fails (for example, a full device), the
  call runs on the host. `TRNTENSOR_REQUIRE_NKI=1` raises instead.
- The first call of each shape on the device pays its compile. These tables
  are warm.

Measured on Trn2 (SDK 2.33, one logical core at LNC=2, fp32, warm medians,
host in to host out). Host threads were pinned to 2 CPUs on two physical
cores:

"Host" is the host route on two physical cores; the column after it is the
same on one physical core (both hyperthreads). "Device" is upload + the
device-resident `auto` route + download. **Bold** is what `auto` now picks.

`mp2_energy`, ms:

| nocc/nvir/naux | work | host | host, 1 core | device |
|---|---|---|---|---|
| 8/56/192 | 3.9e7 | **1.4** | 1.7 | 3.1 |
| 8/80/256 | 1.1e8 | **3.8** | 4.9 | 4.2 |
| 8/96/256 | 1.5e8 | 5.1 | 6.8 | **4.1** |
| 16/112/384 | 1.2e9 | 26.0 | 39.0 | **3.7** |
| 16/240/768 | 1.1e10 | 173 | 253 | **6.5** |
| 32/224/1024 | 5.3e10 | 646 | 1128 | **16.4** |
| 48/336/1152 | 3.0e11 | 3697 | 6457 | **47.4** |
| 64/448/1536 | 1.3e12 | 15849 | 28157 | **135** |

`ao_to_mo_transform`, ms (full `eri` / packed `eri`):

| nbasis/naux/nocc | work | host | host, 1 core | device |
|---|---|---|---|---|
| 64/192/8 | 6.3e6 | **1.0** / **1.3** | 1.1 / 1.7 | 5.7 / 6.7 |
| 128/384/16 | 1.0e8 | **4.7** / 6.7 | 7.4 / 11.0 | 7.5 / **5.3** |
| 128/512/16 | 1.3e8 | **6.2** / – | 10.1 / 21.7 | 8.9 / **4.1** |
| 160/480/20 | 2.5e8 | 9.4 / – | 16.3 / 32.9 | **7.1** / **5.1** |
| 256/512/8 | 2.7e8 | 16.0 / 54.2 | 26.5 / 71.0 | **13.6** / **13.0** |
| 256/1024/32 | 2.2e9 | 66.6 / 138 | 111 / 198 | **25.7** / **15.7** |
| 384/1152/48 | 8.2e9 | 211 / 397 | 355 / 571 | **62.2** / **37.9** |
| 512/1536/64 | 2.6e10 | 574 / 979 | 959 / 1473 | **137** / **79.9** |

`einsum` with host operands, ms:

| contraction | work | MACs per byte | host | host, 1 core | device |
|---|---|---|---|---|---|
| `mi,mnP->inP` 128/16/384 | 1.0e8 | 3.6 | **3.2** | 5.5 | 4.4 |
| `mi,mnP->inP` 256/32/768 | 1.6e9 | 7.1 | 30.5 | 52.4 | **19.7** |
| `mi,mnP->inP` 512/4/1536 | 1.6e9 | 1.0 | – | 252 | **123** |
| `iap,jbp->ijab` 16/112/384 | 1.2e9 | 67 | 7.5 | 13.3 | **3.8** |
| `iaP,PQ->iaQ` 64/448/1536 | 6.8e10 | 187 | 386 | 707 | **31.9** |
| `abc,cd->abd` 256 | 4.3e9 | 32 | 46.9 | 62.7 | **11.7** |
| `mi,na,mnP->iaP` 256/32/224/768 (path) | 3.0e9 | 13.5 | – | 90.1 | **19.0** |
| `ij,ij->ij` 8192² | 6.7e7 | 0.08 | – | **85.8** | 60.3 |
| `ab,cd->abcd` 160 | 6.6e8 | 0.25 | – | **580** | 199 |
| `ij,jk->ik` 640³ | 2.6e8 | 53 | **2.3** | 3.7 | 2.6 |
| `ij,jk->ik` 768³ | 4.5e8 | 66 | **3.4** | 5.6 | 2.7 |
| `ij,jk->ik` 1024³ | 1.1e9 | 85 | 6.5 | 12.1 | **3.2** |

- Results match the host route to 5e-7 relative (fp32 summation order).
- On one physical core the host is about 1.7x slower, so the device wins
  earlier: from 3e7 for the full `eri`, and below 1.3e8 for matmul. The
  thresholds were set on the faster host, so they are on the host's side.
- The device floor varied between runs, 1.2 to 2.6 ms for a small matmul.

Kernel status on the native path after #51:

- **matmul / bmm: retired in favour of aten.** A Trn2 rewrite used both
  physical cores through the `program_id` split, Tensor Engine transposes of
  A and B-tile reuse across four stationaries. It was 1.2–4.3× faster than
  the original kernel and still slower than aten at every size: 4096³ fp32
  took 4.10 ms against 3.80 ms, and 4096³ bf16 1.61 ms against 1.43 ms.
  It was not kept. `set_backend("nki")` still runs the original kernels on
  the native path, with a one-time `FutureWarning`. A later release will
  route them to aten.
- **AO→MO: kept, rewritten for Trn2** (`ao_to_mo_trn2_kernel`). It is
  1.3–1.6× faster than the two aten einsums at nbasis 256–768, and native
  `"auto"` uses it there for device-resident `eri`. Below nbasis ≈ 256
  `"auto"` keeps aten.
- **MP2 energy: kept, rewritten for Trn2** (`mp2_energy_row_kernel`). It is
  1.25–1.9× faster than aten once each row launch carries at least 3 GFLOP,
  and native `"auto"` uses it there.

The original AO→MO and MP2 kernels remain the XLA-path (Trn1) kernels. See
[Quantum-chemistry primitives](quantum.md) for the tables.

## Kernels (internal)

- `matmul_kernel(a, b)` — 2-index matmul with stationary-A tile reuse.
  Tile constants: `TILE_M=128`, `TILE_K=128`, `TILE_N=512`.
- `batched_matmul_kernel(a, b)` — per-batch-slice matmul. Batch dim
  iterated via `nl.affine_range`; each slice reuses the stationary-A
  tile layout.
- `df_half_packed_kernel(eri_v, img, C_occ, nub, tp, tail)` and
  `df_exchange_syrk_kernel(xt, kin, banks)` — the fused DF exchange build
  behind `trntensor.df_exchange` (#60): packed-row gather and half transform
  into `Xt[P, i, ν]` (aux on partitions), then the PSUM SYRK over aux tiles
  with upper-triangle strips mirrored on the Tensor Engine. Tilings from
  `df_half_tiling`, `df_group_chunks`, `df_groups`, `df_syrk_tiles` and
  `df_exchange_blocks`.
- `df_coulomb_d_kernel(B, pw)` and `df_coulomb_j_kernel(B, d_b)` — the DF
  Coulomb build behind `trntensor.df_coulomb` (#60): `d = Bᵀ pw` and
  `J = B d`, each streaming packed `B` once in 128-row tiles on the Vector
  Engine.

The DF-MP2 pair pattern `einsum("ap,bp->ab")` is planned as `matmul` with
`transB=True`: aten on the native path, `matmul_kernel` on the XLA path
above the FLOP threshold. The fused contraction + energy-denominator
kernel (#13) is `trntensor.mp2_energy`.

[m2]: https://github.com/trnsci/trntensor/milestone/2
