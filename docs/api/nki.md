# NKI dispatch

Backend selection between the PyTorch fallback and the NKI-accelerated
path on Trainium. Kernels live in `trntensor.nki._kernels`; the
public-facing dispatch is in `trntensor.nki.dispatch`.

## `set_backend(backend: str)`

- `"auto"` — use NKI if `nki` is importable, else PyTorch
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
  `batched_matmul`, `mp2_energy`, `ao_to_mo_transform`. The ops are
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
| `TRNTENSOR_MIN_NKI_FLOPS_NATIVE=<int>` | Native (TorchNeuron) path, operands on CPU. Default: `1_000_000_000` (provisional: native eager dispatch is ~0.2 ms per launch; re-measure with `scripts/autotune_dispatch.py`). |
| `TRNTENSOR_MIN_NKI_FLOPS_NATIVE_PINNED=<int>` | Native path, operands already on `neuron`. Default: `128_000_000` (provisional). |
| `TRNSCI_NKI_DISPATCH=auto\|native\|xla` | Suite-wide. Force the native or XLA dispatch path; `auto` (default) detects. Forcing an unavailable path raises. |
| `TRNSCI_XLA_DEPRECATION=1` | Suite-wide, off by default. Emit the one-time XLA-path and `to_xla` / `from_xla` deprecation warnings. |
| `NKI_ENABLE_TRACE_CACHE=0` | TorchNeuron setting (not trntensor's). Turns off the cross-process NKI kernel cache, which cannot version editable installs and may serve a stale kernel after a source edit. Set it with `pip install -e`; the test suite sets it by default. |
| `TRNTENSOR_FORCE_BACKEND=pytorch\|nki\|auto` | Override the backend selection at runtime without calling `set_backend()`. Used by benchmarks to sweep both paths on the same machine. |
| `TRNTENSOR_USE_SIMULATOR=1` | Route kernel dispatch through `nki.simulate(kernel)(numpy_args)` on CPU instead of XLA → NEFF → hardware. Catches Python-trace-level errors without AWS round-trips. MLIR verifier errors remain hardware-only. |

## Kernels (internal)

- `matmul_kernel(a, b)` — 2-index matmul with stationary-A tile reuse.
  Tile constants: `TILE_M=128`, `TILE_K=128`, `TILE_N=512`.
- `batched_matmul_kernel(a, b)` — per-batch-slice matmul. Batch dim
  iterated via `nl.affine_range`; each slice reuses the stationary-A
  tile layout.

The DF-MP2 pair pattern `einsum("ap,bp->ab")` is served by
`matmul_kernel` through the planner's `transB=True` route — no
dedicated kernel needed. Fused energy-denominator kernel is tracked
in [#13][i13].

[i13]: https://github.com/trnsci/trntensor/issues/13

[m2]: https://github.com/trnsci/trntensor/milestone/2
