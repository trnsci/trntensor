# NKI dispatch

Backend selection between the PyTorch fallback and the NKI-accelerated
path on Trainium. Kernels live in `trntensor.nki._kernels`; the
public-facing dispatch is in `trntensor.nki.dispatch`.

## `set_backend(backend: str)`

- `"auto"` — use NKI if `neuronxcc` is importable, else PyTorch
- `"pytorch"` — always use `torch.matmul` / `torch.einsum`
- `"nki"` — require NKI; raises `RuntimeError` on non-Neuron hosts

## `get_backend() -> str`

Return the current backend selection.

## `HAS_NKI: bool`

Module-level flag set at import time. `True` when `neuronxcc` is
available, otherwise `False`.

## Environment variables

- `TRNTENSOR_REQUIRE_NKI=1` — re-raise kernel exceptions instead of
  falling back to PyTorch. Useful in the validation loop to surface
  silent kernel breakage.

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
