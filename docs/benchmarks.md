# Benchmarks

Performance results for trntensor — einsum dispatch, contraction planning, and CP / Tucker decompositions — comparing the PyTorch CPU fallback against the NKI Trainium path.

## Reproducing

```bash
# CPU baseline (local)
pytest benchmarks/bench_einsum.py --benchmark-only

# On-hardware (via SSM orchestration)
AWS_PROFILE=aws ./scripts/run_benchmarks.sh trn1
```

## Results

_CPU: Darwin arm64, PyTorch CPU. NKI: trn1.2xlarge (1 NeuronCore), neuronxcc 2.24, end-to-end `einsum()` call via `_nki_matmul` / `_nki_batched_matmul`._

| Op | Shape | PyTorch (CPU) | NKI (trn1) |
|---|---|---:|---:|
| `einsum ap,bp->ab` (DF-MP2 pair) | 48×128 × 48×128 | **4.5 µs** | 1068 µs |
| `einsum mi,mnP->inP` (4-index) | 32×8, 32×32×64 | **10.5 µs** | 46.7 µs |
| `einsum ij,jk->ik` | 512×512 × 512×512 | **104 µs** | 1643 µs |
| `tucker_decompose` | 16³ ranks (4,4,4) | **177 µs** | 926 µs |
| `einsum bij,bjk->bik` | 16×256×256 | **337 µs** | 4309 µs |
| `cp_decompose` | 16³ rank 8 (20 iters) | **4.6 ms** | 21.7 ms |

## Interpretation

The CPU path currently beats the NKI path across the board on these
workload sizes. This is **not** because the on-chip NKI kernels are
slow — it's because the per-call wrapper (XLA device transfer, NEFF
cache lookup, kernel launch) dominates at these sizes. `einsum()` is
called many times with small-to-medium tensors; each call incurs
roughly a millisecond of overhead before the kernel runs.

The kernels themselves are correct (validated by 9 hardware tests in
`tests/test_nki_kernels.py`). Closing the gap means reducing per-call
overhead — either by keeping tensors on the XLA device across multiple
calls, batching dispatches, or switching to a lower-overhead launch
path for small kernels.

Tracked as follow-up work in [#33][i33]. Until that lands, the
practical recommendation for trntensor users is:

- Use the **PyTorch fallback** (`set_backend("pytorch")`) for iterative
  Python loops of small contractions.
- Use the **NKI backend** (`set_backend("auto")` on Trainium) for
  large GEMM-dominant payloads where the per-call overhead is
  amortized.

[i33]: https://github.com/trnsci/trntensor/issues/33
