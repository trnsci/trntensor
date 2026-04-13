# Benchmarks

Performance results for trntensor — einsum dispatch, contraction planning, and CP / Tucker decompositions — comparing the PyTorch fallback and the NKI Trainium path on the same machine.

## Reproducing

```bash
# Local CPU
pytest benchmarks/bench_einsum.py --benchmark-only

# On-hardware (via SSM orchestration)
AWS_PROFILE=aws ./scripts/run_benchmarks.sh trn1

# Force the PyTorch path on hardware (apples-to-apples baseline)
TRNTENSOR_FORCE_BACKEND=pytorch pytest benchmarks/ --benchmark-only
```

## Results (trn1.2xlarge, neuronxcc 2.24)

Both columns ran on the same trn1.2xlarge instance — CPU (Intel Xeon 8375C) vs NKI (1 NeuronCore).

| Op | Shape | FLOPs | PyTorch (trn1 CPU) | NKI (trn1) | NKI / CPU |
|---|---|---:|---:|---:|---:|
| `einsum ap,bp->ab` | 48×128 × 48×128 | 295 K | **19.6 µs** | 1047 µs | 53.4× |
| `einsum mi,mnP->inP` (4-index) | 32×8, 32×32×64 | 524 K | 35.4 µs | 35.1 µs | 0.99× |
| `einsum ij,jk->ik` | 512³ | 134 M | **481 µs** | 1452 µs | 3.0× |
| `tucker_decompose` | 16³ ranks (4,4,4) | — | 875 µs | 859 µs | 0.98× |
| `einsum bij,bjk->bik` | 16×256³ | 268 M | **953 µs** | 2162 µs | 2.3× |
| `einsum ij,jk->ik` | 1024³ | 1.07 G | **3402 µs** | 4022 µs | 1.2× |
| `cp_decompose` | 16³ rank 8 (20 iters) | — | 21.3 ms | 21.9 ms | 1.03× |
| `einsum ij,jk->ik` | 2048³ | 8.6 G | 27.4 ms | **16.9 ms** | **0.62×** |
| `einsum bij,bjk->bik` | 32×1024³ | 34.4 G | **126.3 ms** | 190.8 ms | 1.5× |

**NKI wins**: 2048×2048 matmul (1.6× faster than CPU). All other sizes still favor CPU on this hardware.

## Size-based dispatch threshold

Because per-call NKI dispatch currently carries ~1 ms of XLA launch overhead, `nki_matmul` and `nki_batched_matmul` short-circuit to the PyTorch path when the contraction is below `TRNTENSOR_MIN_NKI_FLOPS` (default **2 GFLOPs**, calibrated at ≈ half the smallest NKI-winning size). The `plan.backend` field reflects this: it reports `"nki"` only when the dispatch will actually invoke a kernel, and `"pytorch"` otherwise.

Overrides:

- `TRNTENSOR_MIN_NKI_FLOPS=0` — always attempt NKI (useful for kernel validation).
- `TRNTENSOR_FORCE_BACKEND=pytorch` — always use the PyTorch path (useful for CPU baselines).
- `trntensor.set_backend("pytorch")` — same, via API.

## Interpretation

The kernels themselves are **correct** (validated by 9 hardware tests in `tests/test_nki_kernels.py`) and the 2048×2048 case confirms NKI can beat CPU at large enough workloads. The gap at smaller sizes is dominated by per-call overhead — not by the Tensor Engine itself — and is tracked as follow-up work in [#33][i33].

Recommendation for users:

- Call trntensor normally. The dispatch layer makes the right choice for typical sizes.
- For GEMM-dominant workloads with single contraction sizes ≥ 2 GFLOPs, NKI will be invoked automatically.
- Until #33 lands, tight loops of small contractions (DF-MP2 pair-energy style) see no NKI benefit and are served by the PyTorch path.

[i33]: https://github.com/trnsci/trntensor/issues/33
