# Benchmarks

Performance results for trntensor — einsum dispatch, contraction planning, and CP / Tucker decompositions — comparing the PyTorch CPU fallback against the NKI Trainium path.

## Reproducing

```bash
# CPU baseline (local)
pytest benchmarks/bench_einsum.py --benchmark-only

# On-hardware (via SSM orchestration)
AWS_PROFILE=... ./scripts/run_benchmarks.sh trn1
```

## PyTorch CPU baseline

Runs on a developer laptop. Use these numbers to sanity-check relative costs of different contraction patterns — not as a ceiling for what Trainium can do.

_Platform: Darwin arm64, PyTorch CPU, median of ≥5 rounds._

| Op | Shape | Median |
|---|---|---|
| `einsum ap,bp->ab` (DF-MP2 pair) | 48×128 × 48×128 | 4.5 µs |
| `einsum mi,mnP->inP` (4-index) | 32×8, 32×32×64 | 10.5 µs |
| `einsum ij,jk->ik` | 512×512 × 512×512 | 104 µs |
| `tucker_decompose` | 16³ ranks (4,4,4) | 177 µs |
| `einsum bij,bjk->bik` | 16×256×256 | 337 µs |
| `cp_decompose` | 16³ rank 8 (20 iters) | 4.6 ms |

## NKI (Trainium) — pending

Fused-contraction kernels are scaffolded in `trntensor/nki/dispatch.py` but not yet implemented. Tracked in [#1](https://github.com/trnsci/trntensor/issues/1). The `scripts/run_benchmarks.sh` SSM orchestration is ready to populate the NKI column once kernels land.

| Op | NKI (Trainium) | Speedup vs CPU |
|---|---|---|
| all | pending | pending |

See [AWS Setup](aws_setup.md) for the on-hardware CI flow.
