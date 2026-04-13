# Benchmarks

Performance results for trntensor — einsum dispatch, contraction planning, and CP / Tucker decompositions — comparing the PyTorch CPU fallback and NKI Trainium path.

## Status

Baseline PyTorch-fallback numbers run on every CI build. NKI numbers are pending on-hardware validation on trn1 / trn2 — fused-contraction kernels are scaffolded but not yet validated. See [AWS Setup](aws_setup.md) for the on-hardware CI flow.

## Reproducing locally

```bash
pytest benchmarks/ --benchmark-only
```

## Results table (placeholder)

| Op | Shape | PyTorch (CPU) | NKI (Trainium) | Speedup |
|---|---|---|---|---|
| einsum `ij,jk->ik` | 1024 shared | TBD | TBD | TBD |
| einsum `iap,ibp->iab` (batched) | 16×32×64, 16×48×64 | TBD | TBD | TBD |
| einsum `mhkn,ukvh->munv` (4-index) | cuTENSOR default | TBD | TBD | TBD |
| cp_decompose | 16³ rank 8 | TBD | TBD | TBD |
| tucker_decompose | 16³ ranks (4,4,4) | TBD | TBD | TBD |

Numbers will be populated once the NKI fused-contraction kernels validate on trn1 / trn2.
