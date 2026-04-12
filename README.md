# trntensor

[![CI](https://github.com/scttfrdmn/trntensor/actions/workflows/ci.yml/badge.svg)](https://github.com/scttfrdmn/trntensor/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trntensor)](https://pypi.org/project/trntensor/)
[![Python](https://img.shields.io/pypi/pyversions/trntensor)](https://pypi.org/project/trntensor/)
[![License](https://img.shields.io/github/license/scttfrdmn/trntensor)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://trnsci.github.io/trntensor/)

Tensor contractions for AWS Trainium via NKI.

Einstein summation with contraction planning, CP and Tucker decompositions. Expresses scientific tensor workloads naturally instead of decomposing to GEMM. Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Install

```bash
pip install trntensor
```

## Usage

```python
import torch
import trntensor

# Einsum — drop-in for torch.einsum with contraction planning
C = trntensor.einsum("ij,jk->ik", A, B)           # matmul
T = trntensor.einsum("ap,bp->ab", B_i, B_j)       # DF-MP2 contraction
X = trntensor.einsum("mi,mnP->inP", C_occ, eri)   # AO→MO transform

# Contraction planning
plan = trntensor.plan_contraction("ij,jk->ik", A, B)
flops = trntensor.estimate_flops("ij,jk->ik", A, B)

# CP decomposition (tensor hypercontraction)
factors, weights = trntensor.cp_decompose(tensor, rank=10)
reconstructed = trntensor.cp_reconstruct(factors, weights)

# Tucker decomposition (HOSVD)
core, factors = trntensor.tucker_decompose(tensor, ranks=(5, 5, 5))
```

## Operations

| Category | Operation | Description |
|----------|-----------|-------------|
| Contraction | `einsum` | General tensor contraction |
| Contraction | `multi_einsum` | Multiple contractions (fusion-ready) |
| Planning | `plan_contraction` | Analyze and select strategy |
| Planning | `estimate_flops` | FLOPs for a contraction |
| Decomposition | `cp_decompose` | CP/PARAFAC via ALS |
| Decomposition | `tucker_decompose` | Tucker via HOSVD |

## Status

- [x] Einsum with matmul/bmm/torch dispatch
- [x] Contraction planner
- [x] CP decomposition (ALS)
- [x] Tucker decomposition (HOSVD)
- [x] DF-MP2 einsum example
- [ ] NKI fused contraction kernels
- [ ] Multi-contraction fusion
- [ ] Optimal contraction ordering (like opt_einsum)

## Related Projects

| Project | What |
|---------|------|
| [trnfft](https://github.com/scttfrdmn/trnfft) | FFT + complex ops |
| [trnblas](https://github.com/scttfrdmn/trnblas) | BLAS operations |
| [trnsolver](https://github.com/scttfrdmn/trnsolver) | Linear solvers |
| [trnrand](https://github.com/scttfrdmn/trnrand) | Random number generation |
| [trnsparse](https://github.com/scttfrdmn/trnsparse) | Sparse operations |

## License

Apache 2.0 — Copyright 2026 Scott Friedman
