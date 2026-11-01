# trntensor

[![CI](https://github.com/trnsci/trntensor/actions/workflows/ci.yml/badge.svg)](https://github.com/trnsci/trntensor/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/trnsci/trntensor/graph/badge.svg)](https://codecov.io/gh/trnsci/trntensor)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/trntensor)](https://pypi.org/project/trntensor/)
[![Python](https://img.shields.io/pypi/pyversions/trntensor)](https://pypi.org/project/trntensor/)
[![License](https://img.shields.io/github/license/trnsci/trntensor)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-trnsci.dev-blue)](https://trnsci.dev/trntensor/)

Tensor contractions for AWS Trainium via NKI.

Einstein summation with contraction planning, CP and Tucker decompositions. Expresses scientific tensor workloads naturally instead of decomposing to GEMM. Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Current phase

trntensor follows the [trnsci 5-phase roadmap](https://trnsci.dev/roadmap/). Active work is tracked in phase-labeled GitHub issues:

- **[Phase 1 — correctness](https://github.com/trnsci/trntensor/issues/27)** (active): matmul + batched-matmul NKI kernels in place; awaiting hardware validation + additional `@pytest.mark.neuron` coverage.
- **[Phase 2 — precision](https://github.com/trnsci/trntensor/issues/28)**: precision-aware contraction path selection (depends on [trnblas#22](https://github.com/trnsci/trnblas/issues/22) double-double GEMM).
- **[Phase 3 — perf](https://github.com/trnsci/trntensor/issues/29)**: opt_einsum-style path planner, plan cache reuse.
- **[Phase 4 — multi-chip](https://github.com/trnsci/trntensor/issues/30)**: sharded tensor contractions.
- **[Phase 5 — generation](https://github.com/trnsci/trntensor/issues/31)**: trn2 fused multi-contraction paths.

Suite-wide tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## Install

```bash
pip install trntensor

# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trntensor[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trntensor[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

Kernels dispatch on whichever Neuron PyTorch lineage is installed:

- **TorchNeuron (native)**: Trn2/Trn3, Neuron SDK 2.33+, Python ≥ 3.12,
  `torch` 2.14, `torch-neuronx` 2.14.*. Each kernel is a PyTorch custom op
  (`torch.ops.trntensor.*`) on the `neuron` device; no `torch_xla`.
- **PyTorch/XLA**: Trn1 (and any host on the `torch-neuronx` 2.9 lineage).
  Kernels dispatch through `torch_xla`, unchanged.

trntensor picks the native path when the native `torch_neuronx` and a
`neuron` device are present, and the XLA path when `torch_xla` is
installed; otherwise it runs on PyTorch. Set `TRNSCI_NKI_DISPATCH=native`
or `xla` (shared by all trnsci libraries) to force a path. Pin operands on
the device with `to_device` / `from_device`; the older `to_xla` /
`from_xla` still work on both lineages (see the
[migration note](docs/api/nki.md#migrating-from-to_xla-from_xla)). With an
editable install (`pip install -e`), set `NKI_ENABLE_TRACE_CACHE=0`: the NKI
kernel cache only versions installed wheels and can serve stale kernels.

### Known platform issue: NKI trace cache and kernel factories

`import torch_neuronx` turns on the NKI trace cache, and the cache key ignores
constants captured in a kernel's closure: two `@nki.jit` kernels built by one
factory, differing only in a captured value, share one trace, and the second
silently returns the first's results
([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454)).
trntensor's kernels do not rely on closure constants (or on module globals that
can differ between processes): every value that specialises a kernel is a
kernel argument, and `tests/test_trace_cache_safety.py` enforces this. If you
write your own NKI kernels with a factory, set `NKI_ENABLE_TRACE_CACHE=0`
before importing `torch_neuronx`.

## Usage

```python
import torch
import trntensor

# Einsum — drop-in for torch.einsum with contraction planning
C = trntensor.einsum("ij,jk->ik", A, B)  # matmul
T = trntensor.einsum("ap,bp->ab", B_i, B_j)  # DF-MP2 contraction
X = trntensor.einsum("mi,mnP->inP", C_occ, eri)  # AO→MO transform

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
- [x] NKI fused contraction kernels (`mp2_energy`, `ao_to_mo_transform`, `df_exchange`, `df_coulomb`)
- [x] Device operand residency (`to_device` / `from_device`; `to_xla` / `from_xla` kept)
- [x] Native TorchNeuron dispatch (Trn2/Trn3, `torch.ops.trntensor.*`) alongside PyTorch/XLA (Trn1)
- [x] NKI CPU simulator + `nki-simulator` CI gate
- [x] Optimal contraction ordering — greedy path search for 3+ operands
- [x] Multi-contraction shared-operand XLA pre-pinning
- [x] Contraction plan cache (`clear_plan_cache` / `plan_cache_info`)
- [x] Alpha/beta scaling for `einsum` (cuTENSOR GEMM-style)
- [x] Input validation with descriptive errors
- [x] Tensor Train (TT) decomposition (TT-SVD)
- [x] Non-negative CP + warm-start CP
- [x] PEP 561 `py.typed` marker
- [x] Mixed precision / `dtype` override for bf16/fp16
- [x] `precision=` kwarg (`"fast"` / `"kahan"` fp64 promotion / `"dd"` stub)

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


## Disclaimer

trnsci is an **independent open-source project**. It is not sponsored by, endorsed by, or affiliated with Amazon.com, Inc., Amazon Web Services, Inc., or Annapurna Labs Ltd.

"AWS", "Amazon", "Trainium", "Inferentia", "NeuronCore", "Neuron SDK", and related identifiers are trademarks of their respective owners and are used here solely for descriptive and interoperability purposes. Use does not imply endorsement, partnership, or any other relationship.

All work, opinions, analyses, benchmark results, architectural commentary, and editorial judgments in this repository and on [trnsci.dev](https://trnsci.dev) are those of the project's contributors. They do not represent the views, positions, or commitments of Amazon, AWS, or Annapurna Labs.

Feedback directed at the Neuron SDK or Trainium hardware is good-faith ecosystem commentary from independent users. It is not privileged information, is not pre-reviewed by AWS, and should not be read as authoritative about product roadmap, behavior, or quality.

For official AWS guidance, see [aws-neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/) and the [AWS Trainium product page](https://aws.amazon.com/ai/machine-learning/trainium/).
