# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`dtype` mixed-precision override for `einsum`** (#22) — accepts string
  aliases (`"bf16"`, `"fp16"`, `"f32"`) or `torch.dtype` instances. When set,
  all operands are cast to the requested dtype before contracting and the
  result is returned in that dtype. Matches Neuron SDK autocast recommendations;
  use `dtype="bf16"` to route fp32 models through the NKI bf16 matmul path
  without changing the model's weight dtype.

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
