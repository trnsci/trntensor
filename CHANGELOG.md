# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `mp2_energy_kernel` 1D-load ambiguity on pre-pinned XLA ε inputs
  (#38). Reshape `eps_occ` / `eps_vir` to 2D `(N, 1)` at the dispatch
  boundary; kernel `nl.load` signatures updated to match. Makes
  partition-dim inference unambiguous regardless of residency state.
  Also switched `mp2_energy_kernel`'s per-(i,j) reduction to trnblas's
  accumulator pattern (persistent `(1, 1)` SBUF tile) so the `nl.sum`
  result doesn't need a 0-D SBUF allocation.
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

### Added

- **`trntensor.to_xla(tensor)` / `trntensor.from_xla(tensor)`** — explicit
  operand residency on the Trainium XLA device. Pre-pinning operands
  lets repeated trntensor calls skip per-dispatch host↔device
  transfer, which otherwise dominates at current kernel sizes. The
  full DF-MP2 pipeline (`ao_to_mo_transform` → `mp2_energy`) with all
  operands pre-pinned pays transfer cost once instead of once per
  call. The dispatch layer's `_to_xla` helper takes a fast path when
  every operand is already on XLA, returning the result on XLA — the
  caller decides when to pull back via `from_xla`. Third architectural
  pattern after fused-reduce and fused multi-contraction; see
  `docs/architecture.md`. Closes #34.
- **`trntensor.ao_to_mo_transform(eri, C_occ, C_vir)`** — fused 4-index
  AO→MO integral transform. One NKI program computes
  `B[i,a,P] = Σ_{μν} C_occ[μ,i] · C_vir[ν,a] · eri[μ,ν,P]` with C_occ
  and C_vir SBUF-resident across all P iterations. The per-P
  intermediate `(i, ν)` tile round-trips through kernel-scratch HBM to
  handle the partition-dim change between the two sequential
  `nc_matmul` steps. Composes with `mp2_energy` for the full DF-MP2
  pipeline from AO integrals to correlation energy. Single-tile path
  this release (`nbasis ≤ 128`); K/N tiling follow-up.
  Validated on trn1 (hardware) and via the CPU simulator CI job.

### Changed

- **Migrated to NKI 0.3.0 / Neuron SDK 2.29.** Canonical `nki.*`
  namespace; the legacy `neuronxcc.nki.*` shim is no longer used.
  Kernels (`matmul_kernel`, `batched_matmul_kernel`,
  `mp2_energy_kernel`) updated for the NKI 0.3.0 breaking-change
  surface:
  - `nisa.nc_matmul(dst=, stationary=, moving=, accumulate=True)`
    (all kwargs; internal accumulate replaces external
    `psum[...] += ...`).
  - `nl.copy(psum, ...)` returns a view; use `nl.ndarray` +
    `nisa.tensor_copy` to move PSUM → SBUF before `nl.store`.
  - Tensor-tensor `nl.divide` dropped; use `multiply × reciprocal`
    in `mp2_energy_kernel`.
- Dev workflow migrated to [uv](https://github.com/astral-sh/uv).
  `uv sync --extra dev` replaces `pip install -e ".[dev]"`; CI uses
  `astral-sh/setup-uv@v6` and invokes `uv run pytest` / `uvx ruff`.
  `uv.lock` is committed for reproducible installs.
- Removed the `[neuron]` optional-dependencies extra. `nki` (the
  runtime) is installed separately from the AWS Neuron pip index in
  CI, or provided by the Deep Learning AMI's pre-built venv on
  hardware.
- `CONTRIBUTING.md` updated to reflect the uv-based setup.

### Added

- **NKI CPU simulator dispatch** via `TRNTENSOR_USE_SIMULATOR=1`.
  Routes kernels through `nki.simulate(kernel)(numpy_args)` on CPU,
  bypassing `torch_xla` + NEFF compile. Iteration loop drops from
  ~5 min per SSM round-trip to seconds. All three dispatch wrappers
  (`nki_matmul`, `nki_batched_matmul`, `_nki_mp2_energy`) carry the
  simulator branch. Correctness-only — MLIR verifier errors remain
  hardware-only. See
  [`docs/developing_kernels.md`](docs/developing_kernels.md).
- `tests/test_nki_sim.py` — curated simulator-backed correctness
  suite, marker `nki_simulator`. Skips unless
  `TRNTENSOR_USE_SIMULATOR=1` + `nki` is importable.
- `scripts/run_simulator_tests.sh` — SSM runner that runs the
  simulator suite on the trn1 DLAMI (parity with
  `run_neuron_tests.sh`; the primary simulator runner is the CI job).
- **`nki-simulator` CI job on `ubuntu-latest`.** Runs the
  `nki_simulator`-marked suite against `nki>=0.3.0` from the AWS
  pip index (`--extra-index-url
  https://pip.repos.neuron.amazonaws.com`) on every push + PR. Zero
  AWS cost for the correctness gate; hardware SSM now reserved for
  perf + MLIR verification.
- `docs/developing_kernels.md` — thin pointer to the umbrella's
  canonical NKI how-to plus trntensor-specific env vars and file
  locations.

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
