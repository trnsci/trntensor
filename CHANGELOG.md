# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Dev workflow migrated to [uv](https://github.com/astral-sh/uv).
  `uv sync --extra dev` replaces `pip install -e ".[dev]"`; CI uses
  `astral-sh/setup-uv@v6` and invokes `uv run pytest` / `uvx ruff`.
  `uv.lock` is committed for reproducible installs.
- Removed the `[neuron]` optional-dependencies extra. `neuronxcc` and
  the NKI runtime come from the AWS Deep Learning AMI's pre-built venv
  (`/opt/aws_neuronx_venv_pytorch_*`), not pip — the extra was never
  resolvable against any index. The DLAMI tracks Neuron SDK 2.29 /
  NKI 0.3.0 Stable as of 2026-04-09.
- `CONTRIBUTING.md` updated to reflect the uv-based setup.

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
