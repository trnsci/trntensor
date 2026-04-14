# Developing NKI kernels

The canonical how-to for developing NKI kernels across the trnsci
suite lives in the umbrella repo:

👉 [trnsci/trnsci/docs/developing_kernels.md](https://github.com/trnsci/trnsci/blob/main/docs/developing_kernels.md)

That page covers NKI 0.3.0 conventions, the CPU simulator loop, the
PSUM / SBUF / Vector Engine mental model, and suite-wide testing
policy. trntensor follows those conventions verbatim.

## trntensor-specific notes

- **Env var prefix**: `TRNTENSOR_USE_SIMULATOR=1` routes dispatch
  through `nki.simulate` (see [api/nki.md](api/nki.md) for the full
  env-var table).
- **Kernels live in** `trntensor/nki/_kernels.py`.
- **Dispatch wrappers live in** `trntensor/nki/dispatch.py` —
  `nki_matmul`, `nki_batched_matmul`, `_nki_mp2_energy`.
- **Simulator tests**: `tests/test_nki_sim.py` (marker
  `nki_simulator`). Runs in the `nki-simulator` job on every PR.
- **Hardware tests**: `tests/test_nki_kernels.py` (marker `neuron`).
  Runs via `AWS_PROFILE=aws scripts/run_neuron_tests.sh` against
  `trntensor-ci-trn1`.

## Iteration loop

1. Edit the kernel in `trntensor/nki/_kernels.py`.
2. Run the simulator suite locally or via CI:
   `TRNTENSOR_USE_SIMULATOR=1 uv run pytest tests/ -m nki_simulator`.
3. Once simulator-green, validate on hardware:
   `AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn1`.

The simulator catches bad kwargs, shape mismatches, and dropped ops;
MLIR verifier errors only surface on hardware. Don't skip the
hardware pass.
