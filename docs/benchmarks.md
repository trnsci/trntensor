# Benchmarks

Performance results for trntensor — einsum dispatch, contraction planning, and CP / Tucker decompositions — comparing the PyTorch fallback and the NKI Trainium path on the same machine.

!!! note "These tables are the Trn1 / PyTorch-XLA results"
    Everything below was measured on trn1.2xlarge through the XLA path
    (neuronxcc 2.24), which is now in maintenance. On the native TorchNeuron
    path (Trn2/Trn3, Neuron SDK 2.33) dense matmul/bmm run on aten, which
    beat the NKI matmul kernels at every measured size (those kernels are
    deprecated there, 0.19.0). The Trn2 numbers for the fused kernels
    (`mp2_energy` 1.25–1.9× vs aten, AO→MO 1.3–1.6×, `df_exchange` K at
    n_ao = 970 304 → 94 ms, DF-RHF SCF at n_ao = 970 9.68 → 4.71 s) live in
    [Quantum-chemistry primitives](api/quantum.md); the measured native
    dispatch thresholds and host-input tables are in
    [NKI dispatch](api/nki.md#native-thresholds-measured-on-trn2).

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

## DF-MP2 energy — fused vs Python-loop

Fused `trntensor.mp2_energy` (single NKI program) vs the reference loop from `examples/df_mp2_einsum.py` (25–256 einsum calls), both on trn1.2xlarge:

| Workload | Python loop | Fused NKI | Winner |
|---|---:|---:|---|
| `(nocc, nvir, naux) = (5, 19, 72)` | **1.5 ms** | 15.6 ms | Loop 10× |
| `(nocc, nvir, naux) = (16, 128, 128)` | **25.5 ms** | 41.3 ms | Loop 1.6× |

The fused kernel is **architecturally** what we want — one program, no intermediate HBM materialization of the four-index `T` tensor, PSUM/SBUF-resident across contract → elementwise → reduce. But NKI's per-call dispatch + compile overhead (~15–40 ms floor) still eats the win at current sizes. The gap closes with scale (10× → 1.6× as workload grows), consistent with the overhead being a fixed cost the compute amortizes. See [#33][i33] and [#34][i34].



## Size-based dispatch threshold (XLA path)

Because per-call NKI dispatch currently carries ~1 ms of XLA launch overhead, `nki_matmul` and `nki_batched_matmul` short-circuit to the PyTorch path when the contraction is below `TRNTENSOR_MIN_NKI_FLOPS` (default **2 GFLOPs**, calibrated at ≈ half the smallest NKI-winning size). The `plan.backend` field reflects this: it reports `"nki"` only when the dispatch will actually invoke a kernel, and `"pytorch"` otherwise.

Overrides:

- `TRNTENSOR_MIN_NKI_FLOPS=0` — always attempt NKI (useful for kernel validation).
- `TRNTENSOR_FORCE_BACKEND=pytorch` — always use the PyTorch path (useful for CPU baselines).
- `trntensor.set_backend("pytorch")` — same, via API.

## Interpretation

The kernels themselves are **correct** (validated by 9 hardware tests in `tests/test_nki_kernels.py`) and the 2048×2048 case confirms NKI can beat CPU at large enough workloads. The gap at smaller sizes is dominated by per-call overhead — not by the Tensor Engine itself — and is tracked as follow-up work in [#33][i33].

Recommendation for users:

- Call trntensor normally. The dispatch layer makes the right choice for typical sizes.
- On the XLA path, for GEMM-dominant workloads with single contraction sizes ≥ 2 GFLOPs, NKI will be invoked automatically. On the native path, matmul/bmm run on aten.
- Tight loops of small contractions (DF-MP2 pair-energy style) see no NKI benefit; use the fused `trntensor.mp2_energy` instead of a Python loop.

[i33]: https://github.com/trnsci/trntensor/issues/33
[i34]: https://github.com/trnsci/trntensor/issues/34
