# einsum

Einstein summation with contraction planning. Mirrors the subscript
notation used by `torch.einsum` and `numpy.einsum`, with additional
control over scaling, accumulation precision, and mixed-precision compute.

## `einsum(subscripts, *operands, *, alpha, beta, out, dtype, precision) -> torch.Tensor`

Execute a single contraction.

```python
einsum(
    subscripts: str,
    *operands: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    out: torch.Tensor | None = None,
    dtype: str | torch.dtype | None = None,
    precision: str = "fast",
) -> torch.Tensor
```

The planner selects `matmul`, `bmm`, `path`, or `torch.einsum` based on
the subscript pattern and operand shapes. When a NKI backend is active and
the pattern maps to a 2-index matmul, the kernel in
`trntensor.nki._kernels.matmul_kernel` is used.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `subscripts` | — | Einstein summation string, e.g. `"ij,jk->ik"` |
| `*operands` | — | Input tensors |
| `alpha` | `1.0` | Scalar multiplier on the contraction result |
| `beta` | `0.0` | Scalar multiplier on `out` before accumulation |
| `out` | `None` | Accumulation tensor; result is `α·contract + β·out` |
| `dtype` | `None` | Cast all operands to this dtype before contracting; result is returned in this dtype. Accepts `torch.dtype` or strings: `"bf16"`, `"bfloat16"`, `"fp16"`, `"float16"`, `"f32"`, `"float32"` |
| `precision` | `"fast"` | Accumulation precision: `"fast"`, `"kahan"`, `"sr"`, or `"dd"` (see below) |

### Basic examples

```python
import trntensor

# Matrix multiply
C = trntensor.einsum("ij,jk->ik", A, B)

# Batched matmul
C = trntensor.einsum("bij,bjk->bik", A, B)

# DF-MP2 pair contraction
T_ab = trntensor.einsum("ap,bp->ab", B_i, B_j)

# 4-index AO→MO transform step
Inu = trntensor.einsum("mi,mnP->inP", C, eri)
```

### `alpha` / `beta` scaling

Matches cuTENSOR's GEMM-style interface:
`alpha·contract(A, B) + beta·C`.

```python
# Scaled GEMM: 2·A@B + 0.5·C
result = trntensor.einsum("ij,jk->ik", A, B, alpha=2.0, beta=0.5, out=C)

# In-place accumulation (beta=1, alpha=1): result += A@B
trntensor.einsum("ij,jk->ik", A, B, beta=1.0, out=acc)
```

### `dtype` — mixed-precision compute

Cast operands to a lower dtype to hit the NKI bf16 matmul path without
changing the model's weight dtype. The result is returned in the requested
dtype.

```python
# Route through NKI bf16 kernel even if weights are fp32
result_bf16 = trntensor.einsum("ij,jk->ik", A, B, dtype="bf16")

# Use torch.dtype directly
result = trntensor.einsum("ij,jk->ik", A, B, dtype=torch.float16)
```

### `precision` — accumulation precision

| Value | Behavior |
|-------|----------|
| `"fast"` | Native operand dtype; NKI kernels accumulate in fp32 via PSUM |
| `"kahan"` | Contracts in fp64 and rounds once back to the operand dtype; bypasses NKI dispatch. Operands on a device without native fp64 (the Neuron device, XLA) are contracted on the host and the result moved back. Error ≤ K·2⁻⁵³ + u_out, so the result is as accurate as its dtype allows (~1 ulp), not ~16 digits, unless the operands are float64 |
| `"sr"` | Stochastic rounding at the PSUM→SBUF downcast (BF16/FP16); mean-zero √K·u error instead of K·u. On Trainium requires `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` at process start (see below) |
| `"dd"` | Double-double accumulation via trnblas Phase 2 — raises `NotImplementedError` until trnblas#22 lands |

```python
# High-precision DF-MP2 energy accumulation
E_corr = trntensor.einsum("ijab,ijab->", T, T2, precision="kahan")
```

#### Error bounds and the output-dtype floor

Every mode's promise is a componentwise bound on
`max |C − C_exact| / (|A||B|)`, with `u` the operand unit roundoff and
`u_out` the unit roundoff of the result dtype (the result is returned in the
operand dtype, so `u_out = u`):

| Mode | Bound |
|------|-------|
| `"fast"` | K·u + u_out |
| `"sr"` | √K·u + u_out (expected value) |
| `"dd"` | K·u² + u_out |
| `"kahan"` | K·2⁻⁵³ + u_out |

`trntensor.forward_error_bound(mode, dtype, K)` returns these values.
The `u_out` term is the final rounding of the result and is a **floor no
accumulation mode can beat**: about 3.9e-3 for bf16 results and 6.0e-8 for
fp32. `target_forward_error=` uses these bounds to pick the cheapest mode.
When the target is below the floor (e.g. `1e-4` on bf16 inputs, `1e-8` on
fp32 inputs), `einsum` / `multi_einsum` emit a one-time `RuntimeWarning`
and still run the most accurate mode, `"kahan"`. To actually meet such a
target, pass float32 or float64 operands (or `dtype=`), or loosen the
target. A target below K·2⁻⁵³ raises `ValueError`.

#### Stochastic rounding on Trainium is a runtime setting

On NKI 0.7+ (Neuron SDK 2.33) the kernel cannot pick its own rounding mode;
the Neuron Runtime picks one for the whole process at init from
`NEURON_RT_STOCHASTIC_ROUNDING_EN`
([AWS: rounding modes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/rounding-modes.html)).

| `NEURON_RT_STOCHASTIC_ROUNDING_EN` | `precision="sr"` | `precision="fast"` |
|---|---|---|
| unset or `0` | round-to-nearest-even + one-time `RuntimeWarning` | round-to-nearest-even |
| `1` | stochastic | **also stochastic** |

- Set the variable **before the process starts** (or at least before the
  first Neuron device access). Setting it later from Python has no effect
  on rounding.
- With it set, *every* FP32→BF16/FP16 downcast in the process rounds
  stochastically, including `"fast"` contractions and other libraries'
  kernels. Results are not bitwise reproducible run to run.
- Without it, `"sr"` still computes but gives round-to-nearest results, so
  the √K·u bound does not apply. This also covers `target_forward_error=`
  when it selects `"sr"`.
- The CPU fallback (no Neuron hardware) emulates SR in software and is not
  affected by the variable.

```bash
NEURON_RT_STOCHASTIC_ROUNDING_EN=1 python my_bf16_job.py
```

---

## `multi_einsum(*contractions) -> list[torch.Tensor]`

Execute several contractions in one call. Each contraction is a tuple
of `(subscripts, *operands)`. Results are returned in input order.

When NKI dispatch is active, operand tensors that appear in more than one
contraction (by object identity) are pre-pinned to the XLA device once
before executing the loop, eliminating redundant host↔device transfers.

```python
results = trntensor.multi_einsum(
    ("ap,bp->ab", B_i, B_j),  # Coulomb term
    ("ap,bp->ab", B_i, B_k),  # exchange term — B_i pinned once
)
```

### Plan cache helpers

These are documented on the [Planning](plan.md) page but are also
relevant to einsum performance:

- `trntensor.clear_plan_cache()` — discard all cached plans
- `trntensor.plan_cache_info()` — return `{"size": N}` cache statistics
