# Tutorial: DF-MP2 with trntensor

Density-fitted second-order Møller–Plesset perturbation theory (DF-MP2) is a widely-used post-Hartree–Fock method in quantum chemistry. It's also a textbook-perfect workload for a tensor library — the whole calculation is a stack of einsum contractions over occupied, virtual, and auxiliary indices. This tutorial walks through implementing DF-MP2 with `trntensor` and shows how the contraction planner routes each step.

Full runnable code lives at [`examples/df_mp2_einsum.py`](https://github.com/trnsci/trntensor/blob/main/examples/df_mp2_einsum.py).

## The calculation in 100 words

DF-MP2 compresses the 4-index two-electron integral tensor `(ia|jb)` into a 3-index tensor `B_ia^P` via the resolution of the identity. The correlation energy is

$$
E_\text{MP2} = \sum_{ijab} \frac{T_{ijab}\,(2T_{ijab} - T_{ijba})}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
$$

where $T_{ijab} = \sum_P B_{ia}^P B_{jb}^P$. Each $(i,j)$ pair needs a matmul; the energy denominator is an element-wise division. Natural territory for an einsum library.

## Setup

```python
import torch
import trntensor

nocc, nvir, naux = 5, 19, 72
torch.manual_seed(42)
B = torch.randn(nocc, nvir, naux) * 0.1
eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
eps_vir =  torch.sort(torch.rand(nvir))[0] + 0.1
```

## Inspecting the plan

Before running, ask the planner what it's going to do:

```python
plan = trntensor.plan_contraction("ap,bp->ab", B[0], B[1])
print(plan.strategy, plan.backend, plan.estimated_flops)
# matmul pytorch 116736
```

`strategy="matmul"` means the planner has reduced the einsum to a single matmul. `backend="pytorch"` means this *specific* call will go through `torch.matmul` — the contraction is too small (116 K FLOPs) for the NKI dispatch overhead to pay off. See [benchmarks](benchmarks.md) for the threshold.

For a *large* matmul, the planner routes the same code path through NKI automatically:

```python
plan = trntensor.plan_contraction(
    "ij,jk->ik", torch.randn(2048, 2048), torch.randn(2048, 2048)
)
print(plan.backend)
# nki   (on Trainium; "pytorch" elsewhere)
```

## The pair-energy loop

```python
def df_mp2_energy(B, eps_occ, eps_vir):
    nocc = B.shape[0]
    e_mp2 = 0.0
    for i in range(nocc):
        for j in range(nocc):
            T = trntensor.einsum("ap,bp->ab", B[i], B[j])
            denom = eps_occ[i] + eps_occ[j] - eps_vir.unsqueeze(1) - eps_vir.unsqueeze(0)
            e_mp2 += (T * (2 * T - T.T) / denom).sum().item()
    return e_mp2
```

Each iteration is:

1. `einsum("ap,bp->ab", B[i], B[j])` — contracts over the auxiliary index `p`. With `B[i]` of shape `(nvir, naux)`, this is exactly a matmul `B[i] @ B[j].T`. The planner picks `strategy="matmul"` with `transB=True`.
2. Denominator tensor built via broadcasting.
3. Antisymmetrized numerator divided element-wise and summed.

## Running it

```python
>>> e = df_mp2_energy(B, eps_occ, eps_vir)
>>> e
-50.5432928801
```

On a trn1.2xlarge, 25 pair contractions complete in ~3 s end-to-end. The contractions themselves are dispatch-bound at this size, so they route to the PyTorch path; the NKI kernels would only kick in if `nocc×nvir` were in the thousands.

## What happens on Trainium vs CPU

Nothing changes in the code. When `neuronxcc` is installed and the operands are large enough, `trntensor.einsum` routes through the NKI `matmul_kernel` transparently. When they aren't, it falls back to `torch.matmul`. The `plan.backend` field tells you which path a given call will take, so a profiling loop like

```python
for i in range(nocc):
    for j in range(nocc):
        p = trntensor.plan_contraction("ap,bp->ab", B[i], B[j])
        print(i, j, p.backend, p.estimated_flops)
```

will reveal exactly where NKI is or isn't being used.

## Fusing the whole calculation

The Python loop above does 25 einsum calls, 25 elementwise passes, and 25 host reductions — each landing in HBM between steps. On Trainium, trntensor can compile the entire DF-MP2 energy into **one NKI program**:

```python
E = trntensor.mp2_energy(B, eps_occ, eps_vir)
```

Inside that single call:

1. `T_ab = Σ_P B[i,a,P] B[j,b,P]` accumulates in PSUM via `nc_matmul`
2. The spin-adapted numerator `(2T − T^T)` and the denominator `Δ_ab = ε_i + ε_j − ε_a − ε_b` are built on the Vector Engine from SBUF-resident ε tiles
3. `(T·(2T − T^T) / Δ).sum()` folds into a scalar accumulator in SBUF
4. One HBM partial per `(i, j)` pair; host sums to the final scalar

No intermediate `T` tensor is ever materialized to HBM. See [API: quantum](api/quantum.md) and [Architecture](architecture.md) for the full description.

## Current limitations

- The energy denominator is a separate element-wise op. A fused kernel that folds the division into the PSUM accumulation — `E = Σ T²/Δ` in one kernel — is tracked in [#13][i13]. Landing it collapses the pair-energy loop into a single tensor-engine invocation per `(i,j)`.
- For typical chemistry sizes (tens of occupied, tens of virtuals, ~100 auxiliary), the per-pair contraction is well below the NKI dispatch threshold. NKI wins start at GEMMs of ~2048² and above. See [#33][i33] for overhead reduction work.

[i13]: https://github.com/trnsci/trntensor/issues/13
[i33]: https://github.com/trnsci/trntensor/issues/33
