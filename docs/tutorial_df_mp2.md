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
eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1
```

## Inspecting the plan

Before running, ask the planner what it's going to do:

```python
plan = trntensor.plan_contraction("ap,bp->ab", B[0], B[1])
print(plan.strategy, plan.backend, trntensor.estimate_flops("ap,bp->ab", B[0], B[1]))
# matmul pytorch 25992
```

`strategy="matmul"` means the planner has reduced the einsum to a single matmul. `backend="pytorch"` means this *specific* call will go through `torch.matmul` — the contraction is too small (26 K multiply-adds) for the NKI dispatch overhead to pay off.

For a *large* matmul on the Trn1 / XLA path, the planner routes the same code path through NKI automatically (above `TRNTENSOR_MIN_NKI_FLOPS`, see [benchmarks](benchmarks.md)). On the native TorchNeuron path (Trn2/Trn3) matmul runs on aten on the device at every size, because aten beat the NKI matmul kernels at every measured size, and `backend` stays `"pytorch"`:

```python
plan = trntensor.plan_contraction("ij,jk->ik", torch.randn(2048, 2048), torch.randn(2048, 2048))
print(plan.backend)
# nki on Trn1 (XLA); "pytorch" on Trn2/Trn3 (native: aten on device) and on CPU
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

The contractions are dispatch-bound at this size, so they route to the PyTorch path on every backend. For real sizes use the fused `mp2_energy` below rather than a Python loop.

## What happens on Trainium vs CPU

Nothing changes in the code. On the Trn1 / XLA path, when the operands are large enough, `trntensor.einsum` routes through the NKI `matmul_kernel` transparently; when they aren't, it falls back to `torch.matmul`. On the native TorchNeuron path, matmul runs on aten (on the device for device-resident operands, or for host operands above the measured offload threshold). The `plan.backend` field tells you which path a given call will take, so a profiling loop like

```python
for i in range(nocc):
    for j in range(nocc):
        p = trntensor.plan_contraction("ap,bp->ab", B[i], B[j])
        print(i, j, p.backend, p.estimated_flops)
```

will reveal exactly where NKI is or isn't being used.

## Fusing the whole calculation

The Python loop above does 25 einsum calls, 25 elementwise passes, and 25 host reductions — each landing in HBM between steps. On Trainium, trntensor fuses the DF-MP2 energy into NKI kernels: on the XLA path (Trn1) one program, on the native path (Trn2/Trn3) one launch per occupied row (`mp2_energy_row_kernel`, using `e_ij = e_ji`), which `"auto"` picks once each launch carries at least 3 GFLOP:

```python
E = trntensor.mp2_energy(B, eps_occ, eps_vir)
```

Inside the XLA-path kernel:

1. `T_ab = Σ_P B[i,a,P] B[j,b,P]` accumulates in PSUM via `nc_matmul`
2. The spin-adapted numerator `(2T − T^T)` and the denominator `Δ_ab = ε_i + ε_j − ε_a − ε_b` are built on the Vector Engine from SBUF-resident ε tiles
3. `(T·(2T − T^T) / Δ).sum()` folds into a scalar accumulator in SBUF
4. One HBM partial per `(i, j)` pair; host sums to the final scalar

No intermediate `T` tensor is ever materialized to HBM. See [API: quantum](api/quantum.md) and [Architecture](architecture.md) for the full description.

## Starting from AO integrals

If you have AO-basis density-fitted integrals rather than pre-transformed MO coefficients, use `ao_to_mo_transform` to get `B`:

```python
import trntensor

# eri: (nbasis, nbasis, naux) density-fitted AO ERIs
# C_occ: (nbasis, nocc) occupied MO coefficients
# C_vir: (nbasis, nvir) virtual MO coefficients
B = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
E = trntensor.mp2_energy(B, eps_occ, eps_vir)
```

`ao_to_mo_transform` is another fused NKI kernel, and it also accepts packed (μ≥ν) integrals, `(npair, naux)` or PySCF's `_cderi` `(naux, npair)`, which halves the upload. See [api/quantum.md](api/quantum.md).

## Keeping tensors on the accelerator

In the above the Python runtime sees `B` as an intermediate tensor and
could try to move it back to the CPU between the two calls — paying
for a large HBM → host transfer just to send it right back to the
device for `mp2_energy`. `trntensor.to_device` and `trntensor.from_device`
(`neuron` on TorchNeuron, the XLA device on Trn1; `to_xla` / `from_xla` are
the older names) let you opt into explicit residency:

```python
eri_d = trntensor.to_device(eri)
C_occ_d = trntensor.to_device(C_occ)
C_vir_d = trntensor.to_device(C_vir)
eps_occ_d = trntensor.to_device(eps_occ)
eps_vir_d = trntensor.to_device(eps_vir)

B_d = trntensor.ao_to_mo_transform(eri_d, C_occ_d, C_vir_d)
E_d = trntensor.mp2_energy(B_d, eps_occ_d, eps_vir_d)
E = trntensor.from_device(E_d)
```

When every operand passed to a trntensor call is already on the device,
the dispatch layer skips the host↔device transfer and returns the result
on the device. Only the final `from_device(E)` pays the round-trip, and
it's pulling a 0-D scalar. On the native path under `"auto"`, large
host-input calls are also uploaded automatically above measured thresholds
(see [NKI dispatch](api/nki.md#host-operands-upload-or-stay-56)).

## Current limitations

- In the Python loop above the energy denominator is a separate element-wise op; `mp2_energy` folds it into the kernel ([#13][i13]).
- For typical chemistry sizes (tens of occupied, tens of virtuals, ~100 auxiliary), the per-pair contraction is well below any NKI dispatch threshold, and on the native path `"auto"` keeps small `mp2_energy` problems on aten or the host, where the launch floor would dominate.

[i13]: https://github.com/trnsci/trntensor/issues/13
