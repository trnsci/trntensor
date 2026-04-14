# Quantum-chemistry primitives

Domain-specific fused kernels where the whole computation — contraction, elementwise shaping, reduction — lives in a single NKI program. The architectural core of trntensor (see [Architecture](../architecture.md)).

## `trntensor.mp2_energy(B, eps_occ, eps_vir) -> scalar`

Density-fitted second-order Møller–Plesset correlation energy.

```
E_MP2 = Σ_{i,j,a,b} T_{i,j,a,b} (2 T_{i,j,a,b} - T_{i,j,b,a}) / Δ_{i,j,a,b}

T_{i,j,a,b}  = Σ_P  B[i, a, P] B[j, b, P]
Δ_{i,j,a,b}  = ε_i + ε_j - ε_a - ε_b
```

### Arguments

- `B: (nocc, nvir, naux) tensor` — density-fitted ERI coefficients
- `eps_occ: (nocc,) tensor` — occupied orbital energies
- `eps_vir: (nvir,) tensor` — virtual orbital energies

### Returns

A 0-D tensor containing the correlation energy.

### Example

```python
import torch
import trntensor

nocc, nvir, naux = 5, 19, 72
B = torch.randn(nocc, nvir, naux) * 0.1
eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
eps_vir =  torch.sort(torch.rand(nvir))[0] + 0.1

E = trntensor.mp2_energy(B, eps_occ, eps_vir)
print(f"E_MP2 = {E.item():.6f}")
```

### Backend behaviour

- **CPU**: falls back to a Python loop over `(i, j)` pairs composing `torch.einsum`
  and element-wise ops. Same as `examples/df_mp2_einsum.py`.
- **Trainium (NKI)**: dispatches a single `@nki.jit` program that
  - accumulates `T` in PSUM via `nisa.nc_matmul`,
  - builds `Δ` on the Vector Engine from SBUF-resident `ε` tiles,
  - folds the energy into a scalar accumulator in SBUF,
  - writes one partial per `(i, j)` pair to HBM.

The host sums the `(nocc, nocc)` partial matrix into the final scalar. No intermediate four-index `T` tensor is ever materialized.

### Current limitations

- Single-tile path only: `nvir ≤ 128` and `naux ≤ 128`. Larger systems raise `NotImplementedError` — K/M tiling is a follow-up.
- Dispatch overhead still dominates at small sizes; see [Benchmarks](../benchmarks.md) for the honest comparison against the Python loop.


## `trntensor.ao_to_mo_transform(eri, C_occ, C_vir) -> Tensor`

Fused 4-index AO→MO integral transform. The canonical quantum-chemistry primitive.

```
B[i, a, P] = Σ_{μ, ν} C_occ[μ, i] · C_vir[ν, a] · eri[μ, ν, P]
```

### Arguments

- `eri: (nbasis, nbasis, naux) tensor` — density-fitted two-electron integrals in the AO basis
- `C_occ: (nbasis, nocc) tensor` — occupied molecular orbital coefficients
- `C_vir: (nbasis, nvir) tensor` — virtual molecular orbital coefficients

### Returns

A `(nocc, nvir, naux)` tensor of transformed DF coefficients B_ia^P.

### Example — full DF-MP2 pipeline

```python
import trntensor

B = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
E = trntensor.mp2_energy(B, eps_occ, eps_vir)
```

### Backend behaviour

- **CPU**: `torch.einsum("mi,na,mnP->iaP", C_occ, C_vir, eri)` — a single three-operand einsum.
- **Trainium (NKI)**: dispatches a single `@nki.jit` program that
  - loads `C_occ` and `C_vir` once, SBUF-resident across all P iterations,
  - for each auxiliary index P: two sequential `nisa.nc_matmul` calls with the intermediate `(i, ν)` tile round-tripping through kernel-scratch HBM to handle the partition-dim change,
  - writes one `(nocc, nvir)` slice per P to the output HBM tensor.

The "two-step decomposed" form (`intermediate = einsum("mi,mnP->inP", ...)` followed by `einsum("na,inP->iaP", ...)`) materializes the `(nocc, nbasis, naux)` intermediate as a user-visible torch tensor. The fused NKI path keeps it in kernel scratch — the user only ever sees `eri`, `C_occ`, `C_vir`, and `B`.

### Current limitations

- Single-tile path only: `nbasis ≤ 128`, `nocc * naux ≤ 512`, and `nvir * naux ≤ 512`. Larger systems raise `NotImplementedError`. K-tiling and N-tiling are a follow-up.
- The intermediate briefly visits kernel-scratch HBM to change partition dim between the two nc_matmul steps. A future version may keep it fully SBUF-resident if NKI adds an in-SBUF transpose primitive.
