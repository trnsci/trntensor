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
eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1

E = trntensor.mp2_energy(B, eps_occ, eps_vir)
print(f"E_MP2 = {E.item():.6f}")
```

### Backend behaviour

- **CPU, and small problems on native TorchNeuron**: aten ops on the
  operands' device. `T[i, j, a, b]` is contracted with `torch.einsum` in
  chunks over `i`, which bounds peak memory to about 64 MiB of `T` at a time.
  The `T`-weighted sum is vectorized.
- **Native TorchNeuron (Trn2/Trn3): the row kernel.** `mp2_energy_row_kernel`
  is launched once per occupied row `i`. It covers `j = i + d (mod nocc)` for
  `d = 0..nocc/2` and uses the pair symmetry `e_ij = e_ji`. Per pair, one
  GEMM fills `T = B[i] B[j]ᵀ` in SBUF, and the `Tᵀ` blocks come from Tensor
  Engine transposes. `T` never reaches device memory. At LNC=2 each physical
  core takes half of the `j` window. The design follows trnblas
  `pair_energy_sym` (trnblas#55, #58).
  - `"auto"` uses the row kernel when `B` is on the device and each row
    launch carries at least 3 GFLOP, counted on the padded sizes as
    `2·(nocc//2+1)·nvir²·naux`. Below that, the ~nocc custom-op launch
    floors lose to aten.
  - `set_backend("nki")` uses it whenever its SBUF working set fits.
  - `TRNTENSOR_MP2_ENERGY=auto|rows|aten|legacy` overrides the choice.
    `legacy` is the original single-launch kernel, which is deprecated on
    the native path.

  Measured on Trn2 (fp32, one LNC=2 core, device-resident, warm, #51):

  | nocc, nvir, naux | aten | row kernel |
  |---|---|---|
  | 16, 128, 512 | 1.5 ms | 5.1 ms |
  | 32, 256, 768 | 12.1 ms | 11.6 ms |
  | 32, 320, 1024 | 24.9 ms | 19.8 ms |
  | 48, 384, 1152 | 75.7 ms | 39.2 ms |
  | 64, 448, 1536 | 240 ms | 120 ms |
  | 96, 672, 2304 | 1.30 s | 0.69 s |

  Holding 12 or 18 GB of the core's 24 GB on the device did not slow either
  path. The aten path's 64 MiB `T` chunks stay clear of the slowdown in
  aws-neuron-sdk#1447. The row kernel's input staging is about 1.5× the
  padded `B`.
- **PyTorch/XLA path (Trn1)**: the original single-launch `mp2_energy_kernel`.
  It accumulates `T` in PSUM, builds `Δ` on the Vector Engine, and writes one
  partial per `(i, j)` pair.

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

- **CPU**: two explicit aten steps on the operands' device:
  `einsum("mi,mnP->inP")` then `einsum("na,inP->iaP")`.
- **Native TorchNeuron (Trn2/Trn3): `ao_to_mo_trn2_kernel`.** Both stages are
  plain GEMMs with the contraction index on the partition axis, so there are
  no DMA transposes.
  - Stage 1: `X[i, ν, P] = Σ_μ C_occ[μ, i] eri[μ, ν, P]`.
  - Stage 2: `B[i, a, P] = Σ_ν C_vir[ν, a] X[i, ν, P]`.
  - The auxiliary index is split across both physical cores at LNC=2.
  - `"auto"` uses the kernel for device-resident `eri` with `nocc ≤ 128` when
    its SBUF working set fits and the stage-1 work `nocc·nbasis²·naux` is at
    least 5e8 multiply-adds (about nbasis ≥ 256). Smaller problems lose to
    the launch floor and the padding copies; otherwise it runs the two aten
    steps.
    `TRNTENSOR_AO_TO_MO=auto|trn2|aten|legacy` overrides the choice.

  Measured on Trn2 (fp32, one LNC=2 core, device-resident, warm, #51):

  | nbasis, nocc, naux | two aten einsums | Trn2 kernel (op) | `ao_to_mo_transform`, `"auto"` | original kernel |
  |---|---|---|---|---|
  | 64, 8, 128 | 0.41 ms | — | 0.41 ms (aten) | 6.0 ms (#49) |
  | 128, 16, 256 | 0.49 ms | 0.43 ms | 0.49 ms (aten) | 16.1 ms |
  | 256, 32, 512 | 1.23 ms | 1.11 ms | 0.92 ms | 112 ms (#49) |
  | 512, 32, 1024 | 7.30 ms | 4.56 ms | 4.72 ms | — |
  | 768, 48, 1536 | 18.3 ms | 11.8 ms | 12.0 ms | — |

  The op column takes tile-multiple inputs (the raw kernel). The `"auto"`
  column includes the padding of `nvir` to 128 and is the shipped route. At
  nbasis 64 and 128 it measured 0.79 and 0.57 ms with the kernel forced,
  which is why `"auto"` keeps aten there.

- **PyTorch/XLA path (Trn1)**: the original kernel. It loops over P, with the
  `(i, ν)` intermediate round-tripping through kernel-scratch HBM. On the
  native path it is deprecated.

### Current limitations

- Both NKI kernels need `nocc ≤ 128`; larger `nocc` runs the aten steps (native) or raises `NotImplementedError` (XLA).
- The `(nocc, nbasis, naux)` intermediate goes through kernel-scratch HBM between the two stages, as it does between the two aten einsums. Stage 1 leaves `i` on the partition axis, and stage 2 needs `ν` there.
