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
  padded `B`, so near a full device it needs more memory than aten does.
  `"auto"` therefore keeps aten when this process's live device allocations
  plus that staging would leave less than 3 GiB free. A shape whose kernel
  launch fails anyway falls back to aten and is remembered for the rest of
  the process.
- **Host `B` on native TorchNeuron (#56).** Under `"auto"`, a host `B` with
  `nocc²·nvir²·naux` of at least 1.5e8 multiply-adds (scaled by
  `torch.get_num_threads() / 2`) is uploaded with the orbital energies, the
  device route above runs, and the energy comes back as a host scalar.
  Smaller problems stay on the host. On Trn2 with 2 host threads this took
  646 ms -> 16 ms at 32/224/1024 and 15.8 s -> 135 ms at 64/448/1536. See
  [host operands](nki.md#host-operands-upload-or-stay-56) for the table and
  `TRNTENSOR_HOST_ROUTE`.
- **PyTorch/XLA path (Trn1)**: the original single-launch `mp2_energy_kernel`.
  It accumulates `T` in PSUM, builds `Δ` on the Vector Engine, and writes one
  partial per `(i, j)` pair.

## `trntensor.ao_to_mo_transform(eri, C_occ, C_vir) -> Tensor`

Fused 4-index AO→MO integral transform. The canonical quantum-chemistry primitive.

```
B[i, a, P] = Σ_{μ, ν} C_occ[μ, i] · C_vir[ν, a] · eri[μ, ν, P]
```

### Arguments

- `eri` — density-fitted three-centre integrals in the AO basis, one of:
  - `(nbasis, nbasis, naux)`: the full square;
  - `(npair, naux)`, `npair = nbasis(nbasis+1)/2`: the packed lower triangle, row `μ(μ+1)/2 + ν` for `ν ≤ μ` (see [Packed integrals](#packed-integrals));
  - `(naux, npair)`: PySCF's `_cderi`, as is. When `naux == npair` the shape is ambiguous and is read as `(npair, naux)`; pass `cderi.T` then.
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

### Packed integrals

Real three-centre integrals are symmetric in μ↔ν, and drivers store only the
lower triangle (PySCF `_cderi`, `trnblas.parallel.pack_tril`). Pass it packed
(#54):

```python
B = trntensor.ao_to_mo_transform(cderi, C_occ_d, C_vir_d)  # cderi: host (naux, npair)
```

- `trntensor.pack_tril(eri)` and `trntensor.unpack_tril(eri_packed, nbasis)`
  convert between the layouts (one `index_select` each, exact).
- **A host `eri` with device `C_occ` is uploaded to `C_occ`'s device**, in the
  layout given. The result stays on the device.
- On the native path, `"auto"` sends packed device integrals to
  `ao_to_mo_trn2_packed_kernel` under the same conditions as the full kernel.
  Its stage 1 fetches each `(128 μ, NUB ν, TP aux)` block from the packed
  rows with one batched indirect DMA. The full square never exists in device
  memory. Its result is bit-identical to the full kernel's.
- The aten and legacy routes unpack on the tensor's device first.

Why: on Trn2 the host link carries about 13.8 GB/s for the whole chip, so the
upload of `eri` was DF-MP2's largest stage. Packed, it is half the bytes.

Upload of `eri` alone (Trn2, one logical core; host threads pinned to 2 CPUs):

| nbasis/naux | full | packed | packed, from pinned host memory |
|---|---|---|---|
| 384/1152 | 49.2 ms | 25.1 ms | 24.9 ms |
| 512/1536 | 115.7 ms | 58.5 ms | 58.4 ms |

Pinned host memory and `copy_` into a preallocated device tensor upload no
faster: H2D already runs at the link rate from ordinary host memory.
Packing a full host tensor costs more than it saves: 180 ms on 2 host
threads at 512/1536, against a 57 ms saving. Neither does uploading the
triangle row by row (281 ms). So a 3-D host `eri` is uploaded as is.
Packing pays only when the integrals already arrive packed.

Kernel alone, device-resident, warm (fp32; padded `nvir` included):

| nbasis/naux/nocc | LNC=2 full | LNC=2 packed | LNC=1 full | LNC=1 packed | device `unpack_tril` |
|---|---|---|---|---|---|
| 256/768/32 | 1.09 ms | 0.91 ms | 1.86 ms | 1.93 ms | — |
| 384/1152/48 | 5.38 ms | 5.81 ms | 5.72 ms | 5.95 ms | 3.1 ms |
| 512/1536/64 | 6.28 ms | 6.27 ms | 12.87 ms | 12.75 ms | 5.5 ms |

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

- **Host inputs on native TorchNeuron (#56).** Under `"auto"`, with `eri`,
  `C_occ` and `C_vir` all on the host, the inputs are uploaded (`eri` in the
  layout given), the device route above runs, and the result is downloaded,
  once `nocc·nbasis²·naux` reaches 2e8 for a full `eri` or 3e7 for a packed
  one (both scaled by `torch.get_num_threads() / 2`). A packed host `eri`
  otherwise pays a host unpack, so its threshold is lower. On Trn2 with 2
  host threads at 512/1536/64 this took 574 -> 137 ms full and 979 -> 80 ms
  packed. PySCF's `(naux, npair)` layout went from 5.2 s to 83 ms on one
  physical core. See [host operands](nki.md#host-operands-upload-or-stay-56)
  for the table and `TRNTENSOR_HOST_ROUTE`.
- **PyTorch/XLA path (Trn1)**: the original kernel. It loops over P, with the
  `(i, ν)` intermediate round-tripping through kernel-scratch HBM. On the
  native path it is deprecated.

### Current limitations

- Both NKI kernels need `nocc ≤ 128`; larger `nocc` runs the aten steps (native) or raises `NotImplementedError` (XLA).
- The `(nocc, nbasis, naux)` intermediate goes through kernel-scratch HBM between the two stages, as it does between the two aten einsums. Stage 1 leaves `i` on the partition axis, and stage 2 needs `ν` there.

## `trntensor.df_exchange(B, C_occ, alpha=1.0) -> Tensor`

Density-fitted exchange matrix straight from packed three-centre integrals (#60).

```
K[μ, ν] = alpha · Σ_{i, P} X[i, μ, P] X[i, ν, P],    X[i, ν, P] = Σ_λ C_occ[λ, i] B[λν, P]
```

For a closed-shell density `D = 2 C_occ C_occᵀ`, the SCF exchange matrix is
`df_exchange(B, C_occ, alpha=2.0)`.

### Arguments

- `B` — the packed lower triangle `(npair, naux)` (row `μ(μ+1)/2 + ν`,
  `pack_tril` order), PySCF's `_cderi` `(naux, npair)`, or the full
  `(nbasis, nbasis, naux)` square (aten route).
- `C_occ: (nbasis, nocc)` — occupied orbital coefficients.
- `alpha` — scale factor.

### Returns

The symmetric `(nbasis, nbasis)` matrix, on `B`'s device.

### Example — the K build of a DF-RHF iteration

```python
B_d = B_packed.to("neuron")          # once: (npair, naux), never unpacked
K = trntensor.df_exchange(B_d, C_occ_d, alpha=2.0)
```

### Backend behaviour

- **Native TorchNeuron (Trn2/Trn3): two NKI launches per group of aux chunks.**
  - `df_half_packed_kernel` gathers the packed rows in place with batched
    indirect DMA (the gather of `ao_to_mo_transform`'s packed kernel, #54)
    and forms the half-transformed `Xt[P, i, ν]` with the aux index on the
    partition axis: each gathered `(128 λ, aux)` block is the stationary
    operand and the resident `C_occ` λ-tile the moving one. The full
    `(n, n, naux)` square never exists.
  - `df_exchange_syrk_kernel` accumulates `K` in PSUM over 128-row aux
    tiles, computes only the upper-triangle 128-row strips, and mirrors them
    on the Tensor Engine on the way out. Two running partial Ks are chained
    through the launches, so there are no extra adds per group.
  - Both launches split across the programs of the logical core: the first
    by aux chunks (or ν groups), the second by occupied orbitals (padded to
    an even count). At LNC=2 both physical cores work.
  - The aux chunk width is a divisor of naux, so `B` is never copied or
    padded. Traced NKI loops unroll, so the dispatcher runs the pair over
    groups of chunks to bound the compile time (about a minute per shape at
    n_ao = 970) and the `Xt` scratch.
  - `"auto"` uses the kernels for device-resident `B` with `nocc ≤ 512`
    once `nocc·nbasis²·naux` reaches 1e7 multiply-adds.
    `TRNTENSOR_DF_EXCHANGE=auto|trn2|aten` overrides the choice.
- **Otherwise (CPU, small problems)**: aten on `B`'s device. Blocks of rows
  of the square are gathered (1 GiB at a time), half-transformed with one
  einsum into `(ν, i, P)` order, and `K` is one GEMM.

### Measured on Trn2

PySCF cc-pVDZ / cc-pvdz-jkfit integrals, fp32, warm medians, device-resident
inputs, one logical core. *Unpack path* is the K build of
`trnsolver/examples/scf_df_rhf.py` before this change: `unpack_tril` of
aux blocks (the square kept resident when it fits in 4 GB), the
half-transform `einsum`, and `trnblas.syrk`. Error is the max-abs error over
max |K| against float64.

| system | n_ao / naux / nocc | LNC | unpack path | `df_exchange` | speedup | error, unpack | error, `df_exchange` |
|---|---|---|---|---|---|---|---|
| water | 24 / 116 / 5 | 2 | 0.82 ms | 1.32 ms (kernels; `"auto"` runs aten, 1.16 ms) | — | 7.5e-8 | 1.9e-7 |
| benzene | 114 / 558 / 21 | 2 | 1.43 ms | 1.35 ms | 1.06x | 5.8e-7 | 2.6e-7 |
| C10H22 | 250 / 1206 / 41 | 2 | 3.17 ms | 2.18 ms | 1.46x | 3.6e-7 | 4.8e-7 |
| C20H42 | 490 / 2366 / 81 | 2 | 15.5 ms | 9.33 ms | 1.66x | 6.1e-7 | 6.6e-7 |
| C40H82 | 970 / 4686 / 161 | 2 | 304 ms | 118 ms | 2.58x | 8.3e-7 | 8.1e-7 |
| C10H22 | 250 / 1206 / 41 | 1 | 4.71 ms | 3.46 ms | 1.36x | 3.6e-7 | 8.1e-7 |
| C20H42 | 490 / 2366 / 81 | 1 | 25.9 ms | 16.9 ms | 1.53x | 6.1e-7 | 1.1e-6 |
| C40H82 | 970 / 4686 / 161 | 1 | 424 ms | 230 ms | 1.84x | 8.3e-7 | 1.5e-6 |

- Up to n_ao = 490 the unpack path keeps its square resident (it fits in
  4 GB), so it pays no gather per call; at 970 it unpacks 9 aux blocks per
  call. `df_exchange` never builds the square.
- `df_exchange` scales 1.59x / 1.81x / 1.95x from LNC=1 to LNC=2 at
  n_ao = 250 / 490 / 970.
- At n_ao = 970 each launch pair takes 2 aux chunks of 213; there are 11
  pairs. Padding naux to 4736 (chunks of 128, so no partial tiles) measured
  126 ms, not faster: the narrower gathered rows cost more than the partial
  tiles. Eight PSUM banks per SYRK pass instead of six made no difference.
- Without the `Xt[P, g, i, v]` store layout (ν innermost, 64-byte DMA
  writes) the same call took 488 ms.

### Current limitations

- `nocc ≤ 512` (one PSUM bank of the stage-1 output).
- `naux` needs a divisor in [32, 512] (else the aten route runs).
- The kernels run the Tensor Engine in fp32, which is what bounds them now.
  At n_ao = 970 the half transform runs at about 17 TFLOP/s and the SYRK at
  about 26 TFLOP/s per LNC=2 core (each counts the padded tiles).
- Host `B` runs the aten route on the host; upload it once and keep it on
  the device.

## `trntensor.df_coulomb(B, D) -> Tensor`

The density-fitted Coulomb matrix from the same packed integrals (#60).

```
J[μ, ν] = Σ_P B[μν, P] d[P],    d[P] = Σ_{λσ} D[λ, σ] B[λσ, P]
```

`D` is a symmetric `(nbasis, nbasis)` density, or its weighted packed form
`(npair,)` (`w·D[μ, ν]` for `ν ≤ μ`, `w` = 1 on the diagonal and 2 off
it), in which case the packed `(npair,)` J comes back. A caller that keeps
the aux index in blocks then packs and unpacks once per call instead of per
block; on the device those are element gathers.

- **Native TorchNeuron**: two launches, `df_coulomb_d_kernel` and
  `df_coulomb_j_kernel`. Each streams `B` once in contiguous 128-row tiles
  (DMAs up to 4096 columns wide), rows split across the programs, with fused
  Vector Engine instructions: a multiply-add per tile for `d`, a
  multiply-reduce per 128 columns for `J`. The partial sums are kept short:
  `d` in groups of 32 tiles, `J` in 128-column chunks. At n_ao = 970 one
  fp32 chain over all 4686 columns gave J 2.4e-6 off float64 and
  512-column chunks 5.7e-7; 128-column chunks give 2.8e-7. The < 256
  leftover rows run as aten GEMVs. `"auto"` uses the kernels for
  device-resident `B` from `npair·naux` = 1e7.
  `TRNTENSOR_DF_COULOMB=auto|trn2|aten` overrides.
- **Otherwise**: two GEMVs.

Measured on Trn2 (fp32, warm; the J of `scf_df_rhf.py`'s `DFJK` on the
core-guess density, error vs float64 relative to max |J|):

| n_ao / naux | LNC | two GEMVs per aux block (before) | `df_coulomb`, one block | speedup |
|---|---|---|---|---|
| 250 / 1206 | 2 | 3.81 ms, 2.1e-7 | 2.90 ms, 2.4e-7 | 1.31x |
| 490 / 2366 | 2 | 12.6 ms, 1.9e-7 | 8.83 ms, 2.7e-7 | 1.43x |
| 970 / 4686 | 2 | 86.7 ms (9 blocks), 1.9e-7 | 46.0 ms, 2.8e-7 | 1.89x |
| 970 / 4686 | 1 | 126 ms (9 blocks), 2.4e-7 | 81.7 ms, 2.6e-7 | 1.55x |

It also allows a single block: at n_ao = 970 the aten GEMV on one 8.8 GB
block compiled to a 26.7 GB graph and did not fit in HBM. With the DMA as
narrow as the 128-column reduce, the kernel was descriptor-bound (76 ms at
970).
