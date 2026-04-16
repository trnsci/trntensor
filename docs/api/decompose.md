# Decompositions

CP (CANDECOMP/PARAFAC), Tucker (HOSVD), and Tensor Train (TT-SVD)
decompositions for compressing high-order tensors. Used in quantum chemistry
for tensor hypercontraction (THC), low-rank factorization of DF-coefficient
tensors, and DMRG-style high-dimensional compression.

## CP (PARAFAC)

### `cp_decompose(tensor, rank, *, max_iter=100, tol=1e-6, nonneg=False, factors=None) -> (factors, weights)`

Decompose a tensor as `T ≈ Σ_r w_r · a_r ⊗ b_r ⊗ c_r ⊗ ...` via
alternating least squares (ALS).

- `factors`: list of factor matrices, one per mode with shape `(I_k, R)`
- `weights`: `(R,)` tensor of component weights

| Argument | Default | Description |
|----------|---------|-------------|
| `tensor` | — | Input tensor of any order |
| `rank` | — | Number of CP components |
| `max_iter` | `100` | Maximum ALS iterations |
| `tol` | `1e-6` | Relative reconstruction-error convergence threshold |
| `nonneg` | `False` | Enforce non-negative factors via multiplicative updates |
| `factors` | `None` | Warm-start factor matrices; list of `(I_k, R)` tensors, one per mode — skips random initialization |

```python
# Standard CP decomposition
factors, weights = trntensor.cp_decompose(T, rank=8)
T_approx = trntensor.cp_reconstruct(factors, weights)

# Non-negative CP (e.g. for density-matrix elements that must stay ≥ 0)
factors_nn, weights_nn = trntensor.cp_decompose(T, rank=8, nonneg=True)

# Warm-start from a previous run (e.g. after increasing rank by one)
factors_new, weights_new = trntensor.cp_decompose(
    T, rank=9, factors=prev_factors + [torch.randn(I, 1)]
)

# nonneg and warm-start compose
factors_ws, weights_ws = trntensor.cp_decompose(
    T, rank=8, nonneg=True, factors=seed_factors
)
```

### `cp_reconstruct(factors, weights) -> torch.Tensor`

Reconstruct the full tensor from CP factors and weights.

---

## Tucker (HOSVD)

### `tucker_decompose(tensor, ranks) -> (core, factors)`

Higher-order SVD: `T ≈ G ×_1 U_1 ×_2 U_2 ...`. Each factor matrix `U_k`
is orthonormal by construction.

```python
core, factors = trntensor.tucker_decompose(T, ranks=(4, 4, 4))
T_approx = trntensor.tucker_reconstruct(core, factors)
```

### `tucker_reconstruct(core, factors) -> torch.Tensor`

Reconstruct from a Tucker core and factor matrices.

---

## Tensor Train (TT-SVD)

### `tt_decompose(tensor, max_rank) -> list[torch.Tensor]`

Decompose a d-dimensional tensor into a chain of d cores via TT-SVD
(Oseledets 2011). Bond dimensions are capped at `max_rank`.

Each core has shape `(r_{k-1}, n_k, r_k)` where:
- `r_0 = r_d = 1` (boundary ranks)
- `n_k` is the size of mode `k`
- `r_k ≤ max_rank`

```
T[i_1, i_2, ..., i_d] ≈ G_1[1, i_1, :] · G_2[:, i_2, :] · ... · G_d[:, i_d, 1]
```

Returns a list of `d` core tensors.

```python
cores = trntensor.tt_decompose(T, max_rank=16)
# cores[k].shape == (r_{k-1}, n_k, r_k)

T_approx = trntensor.tt_reconstruct(cores)
```

**Use case — DMRG-style compression.** For a 6-mode tensor with modes
of size 20 each, full storage is `20^6 ≈ 64M` floats. At `max_rank=10`
the TT representation needs `6 · 10 · 20 · 10 = 12k` floats — a 5000×
reduction, at the cost of some truncation error.

### `tt_reconstruct(cores) -> torch.Tensor`

Contract the TT core chain back to the full tensor by successive
`tensordot` over the bond dimensions.

```python
T_approx = trntensor.tt_reconstruct(cores)
# T_approx.shape == (n_1, n_2, ..., n_d)
```
