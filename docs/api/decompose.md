# Decompositions

CP (CANDECOMP/PARAFAC) and Tucker (HOSVD) decompositions for
compressing high-order tensors. Used in quantum chemistry for tensor
hypercontraction (THC) and low-rank factorization of DF-coefficient
tensors.

## CP (PARAFAC)

### `cp_decompose(tensor, rank, *, max_iter=100, tol=1e-6) -> (factors, weights)`

Decompose a tensor as `T ≈ Σ_r w_r * a_r ⊗ b_r ⊗ c_r ⊗ ...` via
alternating least squares.

- `factors`: list of factor matrices, one per mode with shape `(I_k, R)`
- `weights`: `(R,)` tensor of component weights

```python
factors, weights = trntensor.cp_decompose(T, rank=8, max_iter=20)
T_approx = trntensor.cp_reconstruct(factors, weights)
```

### `cp_reconstruct(factors, weights) -> torch.Tensor`

Reconstruct the full tensor from CP factors.

## Tucker (HOSVD)

### `tucker_decompose(tensor, ranks) -> (core, factors)`

Higher-order SVD: `T ≈ G ×_1 U_1 ×_2 U_2 ...` Each factor matrix `U_k`
is orthonormal by construction.

```python
core, factors = trntensor.tucker_decompose(T, ranks=(4, 4, 4))
T_approx = trntensor.tucker_reconstruct(core, factors)
```

### `tucker_reconstruct(core, factors) -> torch.Tensor`

Reconstruct from a Tucker decomposition.
