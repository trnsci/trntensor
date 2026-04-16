"""
Tensor decompositions for Trainium.

CP (CANDECOMP/PARAFAC), Tucker, and Tensor Train (TT) decompositions for
compressing high-order tensors. Used in quantum chemistry for:
- Tensor hypercontraction (THC) of two-electron integrals
- Reduced-rank approximation of the DF coefficient tensor B_ia^P
- Low-rank factorization of response tensors
- DMRG-style high-dimensional compression (TT)

All inner operations are matmuls → map to Tensor Engine via trnblas.
"""

from __future__ import annotations

import torch


def cp_decompose(
    tensor: torch.Tensor,
    rank: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    nonneg: bool = False,
    factors: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """CP decomposition via alternating least squares (ALS) or multiplicative updates.

    Approximates tensor T ≈ Σ_r λ_r * a_r ⊗ b_r ⊗ c_r ⊗ ...

    Args:
        tensor: Input tensor of arbitrary order.
        rank: Number of components.
        max_iter: Maximum iterations.
        tol: Convergence tolerance (relative reconstruction error).
        nonneg: If True, use multiplicative updates to enforce non-negative factors.
            Initialization uses ``torch.rand`` (positive) unless ``factors`` is given.
        factors: Optional warm-start factor matrices, one per mode, each of shape
            ``(tensor.shape[k], rank)``. When provided the random initialization
            step is skipped. Works with or without ``nonneg``.

    Returns:
        factors: List of factor matrices [(I_0, R), (I_1, R), ...]
        weights: Component weights (R,). For ``nonneg=True`` weights are all 1.
    """
    ndim = tensor.dim()
    shape = tensor.shape

    if factors is not None:
        # Warm-start: validate and clone so we don't mutate caller's tensors
        if len(factors) != ndim:
            raise ValueError(
                f"warm-start factors has {len(factors)} matrices but tensor has {ndim} modes"
            )
        for k, f in enumerate(factors):
            if f.shape != (shape[k], rank):
                raise ValueError(
                    f"warm-start factors[{k}] has shape {tuple(f.shape)}, "
                    f"expected ({shape[k]}, {rank})"
                )
        factors = [f.clone() for f in factors]
        if nonneg:
            factors = [f.clamp(min=1e-10) for f in factors]
    elif nonneg:
        factors = [torch.rand(s, rank).clamp(min=1e-10) for s in shape]
    else:
        factors = [torch.randn(s, rank) for s in shape]
        for k in range(ndim):
            norms = torch.linalg.norm(factors[k], dim=0, keepdim=True)
            factors[k] = factors[k] / (norms + 1e-10)

    weights = torch.ones(rank)

    for _ in range(max_iter):
        for mode in range(ndim):
            V = _khatri_rao_except(factors, mode)
            T_mode = _unfold(tensor, mode)
            VtV = V.T @ V

            if nonneg:
                # Multiplicative update: ensures non-negativity
                numer = (T_mode @ V).clamp(min=0)
                denom = (factors[mode] @ VtV).clamp(min=1e-10)
                factors[mode] = (factors[mode] * numer / denom).clamp(min=1e-10)
            else:
                # ALS: solve normal equations
                rhs = T_mode @ V
                try:
                    factors[mode] = torch.linalg.solve(VtV, rhs.T).T
                except RuntimeError:
                    factors[mode] = rhs @ torch.linalg.pinv(VtV)
                # Extract norms as weights
                norms = torch.linalg.norm(factors[mode], dim=0)
                weights = norms
                factors[mode] = factors[mode] / (norms.unsqueeze(0) + 1e-10)

        # Check convergence
        reconstructed = cp_reconstruct(factors, weights)
        error = torch.linalg.norm(tensor - reconstructed) / torch.linalg.norm(tensor)
        if error.item() < tol:
            break

    return factors, weights


def cp_reconstruct(factors: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Reconstruct tensor from CP factors: T = Σ_r w_r * a_r ⊗ b_r ⊗ ..."""
    rank = weights.shape[0]
    ndim = len(factors)
    shape = tuple(f.shape[0] for f in factors)

    result = torch.zeros(shape)
    for r in range(rank):
        component = weights[r]
        outer = factors[0][:, r]
        for k in range(1, ndim):
            outer = torch.outer(outer.reshape(-1), factors[k][:, r]).reshape(
                *[factors[i].shape[0] for i in range(k + 1)]
            )
        result = result + component * outer

    return result


def tucker_decompose(
    tensor: torch.Tensor,
    ranks: tuple[int, ...],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Tucker decomposition via higher-order SVD (HOSVD).

    T ≈ G ×_1 U_1 ×_2 U_2 ×_3 U_3 ...

    Args:
        tensor: Input tensor
        ranks: Tuple of ranks for each mode

    Returns:
        core: Core tensor of shape ranks
        factors: List of factor matrices [(I_k, R_k) for each mode]
    """
    ndim = tensor.dim()
    assert len(ranks) == ndim

    factors = []
    for mode in range(ndim):
        T_mode = _unfold(tensor, mode)
        U, S, Vh = torch.linalg.svd(T_mode, full_matrices=False)
        factors.append(U[:, : ranks[mode]])

    # Core tensor: G = T ×_1 U_1^T ×_2 U_2^T ×_3 U_3^T ...
    core = tensor.clone()
    for mode in range(ndim):
        core = _mode_product(core, factors[mode].T, mode)

    return core, factors


def tucker_reconstruct(core: torch.Tensor, factors: list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct tensor from Tucker decomposition."""
    result = core.clone()
    for mode in range(len(factors)):
        result = _mode_product(result, factors[mode], mode)
    return result


def tt_decompose(
    tensor: torch.Tensor,
    max_rank: int,
) -> list[torch.Tensor]:
    """Tensor Train (TT) decomposition via TT-SVD (Oseledets 2011).

    Decomposes a d-dimensional tensor into a chain of 3-tensors (cores):

        T[i_1, i_2, ..., i_d] ≈ G_1[:, i_1, :] @ G_2[:, i_2, :] @ ... @ G_d[:, i_d, :]

    where G_k has shape ``(r_{k-1}, n_k, r_k)`` and boundary bond dimensions
    ``r_0 = r_d = 1``.  Bond dimensions are capped at ``max_rank``.

    Args:
        tensor: Input tensor of shape ``(n_1, n_2, ..., n_d)``.
        max_rank: Maximum bond dimension (rank) between adjacent cores.

    Returns:
        List of ``d`` core tensors, each of shape ``(r_{k-1}, n_k, r_k)``.
    """
    shape = tensor.shape
    ndim = tensor.dim()
    if ndim < 2:
        raise ValueError(f"tt_decompose requires ndim ≥ 2, got {ndim}")

    cores: list[torch.Tensor] = []
    C = tensor.reshape(shape[0], -1)  # (n_1, n_2 * ... * n_d)
    r_prev = 1

    for k in range(ndim - 1):
        n_k = shape[k]
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        r_k = min(max_rank, U.shape[1])
        # Core G_k: shape (r_{k-1}, n_k, r_k)
        cores.append(U[:, :r_k].reshape(r_prev, n_k, r_k))
        # Remainder for next step
        remainder = torch.diag(S[:r_k]) @ Vh[:r_k, :]
        # Reshape to (r_k * n_{k+1}, remaining_dims)
        remaining_size = 1
        for s in shape[k + 1 :]:
            remaining_size *= s
        n_next = shape[k + 1]
        C = remainder.reshape(r_k * n_next, remaining_size // n_next)
        r_prev = r_k

    # Last core: shape (r_{d-1}, n_d, 1)
    cores.append(C.reshape(r_prev, shape[-1], 1))
    return cores


def tt_reconstruct(cores: list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct a tensor from its Tensor Train cores.

    Contracts the core chain left to right:
    ``result = G_1 @ G_2 @ ... @ G_d``

    Args:
        cores: List of core tensors, each of shape ``(r_{k-1}, n_k, r_k)``.

    Returns:
        Reconstructed tensor of shape ``(n_1, n_2, ..., n_d)``.
    """
    # result starts as cores[0][0, :, :] — shape (n_1, r_1)
    result = cores[0][0, :, :]
    for core in cores[1:]:
        # result: (..., r_{k-1}), core: (r_{k-1}, n_k, r_k)
        result = torch.tensordot(result, core, dims=1)
    # Squeeze final bond dimension (r_d = 1)
    return result[..., 0]


# --- Tensor algebra helpers ---


def _unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """Mode-n unfolding (matricization) of a tensor."""
    return tensor.moveaxis(mode, 0).reshape(tensor.shape[mode], -1)


def _mode_product(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    """Mode-n product: T ×_n M.

    Multiplies matrix M along mode n of tensor T.
    """
    T_unfolded = _unfold(tensor, mode)
    result_unfolded = matrix @ T_unfolded
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    # Refold
    result = result_unfolded.reshape(
        new_shape[mode], *[new_shape[i] for i in range(len(new_shape)) if i != mode]
    )
    # Move mode axis back
    result = result.moveaxis(0, mode)
    return result


def _khatri_rao_except(factors: list[torch.Tensor], skip: int) -> torch.Tensor:
    """Khatri-Rao product of all factor matrices except index `skip`.

    Column-wise Kronecker product.
    """
    indices = [i for i in range(len(factors)) if i != skip]
    result = factors[indices[0]]
    for i in indices[1:]:
        # Khatri-Rao: column-wise Kronecker
        R = result.shape[1]
        new_rows = result.shape[0] * factors[i].shape[0]
        kr = torch.zeros(new_rows, R)
        for r in range(R):
            kr[:, r] = torch.kron(result[:, r], factors[i][:, r])
        result = kr
    return result
