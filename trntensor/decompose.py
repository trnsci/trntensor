"""
Tensor decompositions for Trainium.

CP (CANDECOMP/PARAFAC) and Tucker decompositions for compressing
high-order tensors. Used in quantum chemistry for:
- Tensor hypercontraction (THC) of two-electron integrals
- Reduced-rank approximation of the DF coefficient tensor B_ia^P
- Low-rank factorization of response tensors

All inner operations are matmuls → map to Tensor Engine via trnblas.
"""

from __future__ import annotations

import torch


def cp_decompose(
    tensor: torch.Tensor,
    rank: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """CP decomposition via alternating least squares (ALS).

    Approximates tensor T ≈ Σ_r λ_r * a_r ⊗ b_r ⊗ c_r ⊗ ...

    Args:
        tensor: Input tensor of arbitrary order
        rank: Number of components
        max_iter: Maximum ALS iterations
        tol: Convergence tolerance (relative reconstruction error)

    Returns:
        factors: List of factor matrices, one per mode [(I_0, R), (I_1, R), ...]
        weights: Component weights (R,)
    """
    ndim = tensor.dim()
    shape = tensor.shape

    # Initialize factors randomly
    factors = [torch.randn(s, rank) for s in shape]

    # Normalize
    for k in range(ndim):
        norms = torch.linalg.norm(factors[k], dim=0, keepdim=True)
        factors[k] = factors[k] / (norms + 1e-10)

    weights = torch.ones(rank)

    for _ in range(max_iter):
        for mode in range(ndim):
            # Compute the Khatri-Rao product of all factors except current mode
            V = _khatri_rao_except(factors, mode)

            # Unfold tensor along current mode
            T_mode = _unfold(tensor, mode)

            # Solve least squares: factors[mode] = T_mode @ V @ (V^T V)^{-1}
            VtV = V.T @ V
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
