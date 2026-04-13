"""
Tensor contractions via einsum for Trainium.

The core operation: generalized tensor contraction expressed in Einstein
summation notation. Decomposes arbitrary contractions into sequences of
matmuls that map to the Tensor Engine.

For DF-MP2, the natural expression is:
    T_ijab = einsum("iap,jbp->ijab", B, B)

rather than the GEMM decomposition:
    for i in range(nocc):
        for j in range(nocc):
            T = B[i] @ B[j].T

The einsum path can fuse the loops and avoid materializing intermediates,
which is where cuTENSOR beats cuBLAS for tensor workloads.
"""

from __future__ import annotations

import torch

from .plan import ContractionPlan, plan_contraction


def einsum(subscripts: str, *operands: torch.Tensor) -> torch.Tensor:
    """Einstein summation with contraction planning.

    Supports the same subscript notation as torch.einsum and numpy.einsum.

    Examples:
        # Matrix multiply
        einsum("ij,jk->ik", A, B)

        # Batched matrix multiply
        einsum("bij,bjk->bik", A, B)

        # Trace
        einsum("ii->", A)

        # Outer product
        einsum("i,j->ij", x, y)

        # DF-MP2 energy contraction
        einsum("iap,jbp->ijab", B, B)

        # Tensor contraction (4-index transform)
        einsum("mi,mnP->inP", C, integrals)
    """
    plan = plan_contraction(subscripts, *operands)
    return _execute_contraction(subscripts, operands, plan)


def _execute_contraction(subscripts: str, operands: tuple, plan: ContractionPlan) -> torch.Tensor:
    """Execute contraction according to plan.

    For now, delegates to torch.einsum. The plan infrastructure
    enables NKI-optimized paths for specific contraction patterns.
    """
    if plan.strategy == "torch":
        return torch.einsum(subscripts, *operands)
    elif plan.strategy == "bmm":
        return _execute_bmm(subscripts, operands, plan)
    elif plan.strategy == "matmul":
        return _execute_matmul(subscripts, operands, plan)
    else:
        return torch.einsum(subscripts, *operands)


def _execute_matmul(subscripts: str, operands: tuple, plan: ContractionPlan) -> torch.Tensor:
    """Execute as a single matmul (2D contraction).

    Pre-transposes per the plan so the NKI kernel always sees the
    canonical ``A @ B`` form.
    """
    from .nki.dispatch import nki_matmul

    A, B = operands
    a = A.T if plan.transA else A
    b = B.T if plan.transB else B
    return nki_matmul(a, b)


def _execute_bmm(subscripts: str, operands: tuple, plan: ContractionPlan) -> torch.Tensor:
    """Execute as batched matmul."""
    from .nki.dispatch import nki_batched_matmul

    A, B = operands
    return nki_batched_matmul(A, B)


def multi_einsum(*contractions: tuple) -> list[torch.Tensor]:
    """Execute multiple contractions, fusing where possible.

    Each contraction is (subscripts, *operands).
    Returns list of results.

    Useful for DF-MP2 where many independent contractions share operands:
        results = multi_einsum(
            ("iap,jbp->ijab", B, B),
            ("ibp,jap->ijab", B, B),  # Exchange term
        )
    """
    results = []
    for contraction in contractions:
        subscripts = contraction[0]
        operands = contraction[1:]
        results.append(einsum(subscripts, *operands))
    return results
