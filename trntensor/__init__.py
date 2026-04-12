"""
trntensor — Tensor contractions for AWS Trainium via NKI.

Einstein summation, contraction planning, and tensor decompositions
for scientific computing. Part of the trn-* suite by Playground Logic.
"""

__version__ = "0.1.0"

from .einsum import einsum, multi_einsum
from .plan import ContractionPlan, plan_contraction, estimate_flops
from .decompose import (cp_decompose, cp_reconstruct,
                        tucker_decompose, tucker_reconstruct)
from .nki import HAS_NKI, set_backend, get_backend

__all__ = [
    "einsum", "multi_einsum",
    "ContractionPlan", "plan_contraction", "estimate_flops",
    "cp_decompose", "cp_reconstruct", "tucker_decompose", "tucker_reconstruct",
    "HAS_NKI", "set_backend", "get_backend",
]
