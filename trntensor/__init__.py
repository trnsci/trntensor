"""
trntensor — Tensor contractions for AWS Trainium via NKI.

Einstein summation, contraction planning, and tensor decompositions
for scientific computing. Part of the trnsci scientific computing suite.
"""

__version__ = "0.1.2"

from .decompose import cp_decompose, cp_reconstruct, tucker_decompose, tucker_reconstruct
from .einsum import einsum, multi_einsum
from .nki import HAS_NKI, get_backend, set_backend
from .plan import ContractionPlan, estimate_flops, plan_contraction
from .quantum import ao_to_mo_transform, mp2_energy

__all__ = [
    "einsum",
    "multi_einsum",
    "ContractionPlan",
    "plan_contraction",
    "estimate_flops",
    "cp_decompose",
    "cp_reconstruct",
    "tucker_decompose",
    "tucker_reconstruct",
    "HAS_NKI",
    "set_backend",
    "get_backend",
    "mp2_energy",
    "ao_to_mo_transform",
]
