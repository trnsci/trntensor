"""
trntensor — Tensor contractions for AWS Trainium via NKI.

Einstein summation, contraction planning, and tensor decompositions
for scientific computing. Part of the trnsci scientific computing suite.
"""

__version__ = "0.17.0"

from .decompose import (
    cp_decompose,
    cp_reconstruct,
    tt_decompose,
    tt_reconstruct,
    tucker_decompose,
    tucker_reconstruct,
)
from .einsum import einsum, multi_einsum
from .nki import HAS_NKI, get_backend, set_backend
from .nki.dispatch import from_device, from_xla, to_device, to_xla
from .parallel import ShardedTensor, gather, scatter
from .plan import (
    ContractionPlan,
    clear_plan_cache,
    estimate_flops,
    forward_error_bound,
    plan_cache_info,
    plan_contraction,
    select_precision_for_error,
)
from .quantum import ao_to_mo_transform, mp2_energy

__all__ = [
    "einsum",
    "multi_einsum",
    "ContractionPlan",
    "plan_contraction",
    "estimate_flops",
    "clear_plan_cache",
    "plan_cache_info",
    "select_precision_for_error",
    "forward_error_bound",
    "cp_decompose",
    "cp_reconstruct",
    "tucker_decompose",
    "tucker_reconstruct",
    "tt_decompose",
    "tt_reconstruct",
    "HAS_NKI",
    "set_backend",
    "get_backend",
    "to_device",
    "from_device",
    "to_xla",
    "from_xla",
    "mp2_energy",
    "ao_to_mo_transform",
    "ShardedTensor",
    "scatter",
    "gather",
]
