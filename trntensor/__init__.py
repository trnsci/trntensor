"""
trntensor — Tensor contractions for AWS Trainium via NKI.

Einstein summation, contraction planning, and tensor decompositions
for scientific computing. Part of the trnsci scientific computing suite.
"""

__version__ = "0.26.0"

from ._region import get_region_backend, set_region_backend
from ._region import region as compiled_region
from .decompose import (
    cp_decompose,
    cp_reconstruct,
    tt_decompose,
    tt_reconstruct,
    tucker_decompose,
    tucker_reconstruct,
)
from .edge import (
    EdgeChainSpec,
    EdgeMLP,
    edge_chain,
    edge_chain_spec,
    sh_coefficients,
    spherical_harmonics,
)
from .einsum import einsum, multi_einsum
from .equivariant import (
    ConvTPSpec,
    SymmetricContractionSpec,
    conv_tp,
    conv_tp_spec,
    symmetric_contraction,
    symmetric_contraction_spec,
    wigner_3j,
    wigner_D,
)
from .linear import EquivariantLinearSpec, equivariant_linear, equivariant_linear_spec
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
from .quantum import (
    ao_to_mo_transform,
    df_coulomb,
    df_exchange,
    mp2_energy,
    pack_tril,
    unpack_tril,
)
from .stream import (
    AuxStream,
    aux_chunk_width,
    mp2_energy_blocked,
    pack_density,
    split_aux,
    unpack_packed,
)

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
    "df_exchange",
    "df_coulomb",
    "pack_tril",
    "unpack_tril",
    "AuxStream",
    "aux_chunk_width",
    "split_aux",
    "pack_density",
    "unpack_packed",
    "mp2_energy_blocked",
    "conv_tp",
    "conv_tp_spec",
    "ConvTPSpec",
    "symmetric_contraction",
    "symmetric_contraction_spec",
    "SymmetricContractionSpec",
    "wigner_3j",
    "wigner_D",
    "spherical_harmonics",
    "sh_coefficients",
    "edge_chain",
    "edge_chain_spec",
    "EdgeChainSpec",
    "EdgeMLP",
    "equivariant_linear",
    "equivariant_linear_spec",
    "EquivariantLinearSpec",
    "set_region_backend",
    "compiled_region",
    "get_region_backend",
    "ShardedTensor",
    "scatter",
    "gather",
]
