"""
NKI dispatch for tensor contractions.

The primary NKI target is fused multi-index contractions that avoid
materializing intermediates. For example, the DF-MP2 energy contraction
    E = Σ_{ijab} B_ia^P B_jb^P / Δ_{ijab}
can be tiled across (i,j) pairs with B slices loaded once to SBUF.

On the Tensor Engine, each (a,b) block is a matmul: B[i] @ B[j]^T.
Fusing the denominator division into the PSUM accumulation avoids
a separate element-wise pass over the output.
"""

from __future__ import annotations

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

_backend = "auto"

def set_backend(backend: str):
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    _backend = backend

def get_backend() -> str:
    return _backend

def _use_nki() -> bool:
    if _backend == "nki": return True
    if _backend == "pytorch": return False
    return HAS_NKI
