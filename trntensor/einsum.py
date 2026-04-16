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

from .plan import ContractionPlan, _parse_subscripts, plan_contraction

# Mapping of user-friendly dtype strings to torch.dtype
_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "f32": torch.float32,
    "float32": torch.float32,
    "f64": torch.float64,
    "float64": torch.float64,
}


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Resolve a dtype argument to a ``torch.dtype`` or ``None``."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPE_MAP[dtype.lower()]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; valid strings: {sorted(_DTYPE_MAP)}") from None


def einsum(
    subscripts: str,
    *operands: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    out: torch.Tensor | None = None,
    dtype: str | torch.dtype | None = None,
) -> torch.Tensor:
    """Einstein summation with contraction planning.

    Supports the same subscript notation as torch.einsum and numpy.einsum.

    Args:
        subscripts: Einstein summation subscript string, e.g. ``"ij,jk->ik"``.
        *operands: Input tensors.
        alpha: Scalar multiplier applied to the contraction result (default 1.0).
        beta: Scalar multiplier applied to ``out`` before accumulation (default 0.0).
        out: Optional accumulation tensor. When provided the return value is
            ``alpha * contract(operands) + beta * out``. Must have the same
            shape as the contraction result.
        dtype: Optional compute dtype. When set, all operands are cast to this
            dtype before contracting and the result is returned in that dtype.
            Accepts ``torch.dtype`` instances or strings: ``"bf16"``,
            ``"bfloat16"``, ``"fp16"``, ``"float16"``, ``"f32"``, ``"float32"``.
            Matches Neuron SDK autocast recommendations for Trainium.

    Examples:
        # Matrix multiply
        einsum("ij,jk->ik", A, B)

        # Force bf16 compute (e.g. to hit NKI bf16 matmul kernel)
        einsum("ij,jk->ik", A, B, dtype="bf16")

        # Scaled GEMM: 2*A@B + 0.5*C  (cuTENSOR-style alpha/beta)
        einsum("ij,jk->ik", A, B, alpha=2.0, beta=0.5, out=C)

        # DF-MP2 energy contraction
        einsum("iap,jbp->ijab", B, B)
    """
    target = _resolve_dtype(dtype)
    if target is not None:
        operands = tuple(op.to(target) for op in operands)
    plan = plan_contraction(subscripts, *operands)
    result = _execute_contraction(subscripts, operands, plan)
    if alpha != 1.0:
        result = result.mul(alpha)
    if out is not None:
        result = result.add(out, alpha=beta)
    return result


def _execute_contraction(subscripts: str, operands: tuple, plan: ContractionPlan) -> torch.Tensor:
    """Execute contraction according to plan."""
    if plan.strategy == "torch":
        return torch.einsum(subscripts, *operands)
    elif plan.strategy == "bmm":
        return _execute_bmm(subscripts, operands, plan)
    elif plan.strategy == "matmul":
        return _execute_matmul(subscripts, operands, plan)
    elif plan.strategy == "path":
        return _execute_path(subscripts, operands, plan)
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


def _execute_path(subscripts: str, operands: tuple, plan: ContractionPlan) -> torch.Tensor:
    """Execute a multi-operand contraction in greedy path order.

    Walks ``plan.contraction_path`` step by step. At each step ``(i, j)``,
    builds the binary subscript for contracting the two current operands,
    calls ``einsum()`` recursively (so each binary step gets its own
    backend routing), then replaces the pair with the result. The last
    step uses the original output subscript to produce the final tensor.
    """
    _, output_str = _parse_subscripts(subscripts)
    input_str, _ = _parse_subscripts(subscripts)
    inputs = input_str.split(",")
    ops = list(operands)

    path = plan.contraction_path
    for step_idx, (i, j) in enumerate(path):
        is_last = step_idx == len(path) - 1
        if is_last:
            inter_str = output_str
        else:
            # Compute indices that must survive this step
            surviving = set(output_str)
            for k, inp in enumerate(inputs):
                if k != i and k != j:
                    surviving |= set(inp)
            pair_union = set(inputs[i]) | set(inputs[j])
            contracted = pair_union - surviving
            # Ordered dedup: free dims from inputs[i] then inputs[j], no repeats
            seen: set[str] = set()
            inter_parts: list[str] = []
            for c in list(inputs[i]) + list(inputs[j]):
                if c not in contracted and c not in seen:
                    seen.add(c)
                    inter_parts.append(c)
            inter_str = "".join(inter_parts)

        sub = f"{inputs[i]},{inputs[j]}->{inter_str}"
        result = torch.einsum(sub, ops[i], ops[j])

        ops = [op for k, op in enumerate(ops) if k not in (i, j)] + [result]
        inputs = [inp for k, inp in enumerate(inputs) if k not in (i, j)] + [inter_str]

    return ops[0]


def multi_einsum(*contractions: tuple) -> list[torch.Tensor]:
    """Execute multiple contractions, fusing where possible.

    Each contraction is (subscripts, *operands).
    Returns list of results.

    Shared operands (same tensor object appearing in more than one
    contraction) are pre-pinned to XLA once when NKI is available,
    avoiding repeated host→device transfers.

    Useful for DF-MP2 where many independent contractions share operands:
        results = multi_einsum(
            ("iap,jbp->ijab", B, B),
            ("ibp,jap->ijab", B, B),  # Exchange term
        )
    """
    from .nki.dispatch import HAS_NKI, _use_nki

    # Count appearances of each tensor object across all contractions
    id_to_tensor: dict[int, torch.Tensor] = {}
    id_count: dict[int, int] = {}
    for c in contractions:
        for op in c[1:]:
            tid = id(op)
            id_to_tensor[tid] = op
            id_count[tid] = id_count.get(tid, 0) + 1
    shared_ids = {tid for tid, n in id_count.items() if n > 1}

    # Pre-pin shared tensors to XLA once (only when NKI dispatch is active)
    xla_map: dict[int, torch.Tensor] = {}
    if HAS_NKI and _use_nki() and shared_ids:
        from .nki.dispatch import to_xla

        xla_map = {tid: to_xla(id_to_tensor[tid]) for tid in shared_ids}

    results = []
    for c in contractions:
        subscripts = c[0]
        ops = tuple(xla_map.get(id(op), op) for op in c[1:])
        result = einsum(subscripts, *ops)
        if hasattr(result, "device") and result.device.type == "xla":
            from .nki.dispatch import from_xla

            result = from_xla(result)
        results.append(result)
    return results
