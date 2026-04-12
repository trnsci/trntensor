"""
Contraction planner for tensor operations.

Analyzes einsum subscripts to select the best execution strategy:
- Direct matmul for 2-operand contractions over a single index
- Batched matmul (bmm) for batched contractions
- torch.einsum fallback for complex patterns
- (Future) NKI-optimized paths for specific patterns

Also plans multi-contraction fusion for DF-MP2-style workloads
where many contractions share operands.
"""

from __future__ import annotations

import re
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContractionPlan:
    """Execution plan for a tensor contraction."""
    subscripts: str
    strategy: str  # "matmul", "bmm", "torch", "nki"
    transA: bool = False
    transB: bool = False
    contraction_indices: list[str] = field(default_factory=list)
    batch_indices: list[str] = field(default_factory=list)
    output_indices: list[str] = field(default_factory=list)
    estimated_flops: int = 0


def plan_contraction(subscripts: str, *operands: torch.Tensor) -> ContractionPlan:
    """Analyze contraction and select execution strategy."""
    input_str, output_str = _parse_subscripts(subscripts)
    input_indices = input_str.split(",")

    if len(operands) == 2:
        return _plan_binary(subscripts, input_indices, output_str, operands)
    else:
        return ContractionPlan(subscripts=subscripts, strategy="torch")


def _parse_subscripts(subscripts: str) -> tuple[str, str]:
    """Parse 'ij,jk->ik' into ('ij,jk', 'ik')."""
    if "->" in subscripts:
        input_str, output_str = subscripts.split("->")
    else:
        input_str = subscripts
        # Implicit output: all indices that appear exactly once
        all_indices = re.findall(r'[a-zA-Z]', input_str.replace(",", ""))
        from collections import Counter
        counts = Counter(all_indices)
        output_str = "".join(sorted(c for c, n in counts.items() if n == 1))
    return input_str, output_str


def _plan_binary(
    subscripts: str,
    input_indices: list[str],
    output_str: str,
    operands: tuple,
) -> ContractionPlan:
    """Plan a 2-operand contraction."""
    idx_a = list(input_indices[0])
    idx_b = list(input_indices[1])
    idx_out = list(output_str)

    set_a = set(idx_a)
    set_b = set(idx_b)
    set_out = set(idx_out)

    # Contracted indices: appear in inputs but not output
    contracted = (set_a & set_b) - set_out
    # Batch indices: appear in both inputs and output
    batch = set_a & set_b & set_out
    # Free indices: appear in one input and output
    free_a = (set_a - set_b) & set_out
    free_b = (set_b - set_a) & set_out

    A, B = operands

    # Simple matmul: ij,jk->ik or ji,jk->ik etc.
    if len(contracted) == 1 and len(batch) == 0 and A.dim() == 2 and B.dim() == 2:
        c_idx = list(contracted)[0]
        transA = idx_a.index(c_idx) == 0  # Contracted index is first → need transpose
        transB = idx_b.index(c_idx) == 1  # Contracted index is second → need transpose
        return ContractionPlan(
            subscripts=subscripts,
            strategy="matmul",
            transA=transA,
            transB=transB,
            contraction_indices=list(contracted),
            output_indices=idx_out,
        )

    # Batched matmul: bij,bjk->bik
    if len(contracted) == 1 and len(batch) == 1 and A.dim() == 3 and B.dim() == 3:
        return ContractionPlan(
            subscripts=subscripts,
            strategy="bmm",
            contraction_indices=list(contracted),
            batch_indices=list(batch),
            output_indices=idx_out,
        )

    # Fallback to torch.einsum
    return ContractionPlan(
        subscripts=subscripts,
        strategy="torch",
        contraction_indices=list(contracted),
        batch_indices=list(batch),
        output_indices=idx_out,
    )


def estimate_flops(subscripts: str, *operands: torch.Tensor) -> int:
    """Estimate FLOPs for a contraction (multiply-add pairs)."""
    input_str, output_str = _parse_subscripts(subscripts)
    all_indices = set(re.findall(r'[a-zA-Z]', input_str))

    # Product of all dimension sizes
    index_sizes = {}
    for op_str, op in zip(input_str.split(","), operands):
        for idx, size in zip(op_str, op.shape):
            index_sizes[idx] = size

    flops = 1
    for idx in all_indices:
        flops *= index_sizes.get(idx, 1)

    return flops
