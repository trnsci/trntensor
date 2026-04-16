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
from dataclasses import dataclass, field

import torch


@dataclass
class ContractionPlan:
    """Execution plan for a tensor contraction."""

    subscripts: str
    strategy: str  # algorithm: "matmul" | "bmm" | "torch" | "path"
    backend: str = "pytorch"  # executor: "nki" | "pytorch"
    transA: bool = False
    transB: bool = False
    contraction_indices: list[str] = field(default_factory=list)
    batch_indices: list[str] = field(default_factory=list)
    output_indices: list[str] = field(default_factory=list)
    estimated_flops: int = 0
    contraction_path: list[tuple[int, int]] = field(default_factory=list)
    # Greedy-optimal order for 3+ operand einsums. Each (i, j) pair refers to
    # indices in the CURRENT operand list at that step (opt_einsum convention):
    # contract ops[i] and ops[j], remove both, append the result.


def _backend_for(strategy: str, operands: tuple) -> str:
    """Resolve which executor will run a given strategy.

    Returns ``"nki"`` only when the strategy maps to a NKI kernel
    (``matmul`` or ``bmm``), ``neuronxcc`` is importable, and the
    contraction has enough FLOPs to clear the dispatch-overhead
    threshold. Otherwise returns ``"pytorch"`` — matching what
    ``nki_matmul`` / ``nki_batched_matmul`` will do at runtime.
    """
    if strategy not in ("matmul", "bmm"):
        return "pytorch"
    from .nki.dispatch import _MIN_NKI_FLOPS, HAS_NKI

    if not HAS_NKI:
        return "pytorch"
    if strategy == "matmul":
        M, K = operands[0].shape
        _, N = operands[1].shape
        flops = M * K * N
    else:  # bmm
        Bsz, M, K = operands[0].shape
        _, _, N = operands[1].shape
        flops = Bsz * M * K * N
    return "nki" if flops >= _MIN_NKI_FLOPS else "pytorch"


def _validate_subscripts(subscripts: str, operands: tuple[torch.Tensor, ...]) -> None:
    """Validate subscripts and operand shapes, raising ValueError with specific messages.

    Checks performed in order:
    1. Subscript string characters are valid (a-zA-Z, commas, optional '->').
    2. Number of comma-separated input terms matches ``len(operands)``.
    3. Each term's length matches the corresponding operand's ndim.
    4. Every index character is consistently sized across all operands that share it.
    """
    # 1 — character-set check
    if not re.fullmatch(r"[a-zA-Z,]+(->[a-zA-Z]*)?", subscripts):
        raise ValueError(
            f"einsum subscript {subscripts!r} contains invalid characters; "
            "only a-zA-Z, commas, and '->' are allowed"
        )

    input_str, _ = _parse_subscripts(subscripts)
    terms = input_str.split(",")

    # 2 — operand count
    if len(terms) != len(operands):
        raise ValueError(
            f"einsum subscript {subscripts!r} has {len(terms)} operand term(s) "
            f"but {len(operands)} operand(s) were given"
        )

    # 3 — rank check; 4 — consistent sizes
    index_sizes: dict[str, tuple[int, int]] = {}  # char → (size, operand_index)
    for op_idx, (term, op) in enumerate(zip(terms, operands, strict=True)):
        if len(term) != op.ndim:
            raise ValueError(
                f"einsum operand {op_idx} has shape {tuple(op.shape)} (ndim={op.ndim}) "
                f"but subscript term {term!r} has {len(term)} indices"
            )
        for char, size in zip(term, op.shape, strict=True):
            size = int(size)
            if char in index_sizes:
                prev_size, prev_op = index_sizes[char]
                if prev_size != size:
                    raise ValueError(
                        f"einsum index {char!r} is size {prev_size} in operand {prev_op} "
                        f"but size {size} in operand {op_idx}"
                    )
            else:
                index_sizes[char] = (size, op_idx)


# Module-level cache: (subscripts, shape_signature) → ContractionPlan
_PLAN_CACHE: dict[tuple, ContractionPlan] = {}


def _shape_key(subscripts: str, operands: tuple[torch.Tensor, ...]) -> tuple:
    """Build a hashable cache key from subscripts and operand shapes."""
    return (subscripts, tuple(tuple(op.shape) for op in operands))


def clear_plan_cache() -> None:
    """Discard all cached contraction plans."""
    _PLAN_CACHE.clear()


def plan_cache_info() -> dict[str, int]:
    """Return cache statistics.

    Returns:
        dict with key ``"size"`` — the number of cached plans.
    """
    return {"size": len(_PLAN_CACHE)}


def plan_contraction(subscripts: str, *operands: torch.Tensor) -> ContractionPlan:
    """Analyze contraction and select execution strategy.

    Results are cached by ``(subscripts, operand shapes)``. Repeated calls
    with the same subscript and shapes skip replanning entirely. Call
    ``clear_plan_cache()`` to invalidate the cache (e.g. after a backend change).
    """
    _validate_subscripts(subscripts, operands)
    key = _shape_key(subscripts, operands)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]

    input_str, output_str = _parse_subscripts(subscripts)
    input_indices = input_str.split(",")

    if len(operands) == 2:
        plan = _plan_binary(subscripts, input_indices, output_str, operands)
    elif len(operands) >= 3:
        size_map: dict[str, int] = {}
        for op_str, op in zip(input_indices, operands, strict=False):
            for idx, sz in zip(op_str, op.shape, strict=False):
                size_map[idx] = int(sz)
        path = _greedy_path_search(input_indices, output_str, size_map)
        plan = ContractionPlan(
            subscripts=subscripts,
            strategy="path",
            contraction_path=path,
        )
    else:
        plan = ContractionPlan(subscripts=subscripts, strategy="torch")
    plan.backend = _backend_for(plan.strategy, operands)
    _PLAN_CACHE[key] = plan
    return plan


def _greedy_path_search(
    input_list: list[str],
    output_str: str,
    size_map: dict[str, int],
) -> list[tuple[int, int]]:
    """Greedy contraction-path search over 3+ operands.

    At each step, considers every pair of current operands and contracts
    the pair with the minimum multiply-add count (product of all index
    sizes in the pair's union, contracted and output alike). Returns a
    list of ``(i, j)`` pairs in the opt_einsum convention: indices refer
    to the CURRENT list at that step; after each step the two operands
    are removed and the intermediate is appended.

    The ``size_map`` (built once from the original operand shapes) is
    sufficient throughout: intermediate indices are always a subset of the
    original index set, so their sizes are already known.
    """
    inputs = list(input_list)
    path: list[tuple[int, int]] = []

    while len(inputs) > 2:
        best_pair: tuple[int, int] = (0, 1)
        best_cost = float("inf")
        best_inter: set[str] = set()

        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                # Indices that must survive this step: output ∪ all other inputs
                surviving = set(output_str)
                for k, inp in enumerate(inputs):
                    if k != i and k != j:
                        surviving |= set(inp)

                pair_union = set(inputs[i]) | set(inputs[j])
                contracted = pair_union - surviving
                intermediate = pair_union - contracted

                # FLOPs ≈ product of all index sizes in the pair union
                cost = 1
                for idx in pair_union:
                    cost *= size_map.get(idx, 1)

                if cost < best_cost:
                    best_cost = cost
                    best_pair = (i, j)
                    best_inter = intermediate

        i, j = best_pair
        path.append((i, j))
        # Replace pair with intermediate; use sorted order for stable subscript strings
        new_input = "".join(sorted(best_inter))
        inputs = [inp for k, inp in enumerate(inputs) if k not in (i, j)] + [new_input]

    path.append((0, 1))
    return path


def _parse_subscripts(subscripts: str) -> tuple[str, str]:
    """Parse 'ij,jk->ik' into ('ij,jk', 'ik')."""
    if "->" in subscripts:
        input_str, output_str = subscripts.split("->")
    else:
        input_str = subscripts
        # Implicit output: all indices that appear exactly once
        all_indices = re.findall(r"[a-zA-Z]", input_str.replace(",", ""))
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
    all_indices = set(re.findall(r"[a-zA-Z]", input_str))

    # Product of all dimension sizes
    index_sizes = {}
    for op_str, op in zip(input_str.split(","), operands, strict=False):
        for idx, size in zip(op_str, op.shape, strict=False):
            index_sizes[idx] = size

    flops = 1
    for idx in all_indices:
        flops *= index_sizes.get(idx, 1)

    return flops
