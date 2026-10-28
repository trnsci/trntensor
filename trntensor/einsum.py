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
    precision: str = "fast",
    target_forward_error: float | None = None,
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
        target_forward_error: Optional relative forward error bound. When set,
            trntensor selects the cheapest precision mode whose theoretical
            error bound satisfies the target: ``"fast"`` → ``"sr"`` → ``"dd"``
            → ``"kahan"``. Cannot be combined with an explicit ``precision=``
            argument. See :func:`~trntensor.select_precision_for_error` for
            the exact error bounds used. The result is returned in the operand
            dtype, whose unit roundoff is a floor no mode can beat (bf16
            ≈ 3.9e-3, fp32 ≈ 6.0e-8); a target below it emits a one-time
            ``RuntimeWarning`` and runs ``"kahan"``.

    Note:
        On Trainium (NKI 0.7+) stochastic rounding — ``precision="sr"``,
        or ``"sr"`` chosen by ``target_forward_error`` — is a process-wide
        Neuron Runtime setting. Set ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``
        before the process starts. Without it, trntensor emits a one-time
        ``RuntimeWarning`` and the result is round-to-nearest. With it, every
        downcast in the process rounds stochastically, including ``"fast"``.

    Examples:
        # Matrix multiply
        einsum("ij,jk->ik", A, B)

        # Force bf16 compute (e.g. to hit NKI bf16 matmul kernel)
        einsum("ij,jk->ik", A, B, dtype="bf16")

        # Scaled GEMM: 2*A@B + 0.5*C  (cuTENSOR-style alpha/beta)
        einsum("ij,jk->ik", A, B, alpha=2.0, beta=0.5, out=C)

        # DF-MP2 energy contraction
        einsum("iap,jbp->ijab", B, B)

        # High-precision accumulation (fp64 internally, result in original dtype)
        einsum("ij,jk->ik", A, B, precision="kahan")

        # Stochastic rounding: correctness requirement for BF16 K ≥ ~300
        einsum("ij,jk->ik", A, B, precision="sr")

        # Automatic precision selection: trntensor picks the cheapest mode
        einsum("ij,jk->ik", A, B, target_forward_error=1e-4)
    """
    if target_forward_error is not None:
        if precision != "fast":
            raise ValueError(
                "Cannot specify both 'precision' and 'target_forward_error'. Use one or the other."
            )
        from .plan import _parse_subscripts, select_precision_for_error

        _input_str, _output_str = _parse_subscripts(subscripts)
        _size_map: dict[str, int] = {}
        for _term, _op in zip(_input_str.split(","), operands, strict=False):
            for _c, _s in zip(_term, _op.shape, strict=False):
                _size_map[_c] = int(_s)
        _contracted = {c for c in _size_map if c not in _output_str}
        K = 1
        for c in _contracted:
            K *= _size_map[c]
        eff_dtype = _resolve_dtype(dtype) or (operands[0].dtype if operands else torch.float32)
        precision = select_precision_for_error(eff_dtype, K, target_forward_error)
    # Early detection: ShardedTensor operands route to the sharded dispatch path.
    from .parallel import ShardedTensor

    if any(isinstance(op, ShardedTensor) for op in operands):
        if out is not None:
            raise NotImplementedError(
                "The 'out' accumulation argument is not supported with ShardedTensor operands."
            )
        return _execute_sharded(subscripts, operands, alpha=alpha, dtype=dtype, precision=precision)

    target = _resolve_dtype(dtype)
    if target is not None:
        operands = tuple(op.to(target) for op in operands)
    plan = plan_contraction(subscripts, *operands, precision=precision)
    result = _execute_contraction(subscripts, operands, plan)
    if alpha != 1.0:
        result = result.mul(alpha)
    if out is not None:
        result = result.add(out, alpha=beta)
    return result


def _execute_sharded(
    subscripts: str,
    operands: tuple,
    alpha: float = 1.0,
    dtype: str | torch.dtype | None = None,
    precision: str = "fast",
):
    """Dispatch sharded tensor contractions (output-parallel and reduce-parallel).

    Classifies each :class:`~trntensor.parallel.ShardedTensor` operand as either:

    - **Output-parallel**: the partition dimension maps to an output index.
      No cross-shard communication required — each shard is contracted
      independently and results are assembled into a ``ShardedTensor``.

    - **Reduce-parallel**: the partition dimension is a contraction index
      (not in the output). Each shard produces a partial result of the full
      output shape; the shards must be summed (all-reduced).

      On Trainium (``HAS_NKI=True``): raises ``NotImplementedError`` — requires
      ``nki.collectives`` arriving in Neuron SDK 2.30+.

      On CPU (``HAS_NKI=False``): dispatches via :func:`~trntensor.parallel._mock_allreduce`,
      a torch-based stand-in that enables reduce-parallel paths to be tested
      without hardware.

    Mixed output-parallel + reduce-parallel operands in the same einsum are
    not yet supported and raise ``NotImplementedError``.
    """
    from .nki.dispatch import HAS_NKI
    from .parallel import ShardedTensor, _mock_allreduce, shard_boundaries
    from .plan import _parse_subscripts

    input_str, output_str = _parse_subscripts(subscripts)
    input_terms = input_str.split(",")

    if len(operands) != len(input_terms):
        raise ValueError(
            f"Subscript has {len(input_terms)} operand terms but {len(operands)} operands given."
        )

    # Classify each ShardedTensor operand as output-parallel or reduce-parallel.
    output_parallel: dict[int, tuple[int, ShardedTensor]] = {}  # op_idx → (out_dim, shard_op)
    reduce_parallel: dict[int, tuple[str, ShardedTensor]] = {}  # op_idx → (shard_char, shard_op)

    for op_idx, op in enumerate(operands):
        if not isinstance(op, ShardedTensor):
            continue
        shard_char = input_terms[op_idx][op.partition_dim]
        if shard_char in output_str:
            out_dim = list(output_str).index(shard_char)
            output_parallel[op_idx] = (out_dim, op)
        else:
            reduce_parallel[op_idx] = (shard_char, op)

    # Reduce-parallel metadata — needed by both the mixed and pure reduce paths.
    rp_chars = {v[0] for v in reduce_parallel.values()}
    rp_n_shards_set = {v[1].n_shards for v in reduce_parallel.values()}

    # ------------------------------------------------------------------ #
    # Mixed output-parallel + reduce-parallel path                         #
    # ------------------------------------------------------------------ #
    if output_parallel and reduce_parallel:
        # Validate output-parallel side (same constraint as pure output-parallel)
        if len(output_parallel) > 1:
            out_dims = {v[0] for v in output_parallel.values()}
            n_shards_set = {v[1].n_shards for v in output_parallel.values()}
            if len(out_dims) > 1 or len(n_shards_set) > 1:
                raise NotImplementedError(
                    "Multiple ShardedTensor operands in a mixed einsum must all be "
                    "partitioned along the same output dimension with the same n_shards. "
                    "Heterogeneous multi-operand sharding is not yet supported."
                )
        # Validate reduce-parallel side (same constraint as pure reduce-parallel)
        if len(rp_chars) > 1 or len(rp_n_shards_set) > 1:
            raise NotImplementedError(
                "Multiple reduce-parallel ShardedTensor operands in a mixed einsum must "
                "all be partitioned along the same contracted dimension with the same "
                "n_shards."
            )
        # NKI gate: inner all-reduce still requires nki.collectives (SDK 2.30+)
        if HAS_NKI:
            shard_char_rp = next(iter(rp_chars))
            raise NotImplementedError(
                f"Mixed output-parallel + reduce-parallel sharding: reducing across "
                f"dimension '{shard_char_rp}' requires all-reduce collective operations "
                "(nki.collectives, Neuron SDK 2.30+). Use output-parallel sharding only "
                "until then."
            )
        # CPU: nested outer-output × inner-reduce dispatch
        first_out_idx = next(iter(output_parallel))
        out_dim, ref_out_shard = output_parallel[first_out_idx]
        out_n_shards = ref_out_shard.n_shards
        shard_char = next(iter(rp_chars))
        ref_rp = next(iter(reduce_parallel.values()))[1]
        rp_n = ref_rp.n_shards
        bounds = shard_boundaries(ref_rp)

        output_chunks = []
        for out_s in range(out_n_shards):
            partials = []
            for rp_t in range(rp_n):
                start, end = bounds[rp_t]
                dense_ops = []
                for op_idx, op in enumerate(operands):
                    if op_idx in output_parallel:
                        chunk = output_parallel[op_idx][1].chunks[out_s]
                        if shard_char in input_terms[op_idx]:
                            dim_in_chunk = input_terms[op_idx].index(shard_char)
                            slices = [slice(None)] * chunk.ndim
                            slices[dim_in_chunk] = slice(start, end)
                            dense_ops.append(chunk[tuple(slices)])
                        else:
                            dense_ops.append(chunk)
                    elif op_idx in reduce_parallel:
                        dense_ops.append(reduce_parallel[op_idx][1].chunks[rp_t])
                    elif not isinstance(op, ShardedTensor) and shard_char in input_terms[op_idx]:
                        dim_in_op = input_terms[op_idx].index(shard_char)
                        slices = [slice(None)] * op.ndim
                        slices[dim_in_op] = slice(start, end)
                        dense_ops.append(op[tuple(slices)])
                    else:
                        dense_ops.append(op)
                partial = einsum(
                    subscripts, *dense_ops, alpha=alpha, dtype=dtype, precision=precision
                )
                partials.append(partial)
            output_chunks.append(_mock_allreduce(partials))

        return ShardedTensor(output_chunks, partition_dim=out_dim)

    # ------------------------------------------------------------------ #
    # Output-parallel path                                                 #
    # ------------------------------------------------------------------ #
    if output_parallel:
        if len(output_parallel) > 1:
            # Multiple sharded operands must share the same output dim and n_shards.
            out_dims = {v[0] for v in output_parallel.values()}
            n_shards_set = {v[1].n_shards for v in output_parallel.values()}
            if len(out_dims) > 1 or len(n_shards_set) > 1:
                raise NotImplementedError(
                    "Multiple ShardedTensor operands must all be partitioned along "
                    "the same output dimension with the same n_shards. "
                    "Heterogeneous multi-operand sharding is not yet supported."
                )

        first_op_idx = next(iter(output_parallel))
        out_dim, ref_shard = output_parallel[first_op_idx]
        n_shards = ref_shard.n_shards

        results = []
        for shard_idx in range(n_shards):
            dense_ops = tuple(
                (output_parallel[i][1].chunks[shard_idx] if i in output_parallel else op)
                for i, op in enumerate(operands)
            )
            result_chunk = einsum(
                subscripts, *dense_ops, alpha=alpha, dtype=dtype, precision=precision
            )
            results.append(result_chunk)

        return ShardedTensor(results, partition_dim=out_dim)

    # ------------------------------------------------------------------ #
    # Pure reduce-parallel path                                            #
    # ------------------------------------------------------------------ #
    # rp_chars and rp_n_shards_set already computed above.

    if HAS_NKI:
        shard_char = next(iter(rp_chars))
        raise NotImplementedError(
            f"Reduce-parallel sharding: dimension '{shard_char}' is a contraction "
            "index (not in output). Reducing across shards requires all-reduce "
            "collective operations, which land in Neuron SDK 2.30+ via "
            "nki.collectives. Use output-parallel sharding (partition along an output "
            "dimension) until then."
        )

    if len(rp_chars) > 1 or len(rp_n_shards_set) > 1:
        raise NotImplementedError(
            "Multiple reduce-parallel ShardedTensor operands must all be partitioned "
            "along the same contracted dimension with the same n_shards."
        )

    shard_char = next(iter(rp_chars))
    ref_shard = next(iter(reduce_parallel.values()))[1]
    n_shards = ref_shard.n_shards
    bounds = shard_boundaries(ref_shard)

    # Dispatch per shard, co-partitioning plain operands that share shard_char.
    partials = []
    for shard_idx in range(n_shards):
        start, end = bounds[shard_idx]
        dense_ops = []
        for op_idx, op in enumerate(operands):
            if op_idx in reduce_parallel:
                dense_ops.append(op.chunks[shard_idx])
            elif not isinstance(op, ShardedTensor) and shard_char in input_terms[op_idx]:
                dim_in_op = input_terms[op_idx].index(shard_char)
                slices = [slice(None)] * op.ndim
                slices[dim_in_op] = slice(start, end)
                dense_ops.append(op[tuple(slices)])
            else:
                dense_ops.append(op)
        partial = einsum(subscripts, *dense_ops, alpha=alpha, dtype=dtype, precision=precision)
        partials.append(partial)

    return _mock_allreduce(partials)


# Device types whose kernels compute genuine IEEE fp64. Anything else (the
# TorchNeuron "neuron" device, XLA, MPS) has no fp64 datapath — Trainium
# engines top out at fp32 — so a float64 contraction there is not a true
# fp64 accumulation. On Trn2 (SDK 2.33) precision="kahan" on device operands
# measured ~1.4 ulp of fp32 componentwise error, which an fp64 accumulation
# followed by one rounding cannot produce (it is bounded by 1 ulp + K·2⁻⁵³).
_FP64_DEVICE_TYPES = ("cpu", "cuda")


def _einsum_fp64(subscripts: str, operands: tuple) -> torch.Tensor:
    """Contract in true fp64 and round once to the operand dtype.

    Operands on a device without native fp64 are contracted on the host and
    the result is moved back, so the only rounding beyond the fp64
    accumulation is the final cast to the operand dtype (``u_out``).
    """
    orig_dtype = operands[0].dtype if operands else torch.float32
    orig_device = operands[0].device if operands else torch.device("cpu")
    if all(op.device.type in _FP64_DEVICE_TYPES for op in operands):
        result = torch.einsum(subscripts, *(op.to(torch.float64) for op in operands))
        return result.to(orig_dtype)
    host = (op.to(device="cpu", dtype=torch.float64) for op in operands)
    result = torch.einsum(subscripts, *host).to(orig_dtype)
    return result.to(orig_device)


def _execute_contraction(subscripts: str, operands: tuple, plan: ContractionPlan) -> torch.Tensor:
    """Execute contraction according to plan."""
    if plan.precision == "kahan":
        return _einsum_fp64(subscripts, operands)
    elif plan.precision == "dd":
        from .nki.dispatch import HAS_NKI, _ozaki_matmul_cpu

        if HAS_NKI:
            raise NotImplementedError(
                "precision='dd' on Trainium requires trnblas Phase 2 double-double "
                "GEMM kernels (trnblas#22). Use precision='kahan' (fp64 accumulation) "
                "as an alternative."
            )
        # CPU: Ozaki-2 for matmul; recurse for path (binary steps get Ozaki);
        # kahan fallback for bmm and torch strategies.
        if plan.strategy == "matmul":
            A, B = operands
            a = A.T if plan.transA else A
            b = B.T if plan.transB else B
            result = _ozaki_matmul_cpu(a.float(), b.float(), k=2)
            return result.to(A.dtype)
        if plan.strategy == "path":
            return _execute_path(subscripts, operands, plan)
        return _einsum_fp64(subscripts, operands)
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
    return nki_matmul(a, b, use_sr=(plan.precision == "sr"))


def _execute_bmm(subscripts: str, operands: tuple, plan: ContractionPlan) -> torch.Tensor:
    """Execute as batched matmul."""
    from .nki.dispatch import nki_batched_matmul

    A, B = operands
    return nki_batched_matmul(A, B, use_sr=(plan.precision == "sr"))


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

        # Normalize operand order: _plan_binary expects A's free dims to appear
        # before B's free dims in the output (standard matmul orientation).
        # When the greedy path leaves operands in the wrong order (e.g.
        # "kl,ik->il" where i comes from the second operand), swap them so the
        # binary subscript is in canonical form.
        step_contracted = (set(inputs[i]) | set(inputs[j])) - set(inter_str)
        free_i = set(inputs[i]) - step_contracted
        if inter_str and inter_str[0] not in free_i:
            sub = f"{inputs[j]},{inputs[i]}->{inter_str}"
            result = einsum(sub, ops[j], ops[i], precision=plan.precision)
        else:
            sub = f"{inputs[i]},{inputs[j]}->{inter_str}"
            result = einsum(sub, ops[i], ops[j], precision=plan.precision)

        ops = [op for k, op in enumerate(ops) if k not in (i, j)] + [result]
        inputs = [inp for k, inp in enumerate(inputs) if k not in (i, j)] + [inter_str]

    return ops[0]


def _try_batched_multi_einsum(
    contractions: list, precision: str = "fast"
) -> list[torch.Tensor] | None:
    """Batch homogeneous 2D matmul contractions into a single dispatch.

    Returns a result list when all contractions share the same subscript,
    are 2-operand, resolve to strategy ``"matmul"``, and have identical
    operand shapes. Returns ``None`` otherwise (caller falls through to
    the per-contraction loop).

    Amortizes the ~0.67 ms fixed XLA dispatch latency across N contractions:
    instead of N individual ``nki_matmul`` calls each paying that overhead,
    the stack is dispatched once via ``nki_batched_matmul``.

    Only optimizes ``"fast"`` and ``"sr"`` precision; ``"kahan"`` and ``"dd"``
    fall through to the per-contraction loop which handles them correctly.
    """
    if precision not in ("fast", "sr"):
        return None
    if len(contractions) < 2:
        return None
    subscripts = contractions[0][0]
    if not all(c[0] == subscripts for c in contractions):
        return None
    if not all(len(c) == 3 for c in contractions):
        return None
    plan = plan_contraction(subscripts, contractions[0][1], contractions[0][2])
    if plan.strategy != "matmul":
        return None
    shape_key = (contractions[0][1].shape, contractions[0][2].shape)
    if not all((c[1].shape, c[2].shape) == shape_key for c in contractions):
        return None

    from .nki.dispatch import nki_batched_matmul

    A_stack = torch.stack([c[1].T if plan.transA else c[1] for c in contractions])
    B_stack = torch.stack([c[2].T if plan.transB else c[2] for c in contractions])
    out_stack = nki_batched_matmul(A_stack, B_stack, use_sr=(precision == "sr"))
    if out_stack.device.type == "xla":
        from .nki.dispatch import from_xla

        out_stack = from_xla(out_stack)
    return [out_stack[i] for i in range(len(contractions))]


def multi_einsum(
    *contractions: tuple,
    precision: str = "fast",
    target_forward_error: float | None = None,
) -> list[torch.Tensor]:
    """Execute multiple contractions, fusing where possible.

    Each contraction is (subscripts, *operands).
    Returns list of results.

    Shared operands (same tensor object appearing in more than one
    contraction) are pre-pinned to XLA once when NKI is available,
    avoiding repeated host→device transfers.

    When all contractions are homogeneous — same subscript, same operand
    shapes, 2-operand matmul — they are stacked and dispatched as a single
    ``nki_batched_matmul`` call, amortizing the fixed XLA dispatch overhead
    across the batch. Heterogeneous contractions fall through to the
    per-contraction loop.

    Args:
        *contractions: Each element is ``(subscripts, *operands)``.
        precision: Accumulation precision applied to every contraction.
            ``"fast"`` (default), ``"kahan"``, ``"sr"``, or ``"dd"``.
            Homogeneous batching is only attempted for ``"fast"`` and
            ``"sr"``; other modes fall through to the per-contraction loop.
            On Trainium, ``"sr"`` requires ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``
            at process start (see :func:`einsum`).
        target_forward_error: Optional relative forward error bound. When set,
            trntensor selects the cheapest precision mode per contraction.
            For homogeneous batching the mode is resolved from the first
            contraction's K and dtype (all share the same shape). For the
            per-contraction fallback each call resolves independently.
            Cannot be combined with an explicit ``precision=`` argument.

    Useful for DF-MP2 where many independent contractions share operands:
        results = multi_einsum(
            ("iap,jbp->ijab", B, B),
            ("ibp,jap->ijab", B, B),  # Exchange term
            precision="sr",
        )
    """
    if target_forward_error is not None and precision != "fast":
        raise ValueError(
            "Cannot specify both 'precision' and 'target_forward_error'. Use one or the other."
        )
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

    # Build substituted contraction list with pre-pinned operands applied
    subst = [(c[0], *(xla_map.get(id(op), op) for op in c[1:])) for c in contractions]

    # Pre-resolve target_forward_error → precision for the homogeneous batching
    # path (_try_batched_multi_einsum accepts only precision=). All contractions
    # in a homogeneous batch share the same subscript and operand shapes, so K
    # is computed once from the first contraction.
    batched_precision = precision
    if target_forward_error is not None and subst:
        from .plan import _parse_subscripts, select_precision_for_error

        _c0 = subst[0]
        _input_str, _output_str = _parse_subscripts(_c0[0])
        _size_map: dict[str, int] = {}
        for _term, _op in zip(_input_str.split(","), _c0[1:], strict=False):
            for _ch, _sz in zip(_term, _op.shape, strict=False):
                _size_map[_ch] = int(_sz)
        _K = 1
        for _ch in {_ch for _ch in _size_map if _ch not in _output_str}:
            _K *= _size_map[_ch]
        _dtype = _c0[1].dtype if len(_c0) > 1 else torch.float32
        batched_precision = select_precision_for_error(_dtype, _K, target_forward_error)

    # Try homogeneous batching: N same-shape matmuls → one batched dispatch
    batched = _try_batched_multi_einsum(subst, precision=batched_precision)
    if batched is not None:
        return batched

    # Fallback: per-contraction loop — pass target_forward_error so each call
    # resolves its own K independently (optimal for heterogeneous contractions).
    results = []
    for c in subst:
        subscripts = c[0]
        ops = c[1:]
        result = einsum(
            subscripts, *ops, precision=precision, target_forward_error=target_forward_error
        )
        if hasattr(result, "device") and result.device.type == "xla":
            from .nki.dispatch import from_xla

            result = from_xla(result)
        results.append(result)
    return results
