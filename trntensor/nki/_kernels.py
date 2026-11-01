"""NKI kernels for tensor contractions.

The 2-index matmul kernel mirrors trnblas's GEMM pattern: stationary A tile
reuse, K tiled to 128 (partition-dim limit), N tiled to 512 (moving free-dim
limit). All transpose variants route through the dispatch layer, which
pre-transposes before entry — the kernel always sees the canonical
``C = A @ B`` form.

Targets NKI 0.3.0 Stable (Neuron SDK 2.29). The legacy
``neuronxcc.nki.*`` shim is not used.
"""

from __future__ import annotations

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False


# NKI 0.3.0 systolic-array limits (unchanged vs 2.24):
# stationary partition dim ≤ 128 (K), free dim ≤ 128 (M).
# moving free dim ≤ 512 (N).
TILE_M = 128
TILE_K = 128
TILE_N = 512


def ao_to_mo_trn2_tiling(naux: int, packed: bool = False) -> tuple[int, int]:
    """``(TP, NUB)`` for the Trn2 AO→MO kernels: aux chunk width and ν per block.

    ``TP`` is the widest aux chunk (≤ 512, one PSUM bank) that still gives
    an even chunk count, so both programs get work at LNC=2; else 128. The
    packed kernel gathers ``TP``-wide rows, so when no even split exists it
    takes the widest divisor of ``naux`` instead (wider rows, fewer DMA
    descriptors; it then runs as one program, like the full kernel).
    ``NUB`` (a power of two, so it divides 128) keeps a stage-1 block at
    ``NUB·TP ≤ 2048`` elements per partition per μ-tile.
    """
    tp = 128
    for cand in (512, 256):
        if naux % (2 * cand) == 0:
            tp = cand
            break
    else:
        if packed:
            for cand in (512, 384, 256):
                if naux % cand == 0:
                    tp = cand
                    break
    return tp, ao_to_mo_trn2_nub(tp)


def ao_to_mo_trn2_nub(tp: int) -> int:
    """ν per stage-1 block for aux chunk ``tp``: the largest power of two
    ≤ 8 with ``NUB·tp ≤ 2048``."""
    nub = 1
    while nub * 2 * tp <= 2048 and nub < 8:
        nub *= 2
    return nub


if HAS_NKI:

    @nki.jit
    def matmul_kernel(a, b, use_sr: bool = False):
        """``C = A @ B`` with stationary-A tile reuse.

        Caller guarantees M, K are multiples of ``TILE_M``/``TILE_K`` and N
        is either ≤ ``TILE_N`` or a clean multiple of ``TILE_N``. PSUM
        accumulates over K-tiles before the single store per (m, n) pair.

        When ``use_sr=True``, the PSUM→SBUF downcast runs on the Scalar
        Engine (``nisa.activation(op=nl.copy)``) instead of
        ``nisa.tensor_copy``. NKI 0.7 has no per-instruction rounding-mode
        argument, so stochastic rounding there is a runtime setting (#41):
        it happens only when ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1`` is set at
        process start, and then for every downcast in the process, including
        ``use_sr=False``. The dispatcher warns once when SR is requested
        without it. SR is a correctness requirement for BF16
        dot products with K ≥ ~300 (Connolly–Higham–Mary, SIAM J. Sci.
        Comput. 2021): stochastic rounding replaces Wilkinson's n·u
        worst-case bias with a mean-zero √n·u bound.
        """
        M, K = a.shape
        _, N = b.shape

        tile_m = TILE_M
        tile_k = TILE_K
        tile_n = N if N <= TILE_N else TILE_N

        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // tile_m):
            for n in nl.affine_range(N // tile_n):
                m_off = m * tile_m
                n_off = n * tile_n

                psum = nl.zeros((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // tile_k):
                    k_off = k * tile_k
                    a_t = nl.load_transpose2d(a[m_off : m_off + tile_m, k_off : k_off + tile_k])
                    b_tile = nl.load(b[k_off : k_off + tile_k, n_off : n_off + tile_n])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)

                # NKI 0.3.0: nl.copy(psum, ...) returns a view. Allocate a
                # fresh SBUF tile and nisa.tensor_copy PSUM into it before
                # the HBM store.
                c_sbuf = nl.ndarray((tile_m, tile_n), dtype=a.dtype, buffer=nl.sbuf)
                if use_sr:
                    # NKI 0.7 dropped activation's ``out_dtype`` / ``round_mode``
                    # kwargs (#41); the dtype comes from ``dst``. Downcast on the
                    # Scalar Engine; whether it rounds stochastically is a
                    # runtime setting, not a per-instruction one.
                    nisa.activation(dst=c_sbuf, op=nl.copy, data=psum)
                else:
                    nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(
                    c[m_off : m_off + tile_m, n_off : n_off + tile_n],
                    value=c_sbuf,
                )

        return c

    @nki.jit
    def batched_matmul_kernel(a, b, use_sr: bool = False):
        """Batched ``C[b] = A[b] @ B[b]`` — one matmul per batch slice.

        Caller pads (M, K, N) the same way as ``matmul_kernel``. The
        batch dim is iterated with ``nl.affine_range``; each slice
        reuses the stationary-A tile layout. ``use_sr`` routes the
        PSUM→SBUF downcast through the Scalar Engine; it rounds
        stochastically only under ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``
        (see ``matmul_kernel`` for the runtime rule and the correctness
        motivation).
        """
        B, M, K = a.shape
        _, _, N = b.shape

        tile_m = TILE_M
        tile_k = TILE_K
        tile_n = N if N <= TILE_N else TILE_N

        c = nl.ndarray((B, M, N), dtype=a.dtype, buffer=nl.shared_hbm)

        for batch in nl.affine_range(B):
            for m in nl.affine_range(M // tile_m):
                for n in nl.affine_range(N // tile_n):
                    m_off = m * tile_m
                    n_off = n * tile_n

                    psum = nl.zeros((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)

                    for k in nl.affine_range(K // tile_k):
                        k_off = k * tile_k
                        a_t = nl.load_transpose2d(
                            a[batch, m_off : m_off + tile_m, k_off : k_off + tile_k]
                        )
                        b_tile = nl.load(b[batch, k_off : k_off + tile_k, n_off : n_off + tile_n])
                        nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)

                    c_sbuf = nl.ndarray((tile_m, tile_n), dtype=a.dtype, buffer=nl.sbuf)
                    if use_sr:
                        nisa.activation(dst=c_sbuf, op=nl.copy, data=psum)
                    else:
                        nisa.tensor_copy(src=psum, dst=c_sbuf)
                    nl.store(
                        c[batch, m_off : m_off + tile_m, n_off : n_off + tile_n],
                        value=c_sbuf,
                    )

        return c

    @nki.jit
    def mp2_energy_kernel(B_T, eps_occ_b, eps_vir, eps_vir_b):
        """Fully fused DF-MP2 correlation energy with K/M tiling.

        NVIR and NAUX must be multiples of ``TILE_M`` / ``TILE_K``; the
        dispatch layer pads both before entry and lays the inputs out so
        that every load is a plain (non-transposing) DMA and every Vector
        Engine op keeps its partition dim (#41).

        For each ``(i, j)`` pair this program tiles over the ``(a, b)``
        virtual-orbital block (M-tiling, TILE_M=128 per side) and over the
        auxiliary index P (K-tiling, TILE_K=128):

        1. For each ``(m_tile, n_tile)`` block:
           K-accumulate ``T[m,n] = Σ_P B[i,m,P] B[j,n,P]`` and
           ``T.T[m,n] = Σ_P B[j,m,P] B[i,n,P]`` in PSUM;
        2. PSUM → SBUF; build ``-Δ_ab = ε_a + ε_b - ε_i - ε_j`` as a
           ``(TILE_M, TILE_M)`` tile: per-partition column
           ``ε_a - ε_i - ε_j`` plus the pre-broadcast row ``ε_b``;
        3. Row-reduce ``T*(2T-T.T)/(-Δ)`` along the free axis and subtract
           into a ``(TILE_M, 1)`` per-partition accumulator.

        The final cross-partition sum is a ``(TILE_M, 1)ᵀ · ones`` matmul on
        the Tensor Engine (the Vector Engine cannot reduce across
        partitions). One HBM store per ``(i, j)``.

        T.T tile correctness:
        ``nc_matmul(stationary=Bj_m, moving=Bi_n)[da, db]``
        ``= Σ_P B[j, m+da, P] · B[i, n+db, P]``
        ``= T_{ji}[m+da, n+db]``
        ``= T_{ij}[n+db, m+da]``  (scalar product commutes)
        ``= T.T[m+da, n+db]`` ✓

        Inputs
        ------
        B_T       : (nocc, naux, nvir) — DF coefficients with the aux index
                                         ahead of the virtual index, so a
                                         ``(P, a)`` tile is a plain load;
                                         nvir % TILE_M == 0, naux % TILE_K == 0.
        eps_occ_b : (TILE_M, nocc)     — occupied energies broadcast across
                                         the 128 partitions.
        eps_vir   : (nvir, 1)          — virtual energies as a column (ε_a).
        eps_vir_b : (TILE_M, nvir)     — virtual energies broadcast across
                                         the 128 partitions (ε_b row).

        Returns
        -------
        (nocc, nocc) float32 partial tensor. Host sums to scalar.
        """
        NOCC, NAUX, NVIR = B_T.shape
        partial = nl.ndarray((NOCC, NOCC), dtype=nl.float32, buffer=nl.shared_hbm)
        ones = nl.ones((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)

        for i in nl.affine_range(NOCC):
            eo_i = nl.load(eps_occ_b[0:TILE_M, i : i + 1])  # (TILE_M, 1)

            for j in nl.affine_range(NOCC):
                eo_j = nl.load(eps_occ_b[0:TILE_M, j : j + 1])  # (TILE_M, 1)
                eo_sum = nl.add(eo_i, eo_j)  # (TILE_M, 1), ε_i + ε_j per lane

                # Per-partition (TILE_M, 1) accumulator for this (i, j) pair;
                # loop-carried, so the tile loops are sequential.
                acc = nl.zeros((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)

                for m_tile in nl.sequential_range(NVIR // TILE_M):
                    m_off = m_tile * TILE_M
                    ev_m = nl.load(eps_vir[m_off : m_off + TILE_M, 0:1])  # (TILE_M, 1)
                    col = nl.subtract(ev_m, eo_sum)  # ε_a - ε_i - ε_j

                    for n_tile in nl.sequential_range(NVIR // TILE_M):
                        n_off = n_tile * TILE_M
                        ev_n = nl.load(eps_vir_b[0:TILE_M, n_off : n_off + TILE_M])  # ε_b

                        psum_T = nl.zeros((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.psum)
                        psum_Tt = nl.zeros((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.psum)

                        for k_p in nl.affine_range(NAUX // TILE_K):
                            p_off = k_p * TILE_K
                            # (TILE_K, TILE_M) tiles: partition = P chunk.
                            Bi_m = nl.load(B_T[i, p_off : p_off + TILE_K, m_off : m_off + TILE_M])
                            Bj_n = nl.load(B_T[j, p_off : p_off + TILE_K, n_off : n_off + TILE_M])
                            Bj_m = nl.load(B_T[j, p_off : p_off + TILE_K, m_off : m_off + TILE_M])
                            Bi_n = nl.load(B_T[i, p_off : p_off + TILE_K, n_off : n_off + TILE_M])
                            # T[m,n] = Σ_P B[i,m,P] * B[j,n,P]
                            nisa.nc_matmul(
                                dst=psum_T, stationary=Bi_m, moving=Bj_n, accumulate=True
                            )
                            # T.T[m,n] = T[n,m] = Σ_P B[j,m,P] * B[i,n,P]
                            nisa.nc_matmul(
                                dst=psum_Tt, stationary=Bj_m, moving=Bi_n, accumulate=True
                            )

                        t = nl.ndarray((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
                        t_T = nl.ndarray((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(src=psum_T, dst=t)
                        nisa.tensor_copy(src=psum_Tt, dst=t_T)

                        # -Δ_ab = ε_b + (ε_a - ε_i - ε_j): (TILE_M, TILE_M) + (TILE_M, 1)
                        neg_denom = nl.add(ev_n, col)

                        # term = T * (2T - T.T) / Δ = -T * (2T - T.T) / (-Δ).
                        # Multiply by the reciprocal (tensor-tensor divide is not
                        # available), reduce the free axis per lane, subtract.
                        two_t_minus_tT = nl.subtract(nl.multiply(t, 2.0), t_T)
                        numerator = nl.multiply(t, two_t_minus_tT)
                        q = nl.multiply(numerator, nl.reciprocal(neg_denom))
                        row = nl.sum(q, axis=1, keepdims=True)  # (TILE_M, 1)
                        acc[...] = nl.subtract(acc, row)

                # Cross-partition sum: accᵀ · ones → (1, 1).
                psum_r = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=psum_r, stationary=acc, moving=ones, accumulate=True)
                e_ij = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_r, dst=e_ij)
                nl.store(partial[i : i + 1, j : j + 1], value=e_ij)

        return partial

    @nki.jit
    def ao_to_mo_transform_kernel(eri, C_occ, C_vir):
        """Fused 4-index AO→MO integral transform with K-tiling over μ/ν and
        N-tiling over the ν (step 1) and virtual-orbital a (step 2) dimensions.

        Computes ``B[i, a, P] = Σ_{μ,ν} C_occ[μ, i] · C_vir[ν, a] · eri[μ, ν, P]``
        as one NKI program. Per auxiliary index P:

        1. **Step 1** (K-tiled over μ, M-tiled over ν):
           ``interᵀ(ν, i) = Σ_μ eri(μ,ν,P) · C_occ(μ,i)``
           — the eri tile is the stationary operand, so ν is partitioned into
           ``TILE_K=128``-row tiles; within each ν-tile, μ is K-tiled in
           ``TILE_K`` chunks. PSUM per ν-tile; result ``(TILE_K, NOCC)``
           (ν-major) written to kernel-scratch HBM. Computing the
           intermediate already transposed replaces the former
           ``load_transpose2d`` of a strided HBM slice, which the trn2 BIR
           verifier rejects ("transposing dimensions should be continuous",
           #41).
        2. **Step 2** (K-tiled over ν, N-tiled over a):
           ``B(i, a) = Σ_ν intermediate(ν,i) · C_vir(ν,a)``
           — a is partitioned into ``TILE_N``-wide tiles; ν is K-tiled as before.
           Result ``(NOCC, TILE_N)`` written to B_out.

        The intermediate lives in kernel-scratch HBM between the two steps;
        it is never a user-visible tensor.

        Constraints (enforced at the dispatch boundary, not here):
        - NBASIS must be a multiple of ``TILE_K`` (=128); dispatcher pads.
        - NVIR must be a multiple of ``TILE_N``; dispatcher pads and slices.
        - NOCC ≤ 128 (stationary M dim of the step-2 nc_matmul).

        Inputs
        ------
        eri   : (nbasis, nbasis, naux) — AO-basis density-fitted ERIs;
                nbasis is a multiple of TILE_K (dispatcher pads).
        C_occ : (nbasis, nocc)         — occupied MO coefficients.
        C_vir : (nbasis, nvir)         — virtual MO coefficients;
                nvir is a multiple of TILE_N (dispatcher pads).

        Returns
        -------
        (nocc, nvir, naux) tensor of transformed coefficients B_ia^P.
        """
        NBASIS, _, NAUX = eri.shape
        _, NOCC = C_occ.shape
        _, NVIR = C_vir.shape

        B_out = nl.ndarray((NOCC, NVIR, NAUX), dtype=eri.dtype, buffer=nl.shared_hbm)
        # Kernel-scratch HBM: holds the post-step-1 intermediate, ν-major
        # (NBASIS, NOCC) for each P. Written and reloaded in TILE_K-row ν-chunks.
        inter_hbm = nl.ndarray((NBASIS, NOCC, NAUX), dtype=eri.dtype, buffer=nl.shared_hbm)

        for p in nl.affine_range(NAUX):
            # ------------------------------------------------------------------
            # Step 1: interᵀ(ν, i) = Σ_μ eri(μ,ν,P) · C_occ(μ,i)
            # Tiled over ν (TILE_K=128 chunks, the stationary free-dim limit);
            # K-tiled over μ (TILE_K). The eri tile is the stationary operand,
            # so PSUM comes out already ν-major — exactly the (partition=ν,
            # free=i) layout step 2 needs as its stationary operand. No
            # transposing DMA anywhere (#41).
            # ------------------------------------------------------------------
            for v_tile in nl.affine_range(NBASIS // TILE_K):
                v_off = v_tile * TILE_K
                psum_1 = nl.zeros((TILE_K, NOCC), dtype=nl.float32, buffer=nl.psum)
                for k_mu in nl.affine_range(NBASIS // TILE_K):
                    k_off = k_mu * TILE_K
                    # eri tile: (TILE_K, TILE_K) — partition=μ_chunk, free=ν_chunk
                    eri_tile = nl.load(eri[k_off : k_off + TILE_K, v_off : v_off + TILE_K, p])
                    # C_occ tile: (TILE_K, NOCC) — partition=μ_chunk, free=i
                    C_occ_tile = nl.load(C_occ[k_off : k_off + TILE_K, 0:NOCC])
                    nisa.nc_matmul(
                        dst=psum_1, stationary=eri_tile, moving=C_occ_tile, accumulate=True
                    )
                # PSUM → SBUF → kernel-scratch HBM for this ν-chunk.
                inter_sbuf = nl.ndarray((TILE_K, NOCC), dtype=eri.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_1, dst=inter_sbuf)
                nl.store(inter_hbm[v_off : v_off + TILE_K, 0:NOCC, p], value=inter_sbuf)

            # ------------------------------------------------------------------
            # Step 2: B(i, a) = Σ_ν intermediate(ν,i) · C_vir(ν,a)
            # N-tiled over a (TILE_N=512 chunks); K-tiled over ν (TILE_K=128).
            # psum_2 accumulates one (NOCC, TILE_N) a-block at a time.
            # ------------------------------------------------------------------
            for a_tile in nl.affine_range(NVIR // TILE_N):
                a_off = a_tile * TILE_N
                psum_2 = nl.zeros((NOCC, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k_nu in nl.affine_range(NBASIS // TILE_K):
                    k_off = k_nu * TILE_K
                    # interᵀ tile: (TILE_K, NOCC) — partition=ν_chunk, free=i.
                    inter_tile = nl.load(inter_hbm[k_off : k_off + TILE_K, 0:NOCC, p])
                    # C_vir tile: (TILE_K, TILE_N) — partition=ν_chunk, free=a_chunk
                    C_vir_tile = nl.load(C_vir[k_off : k_off + TILE_K, a_off : a_off + TILE_N])
                    nisa.nc_matmul(
                        dst=psum_2, stationary=inter_tile, moving=C_vir_tile, accumulate=True
                    )
                out_sbuf = nl.ndarray((NOCC, TILE_N), dtype=eri.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_2, dst=out_sbuf)
                nl.store(B_out[0:NOCC, a_off : a_off + TILE_N, p], value=out_sbuf)

        return B_out


# ── Trn2 (native TorchNeuron) kernels, #51 ───────────────────────────────────
#
# Launched gridless on the native path. At LNC=2 (the Trn2 default) NKI traces
# each kernel once per physical core of the logical core (``num_programs() ==
# 2``) and every program takes its share of the work; at LNC=1 there is one
# program. No DMA transposes: operands are either laid out by the host so the
# contraction index is the leading (partition) axis, or transposed on the
# Tensor Engine from SBUF. PSUM tiles are fresh ``nl.zeros`` per output block,
# at most one 2 KiB bank per partition, accumulated with ``accumulate=True``.

if HAS_NKI:

    def _programs(*counts):
        """(n_prog, pid, axis) for splitting work across the logical core.

        ``axis`` is the index of the first entry of ``counts`` divisible by
        ``num_programs()``; with none divisible (or one program) the kernel
        runs as a single program over everything (n_prog 1, axis -1).
        """
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        if n_prog > 1:
            for axis, count in enumerate(counts):
                if count % n_prog == 0:
                    return n_prog, nl.program_id(0), axis
        return 1, 0, -1

    @nki.jit
    def mp2_energy_row_kernel(bt_i, Bw, eps_occ_i, eps_occ_w, eps_vir_col, eps_vir_row):
        """DF-MP2 pair energies for one occupied row ``i`` over a window of ``j``.

        Same design as trnblas ``_pair_energy_sym_kernel`` (trnblas#55/#58).
        The host launches it once per ``i`` with ``j = i + d (mod nocc)``,
        ``d = 0..ND-1`` (pair symmetry ``e_ij = e_ji``), and weights each ``d``.

        bt_i:        (NAUX, NVIR) = B[i].T; loaded into SBUF once.
        Bw:          (ND, NAUX, NVIR) window of B[j].T; ND even.
        eps_occ_i:   (1, 1); eps_occ_w: (1, ND).
        eps_vir_col: (NVIR, 1); eps_vir_row: (1, NVIR).
        NAUX, NVIR multiples of 128 (zero padded by the host).

        Per pair, one GEMM fills ``T = B[i] B[j].T`` in SBUF a 128-row strip
        at a time; ``T.T`` blocks are Tensor Engine transposes of the
        SBUF-resident ``T``. ``T`` never reaches HBM. Each program takes half
        of the ``j`` window at LNC=2. Returns per-partition partials
        ``(128, ND)``.
        """
        ND = Bw.shape[0]
        NAUX = Bw.shape[1]
        NVIR = Bw.shape[2]
        TILE = 128
        TN = 512 if NVIR % 512 == 0 else 128
        NA = NVIR // TILE
        NB = NVIR // TN
        NK = NAUX // TILE
        n_prog, pid, _axis = _programs(ND)
        NJ = ND // n_prog
        e_partial = nl.ndarray((TILE, ND), dtype=nl.float32, buffer=nl.shared_hbm)
        eps_i = nl.load(eps_occ_i[0:1, 0:1])
        ev_row = nl.load(eps_vir_row[0:1, 0:NVIR])
        bi_sb = nl.ndarray((TILE, NK * NVIR), dtype=nl.float32, buffer=nl.sbuf)
        for k in nl.affine_range(NK):
            bi_sb[0:TILE, k * NVIR : (k + 1) * NVIR] = nl.load(
                bt_i[k * TILE : (k + 1) * TILE, 0:NVIR]
            )
        acc_j = nl.zeros((TILE, NJ), dtype=nl.float32, buffer=nl.sbuf)
        for jj in nl.affine_range(NJ):
            j = pid * NJ + jj
            eps_j = nl.load(eps_occ_w[0:1, j : j + 1])
            eo = nl.add(eps_i, eps_j)
            bj_sb = nl.ndarray((TILE, NK * NVIR), dtype=nl.float32, buffer=nl.sbuf)
            for k in nl.affine_range(NK):
                bj_sb[0:TILE, k * NVIR : (k + 1) * NVIR] = nl.load(
                    Bw[j, k * TILE : (k + 1) * TILE, 0:NVIR]
                )
            t_all = nl.ndarray((TILE, NA * NVIR), dtype=nl.float32, buffer=nl.sbuf)
            for a in nl.affine_range(NA):
                a_off = a * TILE
                for b in nl.affine_range(NB):
                    b_off = b * TN
                    pt = nl.zeros((TILE, TN), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(NK):
                        nisa.nc_matmul(
                            dst=pt,
                            stationary=bi_sb[0:TILE, k * NVIR + a_off : k * NVIR + a_off + TILE],
                            moving=bj_sb[0:TILE, k * NVIR + b_off : k * NVIR + b_off + TN],
                            accumulate=True,
                        )
                    nisa.tensor_copy(
                        src=pt, dst=t_all[0:TILE, a * NVIR + b_off : a * NVIR + b_off + TN]
                    )
            acc_ab = nl.zeros((TILE, NA * NA), dtype=nl.float32, buffer=nl.sbuf)
            for a in nl.affine_range(NA):
                a_off = a * TILE
                ev_a = nl.load(eps_vir_col[a_off : a_off + TILE, 0:1])
                for b in nl.affine_range(NA):
                    b_off = b * TILE
                    # T.T[a-strip, b-strip] = (T[b-strip, a-strip]).T
                    ptt = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=ptt, data=t_all[0:TILE, b * NVIR + a_off : b * NVIR + a_off + TILE]
                    )
                    tt = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(src=ptt, dst=tt)
                    t = t_all[0:TILE, a * NVIR + b_off : a * NVIR + b_off + TILE]
                    denom = nl.subtract(
                        nl.subtract(
                            nl.broadcast_to(eo, (TILE, TILE)), nl.broadcast_to(ev_a, (TILE, TILE))
                        ),
                        nl.broadcast_to(ev_row[0:1, b_off : b_off + TILE], (TILE, TILE)),
                    )
                    num = nl.multiply(t, nl.subtract(nl.multiply(t, 2.0), tt))
                    # Scalar Engine reciprocal (about 1.5e-6 relative), which
                    # keeps the Vector Engine free for the rest of the fold.
                    # It saturates outside about |x| in [2^-42, 2^41]; MP2
                    # denominators are orbital-energy gaps well inside that,
                    # and padded entries have a zero numerator. The Vector
                    # Engine's nisa.reciprocal (correctly rounded, full range)
                    # measured ~10% slower end to end on Trn2 (#51).
                    term = nl.multiply(num, nl.reciprocal(denom))
                    acc_ab[0:TILE, a * NA + b : a * NA + b + 1] = nl.sum(
                        term, axis=1, keepdims=True
                    )
            acc_j[0:TILE, jj : jj + 1] = nl.sum(acc_ab, axis=1, keepdims=True)
        nl.store(e_partial[0:TILE, pid * NJ : pid * NJ + NJ], value=acc_j)
        return e_partial

    def _ao_to_mo_trn2_body(eri, img, C_occ, C_vir, NB, NAUX, TP, NUB):
        """Shared body of the full and packed Trn2 AO→MO kernels.

        ``img`` is None for the full ``(NB, NB, NAUX)`` ``eri``. Otherwise
        ``eri`` is the packed lower triangle viewed as ``(npair·NPC, TP)``
        rows (row ``NPC·pair + c`` = aux chunk ``c`` of packed row ``pair``)
        and ``img`` the ``(NPC, 128, NKB·NB)`` uint32 gather indices; stage 1
        then gathers each ``(128, NUB, TP)`` block straight from the packed
        tensor with one batched indirect DMA, so the full square never
        exists in HBM.
        """
        _, NOCC = C_occ.shape
        _, NVIR = C_vir.shape
        TILE = 128
        NPC = NAUX // TP
        NKB = NB // TILE
        NAV = NVIR // TILE
        n_prog, pid, _axis = _programs(NPC)
        NPC_per = NPC // n_prog

        B_out = nl.ndarray((NOCC, NVIR, NAUX), dtype=eri.dtype, buffer=nl.shared_hbm)
        x_hbm = nl.ndarray((NOCC, NB, NAUX), dtype=eri.dtype, buffer=nl.shared_hbm)

        co_sb = nl.ndarray((TILE, NKB * NOCC), dtype=C_occ.dtype, buffer=nl.sbuf)
        cv_sb = nl.ndarray((TILE, NKB * NVIR), dtype=C_vir.dtype, buffer=nl.sbuf)
        for k in nl.affine_range(NKB):
            co_sb[0:TILE, k * NOCC : (k + 1) * NOCC] = nl.load(
                C_occ[k * TILE : (k + 1) * TILE, 0:NOCC]
            )
            cv_sb[0:TILE, k * NVIR : (k + 1) * NVIR] = nl.load(
                C_vir[k * TILE : (k + 1) * TILE, 0:NVIR]
            )

        for pc in nl.affine_range(NPC_per):
            c = pid * NPC_per + pc
            p_off = c * TP
            if img is not None:
                # This chunk's gather indices: NKB·NB·4 bytes per partition.
                idx_sb = nl.ndarray((TILE, NKB * NB), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.dma_copy(dst=idx_sb, src=img[c])
            # Stage 1: X[:, ν, Pc] for every ν, NUB ν per DMA (one
            # (128, NUB, TP) eri block per μ-tile instead of NUB small loads).
            for vb in nl.affine_range(NB // NUB):
                v0 = vb * NUB
                e_blk = nl.ndarray((TILE, NKB, NUB, TP), dtype=eri.dtype, buffer=nl.sbuf)
                for k in nl.affine_range(NKB):
                    if img is None:
                        e_blk[0:TILE, k, 0:NUB, 0:TP] = nl.load(
                            eri[k * TILE : (k + 1) * TILE, v0 : v0 + NUB, p_off : p_off + TP]
                        )
                    else:
                        # DMA batching: a (128, NUB) index block, read
                        # column-major, fills e_blk[:, k] row-major.
                        nisa.dma_copy(
                            dst=e_blk[0:TILE, k, 0:NUB, 0:TP],
                            src=eri.ap(
                                pattern=[[TP, TILE * NUB], [1, TP]],
                                vector_offset=idx_sb[0:TILE, k * NB + v0 : k * NB + v0 + NUB],
                                indirect_dim=0,
                            ),
                        )
                x_sb = nl.ndarray((NOCC, NUB, TP), dtype=eri.dtype, buffer=nl.sbuf)
                for u in nl.affine_range(NUB):
                    px = nl.zeros((NOCC, TP), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(NKB):
                        nisa.nc_matmul(
                            dst=px,
                            stationary=co_sb[0:TILE, k * NOCC : (k + 1) * NOCC],
                            moving=e_blk[0:TILE, k, u, 0:TP],
                            accumulate=True,
                        )
                    nisa.tensor_copy(src=px, dst=x_sb[0:NOCC, u, 0:TP])
                nl.store(x_hbm[0:NOCC, v0 : v0 + NUB, p_off : p_off + TP], value=x_sb)
            # Stage 2: B[i, a-tile, Pc].
            for i in nl.affine_range(NOCC):
                x_i = nl.ndarray((TILE, NKB * TP), dtype=eri.dtype, buffer=nl.sbuf)
                for k in nl.affine_range(NKB):
                    x_i[0:TILE, k * TP : (k + 1) * TP] = nl.load(
                        x_hbm[i, k * TILE : (k + 1) * TILE, p_off : p_off + TP]
                    )
                for av in nl.affine_range(NAV):
                    pb = nl.zeros((TILE, TP), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(NKB):
                        nisa.nc_matmul(
                            dst=pb,
                            stationary=cv_sb[
                                0:TILE, k * NVIR + av * TILE : k * NVIR + (av + 1) * TILE
                            ],
                            moving=x_i[0:TILE, k * TP : (k + 1) * TP],
                            accumulate=True,
                        )
                    b_sb = nl.ndarray((TILE, TP), dtype=eri.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(src=pb, dst=b_sb)
                    nl.store(B_out[i, av * TILE : (av + 1) * TILE, p_off : p_off + TP], value=b_sb)
        return B_out

    @nki.jit
    def ao_to_mo_trn2_kernel(eri, C_occ, C_vir):
        """``B[i, a, P] = Σ_{μ,ν} C_occ[μ,i] C_vir[ν,a] eri[μ,ν,P]`` (native, Trn2).

        Two plain GEMM stages per auxiliary chunk of ``TP`` columns:

        1. ``X[i, ν, Pc] = Σ_μ C_occ[μ, i] · eri[μ, ν, Pc]``: stationary
           ``C_occ`` μ-tile (resident in SBUF), moving ``eri[μ-tile, ν, Pc]``.
           X goes to kernel-scratch HBM, i-major.
        2. ``B[i, a, Pc] = Σ_ν C_vir[ν, a] · X[i, ν, Pc]``: stationary ``C_vir``
           tile (resident), moving ``X[i, ν-tile, Pc]`` (ν on partitions).

        Every load has the contraction index on the partition axis, so there
        are no DMA transposes. The aux chunks are split across the programs
        of the logical core (both physical cores at LNC=2); each program
        reads back only the X it wrote.

        NBASIS, NVIR multiples of 128; NOCC ≤ 128; NAUX a multiple of 128.
        """
        NB, _, NAUX = eri.shape
        TP, NUB = ao_to_mo_trn2_tiling(NAUX)
        return _ao_to_mo_trn2_body(eri, None, C_occ, C_vir, NB, NAUX, TP, NUB)

    @nki.jit
    def ao_to_mo_trn2_packed_kernel(eri_v, img, C_occ, C_vir):
        """:func:`ao_to_mo_trn2_kernel` reading packed μ≥ν integrals in place (#54).

        ``eri_v``: the packed lower triangle ``(npair, NAUX)`` (row
        ``μ(μ+1)/2 + ν`` for ``ν ≤ μ``, PySCF ``lib.pack_tril`` order;
        ``_cderi`` is its transpose), viewed as ``(npair·NPC, TP)`` so each
        aux chunk of a pair is one contiguous row.
        ``img``: ``(NPC, 128, NKB·NB)`` uint32 gather indices built by
        ``dispatch._tril_gather_images``: for chunk ``c``, μ-tile ``k`` and
        ν block ``v0``, the ``(128, NUB)`` block at columns ``k·NB + v0``
        holds, column-major, the rows of ``eri[k·128 + p, v0 + u, chunk c]``
        in ``(p, u)`` row-major order.

        Stage 1 fetches each ``(128, NUB, TP)`` block with one batched
        indirect DMA (vs ``NUB`` strided loads in the full kernel); stage 2
        is unchanged. The full square is never written. Same constraints as
        the full kernel; ``TP``/``NUB`` from ``ao_to_mo_trn2_tiling(...,
        packed=True)``.
        """
        NB = C_occ.shape[0]
        NPC = img.shape[0]
        TP = eri_v.shape[1]
        NUB = ao_to_mo_trn2_nub(TP)
        return _ao_to_mo_trn2_body(eri_v, img, C_occ, C_vir, NB, NPC * TP, TP, NUB)


# ---------------------------------------------------------------------------
# DF exchange (K) from packed integrals (#60)
# ---------------------------------------------------------------------------
#
# K[μ, ν] = Σ_{i, P} X[i, μ, P] X[i, ν, P],  X[i, ν, P] = Σ_λ C[λ, i] B[λν, P],
# with B the packed μ≥ν three-centre integrals. Two launches:
#
# 1. ``df_half_packed_kernel`` gathers packed rows in place (the batched
#    indirect DMA of ``ao_to_mo_trn2_packed_kernel``) and forms the
#    half-transformed ``Xt[P, i, ν]`` with the aux index on partitions: the
#    gathered ``(128 λ, TP P)`` block is the stationary operand and the
#    resident ``C_occ`` λ-tile the moving one, so no transposes are needed.
# 2. ``df_exchange_syrk_kernel`` accumulates ``K = Σ_{P-tile, i} Xt[:, i, :]ᵀ
#    Xt[:, i, :]`` in PSUM: 128 aux rows per contraction, upper-triangle
#    128-row strips only, mirrored on the Tensor Engine on the way out.
#
# Launch 1 splits aux chunks (or ν groups) and launch 2 the occupied index
# across the programs of the logical core, so both physical cores work at
# LNC=2. Each program of launch 2 keeps its own running partial K. Traced
# loops unroll, so the dispatcher runs the pair over groups of aux chunks
# (``df_group_chunks``) to bound compile time and the ``Xt`` scratch.

DF_K_SBUF_BUDGET = 160 * 1024  # bytes per partition for launch 1's buffers
DF_K_LAUNCH_INSTR = 48_000  # estimated instructions per half-transform launch


def df_half_sbuf_bytes(nb: int, nocc: int, tp: int, nub: int, nvc: int) -> int:
    """SBUF bytes per partition for ``df_half_packed_kernel`` (double-buffered)."""
    nkb = nb // TILE_K
    ns = -(-tp // TILE_K)
    return 4 * (nkb * nocc + 2 * nkb * nvc + 2 * nkb * nub * tp + 2 * ns * nocc * nvc)


def df_half_tiling(naux: int, nb: int, nocc: int) -> tuple[int, int, int] | None:
    """``(TP, NUB, NVC)`` for ``df_half_packed_kernel``, or None if nothing fits.

    ``TP`` divides ``naux`` (the packed rows are viewed as ``(npair·NPC, TP)``
    without a padding copy) and is in ``[32, 512]`` (or ``naux`` itself when
    smaller); an ``naux`` with no such divisor gets None. ``NUB`` (ν per gather, a power of
    two ≤ 8) sets the batch of one indirect DMA, ``128·NUB`` rows; ``NVC``
    (a power of two ≤ 128 dividing ``nb``) is the ν run kept in SBUF and
    stored per DMA. Choice, in order: fewest ≤128-row aux tiles
    ``NPC·ceil(TP/128)`` (Tensor Engine work), then the largest gather batch
    (on Trn2 at n_ao = 490, NUB 2 → 4 → 8 took the K build from 19 to 15 to
    12 ms), then the widest ``NVC``, then the widest ``TP``, all within
    ``DF_K_SBUF_BUDGET``.
    """
    best = None
    for tp in range(min(naux, 512), min(naux, 32) - 1, -1):
        if naux % tp:
            continue
        tiles = (naux // tp) * -(-tp // TILE_K)
        for nub in (8, 4, 2, 1):
            if nub * tp > 4096:
                continue
            nvc = 128
            while nvc >= nub and (
                nb % nvc or df_half_sbuf_bytes(nb, nocc, tp, nub, nvc) > DF_K_SBUF_BUDGET
            ):
                nvc //= 2
            if nvc < nub:
                continue
            key = (tiles, -nub, -nvc, -tp)
            if best is None or key < best[0]:
                best = (key, (tp, nub, nvc))
            break
    return None if best is None else best[1]


def df_group_chunks(npc: int, nb: int, tp: int, nub: int, nvc: int) -> int:
    """Aux chunks per launch pair: the most whose estimated instruction count
    (both programs of an LNC=2 core) stays under ``DF_K_LAUNCH_INSTR``.

    The kernels' loops are unrolled when traced, so this bounds the compile
    time (about 2 minutes per 50k instructions on Trn2) and the ``Xt``
    scratch; the dispatcher loops over the groups.
    """
    nkb = nb // TILE_K
    ns = -(-tp // TILE_K)
    per_chunk = (nb // nub) * (nkb + nub * ns * (nkb + 1))
    return max(1, min(npc, DF_K_LAUNCH_INSTR // max(1, per_chunk)))


def df_exchange_blocks(nb: int, banks: int) -> list[list[tuple[int, int, int]]]:
    """Passes of ``(m, ν0, w)`` K blocks for ``df_exchange_syrk_kernel``.

    Strip ``m`` (rows ``128m..128m+127``) covers columns ``ν ≥ 128m`` in
    ≤512-wide blocks (one PSUM bank each); consecutive blocks are grouped
    into passes of at most ``banks`` PSUM banks.
    """
    blocks = []
    for m in range(nb // TILE_K):
        for v0 in range(m * TILE_K, nb, TILE_N):
            blocks.append((m, v0, min(TILE_N, nb - v0)))
    return [blocks[i : i + banks] for i in range(0, len(blocks), banks)]


if HAS_NKI:

    @nki.jit
    def df_half_packed_kernel(eri_v, img, C_occ, nub: int = 8):
        """``Xt[P, i, ν] = Σ_λ C_occ[λ, i] B[λν, P]`` from packed integrals (#60).

        ``eri_v``: packed ``(npair, naux)`` viewed as ``(npair·NPC, TP)``.
        ``img``: ``(G, NVG, 128, NKB·NVC)`` uint32 gather indices for G
        consecutive aux chunks (a slice of ``dispatch._df_gather_images``):
        for chunk ``c`` and ν group ``g``, the ``(128, NUB)`` block at
        columns ``k·NVC + b·NUB`` gathers ``B[k·128 + p, g·NVC + b·NUB + u,
        chunk]`` into ``(p, u)``. ``C_occ``: ``(NB, NOCC)``, NB a multiple
        of 128 (zero rows past nbasis), NOCC ≤ 512. ``nub``: ν per gather,
        the block width the indices were built for.

        Returns ``Xt`` ``(G·TP, NVG, NOCC, NVC)``: ``Xt[P, g, i, v] = X[i, g·NVC
        + v, P]`` for the aux rows of these chunks. Each stored SBUF block is
        one contiguous ``NOCC·NVC`` run per row; with ν innermost the run
        would be only ``NVC`` elements, and at n_ao = 970 (NVC = 16) those
        64-byte DMA writes made this launch 5x slower. Columns ν past
        nbasis hold finite junk that only reaches K's padding.
        """
        NB, NOCC = C_occ.shape
        G, NVG = img.shape[0], img.shape[1]
        TP = eri_v.shape[1]
        TILE = 128
        NKB = NB // TILE
        NVC = NB // NVG
        NUB = nub
        NS = (TP + TILE - 1) // TILE
        n_prog, pid, axis = _programs(G, NVG)
        G_per = G // n_prog if axis == 0 else G
        NVG_per = NVG // n_prog if axis == 1 else NVG

        xt = nl.ndarray((G * TP, NVG, NOCC, NVC), dtype=nl.float32, buffer=nl.shared_hbm)

        co_sb = nl.ndarray((TILE, NKB * NOCC), dtype=C_occ.dtype, buffer=nl.sbuf)
        for k in nl.affine_range(NKB):
            co_sb[0:TILE, k * NOCC : (k + 1) * NOCC] = nl.load(
                C_occ[k * TILE : (k + 1) * TILE, 0:NOCC]
            )

        for pc in nl.affine_range(G_per):
            c = pid * G_per + pc if axis == 0 else pc
            for gi in nl.affine_range(NVG_per):
                g = pid * NVG_per + gi if axis == 1 else gi
                idx_sb = nl.ndarray((TILE, NKB * NVC), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.dma_copy(dst=idx_sb, src=img[c, g])
                x_sb = nl.ndarray((TILE, NS, NOCC, NVC), dtype=nl.float32, buffer=nl.sbuf)
                for vb in nl.affine_range(NVC // NUB):
                    e_blk = nl.ndarray((TILE, NKB, NUB, TP), dtype=eri_v.dtype, buffer=nl.sbuf)
                    for k in nl.affine_range(NKB):
                        nisa.dma_copy(
                            dst=e_blk[0:TILE, k, 0:NUB, 0:TP],
                            src=eri_v.ap(
                                pattern=[[TP, TILE * NUB], [1, TP]],
                                vector_offset=idx_sb[
                                    0:TILE, k * NVC + vb * NUB : k * NVC + vb * NUB + NUB
                                ],
                                indirect_dim=0,
                            ),
                        )
                    for u in nl.affine_range(NUB):
                        for s in range(NS):
                            ms = min(TILE, TP - s * TILE)
                            px = nl.zeros((ms, NOCC), dtype=nl.float32, buffer=nl.psum)
                            for k in nl.affine_range(NKB):
                                nisa.nc_matmul(
                                    dst=px,
                                    stationary=e_blk[0:TILE, k, u, s * TILE : s * TILE + ms],
                                    moving=co_sb[0:TILE, k * NOCC : (k + 1) * NOCC],
                                    accumulate=True,
                                )
                            nisa.tensor_copy(src=px, dst=x_sb[0:ms, s, 0:NOCC, vb * NUB + u])
                for s in range(NS):
                    ms = min(TILE, TP - s * TILE)
                    r0 = c * TP + s * TILE
                    nl.store(xt[r0 : r0 + ms, g, 0:NOCC, 0:NVC], value=x_sb[0:ms, s, 0:NOCC, 0:NVC])
        return xt

    @nki.jit
    def df_exchange_syrk_kernel(xt, kin, banks: int = 6):
        """``out[p] = kin[p] + Σ_{P, i ∈ program p} X[i, :, P] X[i, :, P]ᵀ`` (#60).

        ``xt``: ``(NR, NVG, NOCC, NVC)`` from ``df_half_packed_kernel``; the
        last 128-row aux tile may be partial. ``kin``: ``(2, NB, NB)``
        running partial sums (symmetric), so a chain of launches over aux
        groups accumulates without extra adds; ``K = out[0] + out[1]`` at
        the end.

        Per aux tile and group of ``GI`` occupied orbitals one DMA brings
        ``(rows, ν, GI)`` in ≥1 KiB runs; an engine copy then lays each
        orbital's row out contiguously in ν for the Tensor Engine. PSUM
        holds the K blocks of one pass (``df_exchange_blocks``: the
        upper-triangle 128-row strips in ≤512-wide blocks). Each finished
        strip is stored and its off-diagonal 128×128 tiles are transposed on
        the Tensor Engine into the lower triangle; diagonal tiles are stored
        as ``0.5·(T + Tᵀ)``, so each ``out[p]`` is exactly symmetric. The
        occupied index is split across the programs (NOCC even at LNC=2; the
        dispatcher pads it with a zero orbital); with one program ``out[1] =
        kin[1]``.
        """
        NR, NVG, NOCC, NVC = xt.shape
        NB = NVG * NVC
        TILE = 128
        NTF = NR // TILE
        TAIL = NR - NTF * TILE
        n_prog, pid, _axis = _programs(NOCC)
        NO_per = NOCC // n_prog
        GI = max(1, min(NO_per, 16384 // NB))
        NGF = NO_per // GI
        GR = NO_per - NGF * GI
        out = nl.ndarray((2, NB, NB), dtype=nl.float32, buffer=nl.shared_hbm)
        if n_prog == 1:
            for m in nl.affine_range(NB // TILE):
                kc = nl.ndarray((TILE, NB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=kc, src=kin[1, m * TILE : (m + 1) * TILE, 0:NB])
                nl.store(out[1, m * TILE : (m + 1) * TILE, 0:NB], value=kc)

        for blocks in df_exchange_blocks(NB, banks):
            lo = blocks[0][0] * TILE
            g_lo = lo // NVC
            W = NB - lo
            accs = [
                nl.zeros((TILE, w), dtype=nl.float32, buffer=nl.psum) for (_m, _v0, w) in blocks
            ]
            for r0, nt, ms in ((0, NTF, TILE), (NTF * TILE, 1 if TAIL else 0, TAIL)):
                for t in nl.affine_range(nt):
                    for i_base, ng, gi in ((0, NGF, GI), (NGF * GI, 1 if GR else 0, GR)):
                        for ig in nl.affine_range(ng):
                            i0 = pid * NO_per + i_base + ig * gi
                            xg = nl.ndarray(
                                (ms, NVG - g_lo, gi, NVC), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.dma_copy(
                                dst=xg,
                                src=xt[
                                    r0 + t * TILE : r0 + t * TILE + ms,
                                    g_lo:NVG,
                                    i0 : i0 + gi,
                                    0:NVC,
                                ],
                            )
                            for ii in nl.affine_range(gi):
                                xr = nl.ndarray((ms, W), dtype=nl.float32, buffer=nl.sbuf)
                                nisa.tensor_copy(
                                    dst=xr.reshape_dim(1, (NVG - g_lo, NVC)),
                                    src=xg[0:ms, 0 : NVG - g_lo, ii, 0:NVC],
                                )
                                for b, (m, v0, w) in enumerate(blocks):
                                    a0 = m * TILE - lo
                                    nisa.nc_matmul(
                                        dst=accs[b],
                                        stationary=xr[0:ms, a0 : a0 + TILE],
                                        moving=xr[0:ms, v0 - lo : v0 - lo + w],
                                        accumulate=True,
                                    )
            for b, (m, v0, w) in enumerate(blocks):
                kb = nl.ndarray((TILE, w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=kb, src=kin[pid, m * TILE : (m + 1) * TILE, v0 : v0 + w])
                ks = nl.ndarray((TILE, w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=ks, data1=accs[b], data2=kb, op=nl.add)
                if v0 == m * TILE:
                    # The diagonal tile is computed whole and the Tensor
                    # Engine's fp32 products are not bitwise symmetric in
                    # (μ, ν); store 0.5·(T + Tᵀ) so K is exactly symmetric.
                    pd = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=pd, data=ks[0:TILE, 0:TILE])
                    kd = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=kd, data1=ks[0:TILE, 0:TILE], data2=pd, op=nl.add)
                    nisa.tensor_scalar(
                        dst=ks[0:TILE, 0:TILE], data=kd, op0=nl.multiply, operand0=0.5
                    )
                nl.store(out[pid, m * TILE : (m + 1) * TILE, v0 : v0 + w], value=ks)
                for c0 in range(v0, v0 + w, TILE):
                    if c0 // TILE <= m:
                        continue
                    pt = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=pt, data=ks[0:TILE, c0 - v0 : c0 - v0 + TILE])
                    kt = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(src=pt, dst=kt)
                    nl.store(out[pid, c0 : c0 + TILE, m * TILE : (m + 1) * TILE], value=kt)
        return out


# ---------------------------------------------------------------------------
# DF Coulomb (J) from packed integrals (#60)
# ---------------------------------------------------------------------------
#
# d[P] = Σ_r B[r, P] p[r] and J[r] = Σ_P B[r, P] d[P] over the packed rows
# r = μ(μ+1)/2 + ν. Both stream B once in contiguous 128-row tiles (rows on
# partitions, aux on the free axis) and do the arithmetic as one fused
# multiply-add or multiply-reduce per tile on the Vector Engine, so they run
# at DMA rate. Full row tiles are split evenly across the programs; the
# dispatcher handles the few leftover rows with aten.

DF_J_COLS = 4096  # aux columns per SBUF tile (16 KiB per partition)
DF_J_RED = 128  # aux columns per J partial sum (fewer terms per fp32 chain)
DF_D_GROUP = 32  # row tiles per partial d accumulator

if HAS_NKI:

    @nki.jit
    def df_coulomb_d_kernel(B, pw):
        """``out[p, P] = Σ_{r ∈ program p} B[r, P] pw[r]`` over full 128-row tiles.

        ``B``: packed ``(NR, NAUX)``; ``pw``: ``(NR, 1)`` weighted packed
        density. Uses rows ``[0, NT2·128)``, ``NT2 = 2·(NR // 256)``, at any
        program count; the dispatcher adds the leftover rows. Returns ``(2,
        1, NAUX)`` (``out[1] = 0`` with one program); ``d = out[0] +
        out[1]``.
        """
        NR, NAUX = B.shape
        TILE = 128
        n_prog, pid, _axis = _programs(2)
        NT_per = (NR // (2 * TILE)) * 2 // n_prog
        out = nl.ndarray((2, 1, NAUX), dtype=nl.float32, buffer=nl.shared_hbm)
        if n_prog == 1:
            z = nl.zeros((1, NAUX), dtype=nl.float32, buffer=nl.sbuf)
            nl.store(out[1, 0:1, 0:NAUX], value=z)
        ones = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        NG = NT_per // DF_D_GROUP
        for c0 in range(0, NAUX, DF_J_COLS):
            w = min(DF_J_COLS, NAUX - c0)
            acc = nl.zeros((TILE, w), dtype=nl.float32, buffer=nl.sbuf)
            # Tiles in groups of DF_D_GROUP, each into a fresh accumulator
            # that is then added to ``acc``: shorter fp32 chains (on Trn2 at
            # n_ao = 970 one chain of 1840 tiles gave J 2.4e-6 off fp64).
            for t0, ng, gs in ((0, NG, DF_D_GROUP), (NG * DF_D_GROUP, 1, NT_per - NG * DF_D_GROUP)):
                if gs == 0 or ng == 0:
                    continue
                for tg in nl.sequential_range(ng):
                    accg = nl.zeros((TILE, w), dtype=nl.float32, buffer=nl.sbuf)
                    for tt in nl.sequential_range(gs):
                        r0 = (pid * NT_per + t0 + tg * gs + tt) * TILE
                        bt = nl.ndarray((TILE, w), dtype=B.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(dst=bt, src=B[r0 : r0 + TILE, c0 : c0 + w])
                        pc = nl.ndarray((TILE, 1), dtype=pw.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(dst=pc, src=pw[r0 : r0 + TILE, 0:1])
                        nisa.scalar_tensor_tensor(
                            dst=accg,
                            data=bt,
                            op0=nl.multiply,
                            operand0=pc,
                            op1=nl.add,
                            operand1=accg,
                        )
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=accg, op=nl.add)
            # Sum the 128 partitions on the Tensor Engine.
            dsb = nl.ndarray((1, w), dtype=nl.float32, buffer=nl.sbuf)
            for j0 in range(0, w, TILE_N):
                jw = min(TILE_N, w - j0)
                ps = nl.zeros((1, jw), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=ps, stationary=ones, moving=acc[0:TILE, j0 : j0 + jw], accumulate=True
                )
                nisa.tensor_copy(src=ps, dst=dsb[0:1, j0 : j0 + jw])
            nl.store(out[pid, 0:1, c0 : c0 + w], value=dsb)
        return out

    @nki.jit
    def df_coulomb_j_kernel(B, d_b):
        """``J[r] = Σ_P B[r, P] d[P]`` over full 128-row tiles.

        ``d_b``: ``(128, NAUX)``, ``d`` replicated on every partition. Uses
        the rows of :func:`df_coulomb_d_kernel`. Returns ``(128, NT2)`` with
        ``J[t·128 + p] = out[p, t]``.
        """
        NR, NAUX = B.shape
        TILE = 128
        n_prog, pid, _axis = _programs(2)
        NT2 = (NR // (2 * TILE)) * 2
        NT_per = NT2 // n_prog
        out = nl.ndarray((TILE, NT2), dtype=nl.float32, buffer=nl.shared_hbm)
        db = nl.ndarray((TILE, NAUX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=db, src=d_b[0:TILE, 0:NAUX])
        jall = nl.ndarray((TILE, NT_per), dtype=nl.float32, buffer=nl.sbuf)
        for t in nl.affine_range(NT_per):
            r0 = (pid * NT_per + t) * TILE
            parts = nl.ndarray(
                (TILE, (NAUX + DF_J_RED - 1) // DF_J_RED), dtype=nl.float32, buffer=nl.sbuf
            )
            # One DMA per DF_J_COLS-wide piece of the row tile (long
            # contiguous runs), then one multiply-reduce per DF_J_RED
            # columns (short fp32 chains). 128-column DMAs made this kernel
            # descriptor-bound: 76 ms at n_ao = 970.
            for b0 in range(0, NAUX, DF_J_COLS):
                bw = min(DF_J_COLS, NAUX - b0)
                bt = nl.ndarray((TILE, bw), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=bt, src=B[r0 : r0 + TILE, b0 : b0 + bw])
                prod = nl.ndarray((TILE, bw), dtype=nl.float32, buffer=nl.sbuf)
                for c0 in range(b0, b0 + bw, DF_J_RED):
                    w = min(DF_J_RED, b0 + bw - c0)
                    ci = c0 // DF_J_RED
                    nisa.scalar_tensor_tensor(
                        dst=prod[0:TILE, c0 - b0 : c0 - b0 + w],
                        data=bt[0:TILE, c0 - b0 : c0 - b0 + w],
                        op0=nl.multiply,
                        operand0=1.0,
                        op1=nl.multiply,
                        operand1=db[0:TILE, c0 : c0 + w],
                        reduce_op=nl.add,
                        reduce_res=parts[0:TILE, ci : ci + 1],
                    )
            nisa.tensor_reduce(
                dst=jall[0:TILE, t : t + 1], op=nl.add, data=parts, axis=1, keepdims=True
            )
        nl.store(out[0:TILE, pid * NT_per : (pid + 1) * NT_per], value=jall)
        return out
