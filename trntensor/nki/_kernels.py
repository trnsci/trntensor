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
