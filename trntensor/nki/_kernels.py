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

        When ``use_sr=True``, the PSUM→SBUF downcast uses
        ``nisa.activation(..., round_mode="stochastic")`` instead of
        ``nisa.tensor_copy``. This is a correctness requirement for BF16
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
                    nisa.activation(
                        psum,
                        bias=None,
                        scale=1.0,
                        out_dtype=a.dtype,
                        round_mode="stochastic",
                        dst=c_sbuf,
                    )
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
        reuses the stationary-A tile layout. ``use_sr`` applies
        stochastic rounding at the PSUM→SBUF downcast (see
        ``matmul_kernel`` for the correctness motivation).
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
                        nisa.activation(
                            psum,
                            bias=None,
                            scale=1.0,
                            out_dtype=a.dtype,
                            round_mode="stochastic",
                            dst=c_sbuf,
                        )
                    else:
                        nisa.tensor_copy(src=psum, dst=c_sbuf)
                    nl.store(
                        c[batch, m_off : m_off + tile_m, n_off : n_off + tile_n],
                        value=c_sbuf,
                    )

        return c

    @nki.jit
    def mp2_energy_kernel(B, eps_occ, eps_vir):
        """Fully fused DF-MP2 correlation energy with K/M tiling.

        Lifts the single-tile ``nvir ≤ 128`` / ``naux ≤ 128`` constraints.
        NVIR and NAUX must be multiples of ``TILE_M`` / ``TILE_K``; the
        dispatch layer pads both before entry.

        For each ``(i, j)`` pair this program tiles over the ``(a, b)``
        virtual-orbital block (M-tiling, TILE_M=128 per side) and over the
        auxiliary index P (K-tiling, TILE_K=128):

        1. For each ``(m_tile, n_tile)`` block:
           K-accumulate ``T[m,n] = Σ_P B[i,m,P] B[j,n,P]`` and
           ``T.T[m,n] = Σ_P B[j,m,P] B[i,n,P]`` in PSUM;
        2. PSUM → SBUF; build the ``(TILE_M, TILE_M)`` denominator
           ``Δ_ab = ε_i + ε_j - ε_a - ε_b`` on the Vector Engine;
        3. Fold ``Σ_ab T*(2T-T.T)/Δ`` into the ``(1,1)`` pair accumulator.

        One HBM store per ``(i, j)`` — same as the original single-tile kernel.

        T.T tile correctness:
        ``nc_matmul(stationary=Bj_m, moving=Bi_n)[da, db]``
        ``= Σ_P B[j, m+da, P] · B[i, n+db, P]``
        ``= T_{ji}[m+da, n+db]``
        ``= T_{ij}[n+db, m+da]``  (scalar product commutes)
        ``= T.T[m+da, n+db]`` ✓

        Inputs
        ------
        B       : (nocc, nvir, naux) — DF coefficients; nvir % TILE_M == 0,
                                       naux % TILE_K == 0 (dispatcher pads).
        eps_occ : (nocc, 1)         — occupied orbital energies (2D for NKI
                                       load hygiene regardless of residency).
        eps_vir : (nvir, 1)         — virtual orbital energies (2D, same).

        Returns
        -------
        (nocc, nocc) float32 partial tensor. Host sums to scalar.
        """
        NOCC, NVIR, NAUX = B.shape
        partial = nl.ndarray((NOCC, NOCC), dtype=nl.float32, buffer=nl.shared_hbm)

        for i in nl.affine_range(NOCC):
            eo_i = nl.load(eps_occ[i : i + 1, 0:1])  # (1, 1)

            for j in nl.affine_range(NOCC):
                eo_j = nl.load(eps_occ[j : j + 1, 0:1])  # (1, 1)
                eo_sum = nl.add(eo_i, eo_j)  # (1, 1)

                # Persistent (1,1) SBUF accumulator for this (i,j) pair.
                acc = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)

                for m_tile in nl.affine_range(NVIR // TILE_M):
                    m_off = m_tile * TILE_M
                    # eps_vir slice for the a-dimension; reused for all n_tile.
                    ev_m = nl.load(eps_vir[m_off : m_off + TILE_M, 0:1])  # (TILE_M, 1)

                    for n_tile in nl.affine_range(NVIR // TILE_M):
                        n_off = n_tile * TILE_M
                        ev_n = nl.load(eps_vir[n_off : n_off + TILE_M, 0:1])  # (TILE_M, 1)

                        psum_T = nl.zeros((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.psum)
                        psum_Tt = nl.zeros((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.psum)

                        for k_p in nl.affine_range(NAUX // TILE_K):
                            p_off = k_p * TILE_K
                            # T[m,n] = Σ_P B[i,m,P] * B[j,n,P]
                            # load_transpose2d: (TILE_M, TILE_K) → (TILE_K, TILE_M) in SBUF
                            Bi_m = nl.load_transpose2d(
                                B[i, m_off : m_off + TILE_M, p_off : p_off + TILE_K]
                            )
                            Bj_n = nl.load_transpose2d(
                                B[j, n_off : n_off + TILE_M, p_off : p_off + TILE_K]
                            )
                            nisa.nc_matmul(
                                dst=psum_T, stationary=Bi_m, moving=Bj_n, accumulate=True
                            )
                            # T.T[m,n] = T[n,m] = Σ_P B[j,m,P] * B[i,n,P]
                            Bj_m = nl.load_transpose2d(
                                B[j, m_off : m_off + TILE_M, p_off : p_off + TILE_K]
                            )
                            Bi_n = nl.load_transpose2d(
                                B[i, n_off : n_off + TILE_M, p_off : p_off + TILE_K]
                            )
                            nisa.nc_matmul(
                                dst=psum_Tt, stationary=Bj_m, moving=Bi_n, accumulate=True
                            )

                        # PSUM → SBUF: NKI 0.3.0 requires an explicit tensor_copy
                        # into a fresh SBUF tile for downstream Vector Engine ops.
                        t = nl.ndarray((TILE_M, TILE_M), dtype=B.dtype, buffer=nl.sbuf)
                        t_T = nl.ndarray((TILE_M, TILE_M), dtype=B.dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(src=psum_T, dst=t)
                        nisa.tensor_copy(src=psum_Tt, dst=t_T)

                        # Δ_ab = (ε_i + ε_j) - ε_a - ε_b for this (m_tile, n_tile) block.
                        # ev_m: (TILE_M, 1) — a-dimension energies
                        # ev_n: (TILE_M, 1) — b-dimension energies
                        denom_rows = nl.subtract(eo_sum, ev_m)  # (TILE_M, 1)
                        denom = nl.subtract(
                            denom_rows, ev_n.reshape((1, TILE_M))
                        )  # (TILE_M, TILE_M)

                        # term = T * (2T - T.T) / Δ. NKI 0.3.0 drops tensor-tensor
                        # nl.divide; use multiply × reciprocal.
                        two_t_minus_tT = nl.subtract(nl.multiply(t, 2.0), t_T)
                        numerator = nl.multiply(t, two_t_minus_tT)
                        term = nl.multiply(numerator, nl.reciprocal(denom))
                        acc[...] = nl.add(acc, nl.sum(term, axis=(0, 1)))

                nl.store(partial[i : i + 1, j : j + 1], value=acc)

        return partial

    @nki.jit
    def ao_to_mo_transform_kernel(eri, C_occ, C_vir):
        """Fused 4-index AO→MO integral transform with K-tiling over μ/ν and
        N-tiling over the ν (step 1) and virtual-orbital a (step 2) dimensions.

        Computes ``B[i, a, P] = Σ_{μ,ν} C_occ[μ, i] · C_vir[ν, a] · eri[μ, ν, P]``
        as one NKI program. Per auxiliary index P:

        1. **Step 1** (K-tiled over μ, N-tiled over ν):
           ``intermediate(i, ν) = Σ_μ C_occ(μ,i) · eri(μ,ν,P)``
           — ν is partitioned into ``TILE_N=512``-wide tiles; within each ν-tile, μ
           is K-tiled in ``TILE_K=128`` chunks. PSUM per ν-tile; result
           ``(NOCC, TILE_N)`` written to kernel-scratch HBM.
        2. **Step 2** (K-tiled over ν, N-tiled over a):
           ``B(i, a) = Σ_ν intermediate(ν,i) · C_vir(ν,a)``
           — a is partitioned into ``TILE_N``-wide tiles; ν is K-tiled as before.
           Result ``(NOCC, TILE_N)`` written to B_out.

        The intermediate lives in kernel-scratch HBM between the two steps;
        it is never a user-visible tensor.

        Constraints (enforced at the dispatch boundary, not here):
        - NBASIS must be a multiple of ``TILE_N`` (=512); dispatcher pads.
          (Implies TILE_K alignment since TILE_N = 4×TILE_K.)
        - NVIR must be a multiple of ``TILE_N``; dispatcher pads and slices.
        - NOCC ≤ 128 (stationary M dim of both nc_matmul calls).

        Inputs
        ------
        eri   : (nbasis, nbasis, naux) — AO-basis density-fitted ERIs;
                nbasis is a multiple of TILE_N (dispatcher pads).
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
        # Kernel-scratch HBM: holds the post-step-1 intermediate (NOCC, NBASIS) for
        # each P. Written in TILE_N-wide ν-chunks; reloaded in TILE_K-wide slices.
        inter_hbm = nl.ndarray((NOCC, NBASIS, NAUX), dtype=eri.dtype, buffer=nl.shared_hbm)

        for p in nl.affine_range(NAUX):
            # ------------------------------------------------------------------
            # Step 1: intermediate(i, ν) = Σ_μ C_occ(μ,i) · eri(μ,ν,P)
            # N-tiled over ν (TILE_N=512 chunks); K-tiled over μ (TILE_K=128).
            # psum_1 accumulates one (NOCC, TILE_N) ν-block at a time.
            # ------------------------------------------------------------------
            for v_tile in nl.affine_range(NBASIS // TILE_N):
                v_off = v_tile * TILE_N
                psum_1 = nl.zeros((NOCC, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k_mu in nl.affine_range(NBASIS // TILE_K):
                    k_off = k_mu * TILE_K
                    # C_occ tile: (TILE_K, NOCC) — partition=μ_chunk, free=i
                    C_occ_tile = nl.load(C_occ[k_off : k_off + TILE_K, 0:NOCC])
                    # eri tile: (TILE_K, TILE_N) — partition=μ_chunk, free=ν_chunk
                    eri_tile = nl.load(eri[k_off : k_off + TILE_K, v_off : v_off + TILE_N, p])
                    nisa.nc_matmul(
                        dst=psum_1, stationary=C_occ_tile, moving=eri_tile, accumulate=True
                    )
                # PSUM → SBUF → kernel-scratch HBM for this ν-chunk.
                inter_sbuf = nl.ndarray((NOCC, TILE_N), dtype=eri.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_1, dst=inter_sbuf)
                nl.store(inter_hbm[0:NOCC, v_off : v_off + TILE_N, p], value=inter_sbuf)

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
                    # Load (NOCC, TILE_K) slice and transpose → (TILE_K, NOCC)
                    # so partition=ν_chunk, free=i.
                    inter_tile = nl.load_transpose2d(inter_hbm[0:NOCC, k_off : k_off + TILE_K, p])
                    # C_vir tile: (TILE_K, TILE_N) — partition=ν_chunk, free=a_chunk
                    C_vir_tile = nl.load(C_vir[k_off : k_off + TILE_K, a_off : a_off + TILE_N])
                    nisa.nc_matmul(
                        dst=psum_2, stationary=inter_tile, moving=C_vir_tile, accumulate=True
                    )
                out_sbuf = nl.ndarray((NOCC, TILE_N), dtype=eri.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_2, dst=out_sbuf)
                nl.store(B_out[0:NOCC, a_off : a_off + TILE_N, p], value=out_sbuf)

        return B_out
