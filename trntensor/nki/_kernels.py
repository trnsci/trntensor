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
    def matmul_kernel(a, b):
        """``C = A @ B`` with stationary-A tile reuse.

        Caller guarantees M, K are multiples of ``TILE_M``/``TILE_K`` and N
        is either ≤ ``TILE_N`` or a clean multiple of ``TILE_N``. PSUM
        accumulates over K-tiles before the single store per (m, n) pair.
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
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(
                    c[m_off : m_off + tile_m, n_off : n_off + tile_n],
                    value=c_sbuf,
                )

        return c

    @nki.jit
    def batched_matmul_kernel(a, b):
        """Batched ``C[b] = A[b] @ B[b]`` — one matmul per batch slice.

        Caller pads (M, K, N) the same way as ``matmul_kernel``. The
        batch dim is iterated with ``nl.affine_range``; each slice
        reuses the stationary-A tile layout.
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
                    nisa.tensor_copy(src=psum, dst=c_sbuf)
                    nl.store(
                        c[batch, m_off : m_off + tile_m, n_off : n_off + tile_n],
                        value=c_sbuf,
                    )

        return c

    @nki.jit
    def mp2_energy_kernel(B, eps_occ, eps_vir):
        """Fully fused DF-MP2 correlation energy.

        For each ``(i, j)`` pair, this program:

        1. computes ``T_ab = Σ_P B[i,a,P] B[j,b,P]`` via ``nc_matmul`` —
           result lives in PSUM, never hits HBM;
        2. computes ``T^T_ab = T_ba`` via a second ``nc_matmul`` with
           the operand roles swapped, also in PSUM;
        3. builds the MP2 denominator ``Δ_ab = ε_i + ε_j - ε_a - ε_b``
           on the Vector Engine from SBUF-resident ε tiles;
        4. evaluates ``term = T * (2T - T^T) / Δ`` element-wise in
           SBUF;
        5. folds ``Σ_ab term`` into a scalar SBUF accumulator — one
           HBM store per (i, j) instead of one per ab tile.

        The whole energy summation is one NKI program — one dispatch,
        one HBM round-trip for the final partial array. Assumes
        ``nvir ≤ 128`` and ``naux ≤ 128`` (single-tile path). Larger
        cases will need K/M tiling; not in this kernel.

        Inputs
        ------
        B       : (nocc, nvir, naux) — DF coefficients.
        eps_occ : (nocc, 1)         — occupied orbital energies (2D for NKI load hygiene).
        eps_vir : (nvir, 1)         — virtual orbital energies (2D for NKI load hygiene).

        Returns
        -------
        (nocc, nocc) float32 partial tensor. Host sums to scalar.
        """
        NOCC, NVIR, NAUX = B.shape
        partial = nl.ndarray((NOCC, NOCC), dtype=nl.float32, buffer=nl.shared_hbm)

        # Pre-load eps_vir once; reused for every (i, j). Loaded as 2D
        # (NVIR, 1) so partition dim is unambiguous to the NKI compiler
        # regardless of whether eps_vir arrived fresh-from-CPU or was
        # pre-pinned on the XLA device. See #38.
        ev = nl.load(eps_vir[0:NVIR, 0:1])  # shape (NVIR, 1)

        for i in nl.affine_range(NOCC):
            eo_i = nl.load(eps_occ[i : i + 1, 0:1])  # (1, 1)
            # Load Bi once per i, transposed so partition dim = NAUX.
            # Shape becomes (NAUX, NVIR) in SBUF.
            Bi_t = nl.load_transpose2d(B[i, 0:NVIR, 0:NAUX])

            for j in nl.affine_range(NOCC):
                eo_j = nl.load(eps_occ[j : j + 1, 0:1])  # (1, 1)
                eo_sum = nl.add(eo_i, eo_j)  # (1, 1)

                # Bj for this (i,j). Load transposed → (NAUX, NVIR).
                Bj_t = nl.load_transpose2d(B[j, 0:NVIR, 0:NAUX])

                # T = Bi @ Bj.T   — stationary Bi_t, moving Bj_t.
                # T.T = Bj @ Bi.T — swap the roles.
                psum_T = nl.zeros((NVIR, NVIR), dtype=nl.float32, buffer=nl.psum)
                psum_Tt = nl.zeros((NVIR, NVIR), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=psum_T, stationary=Bi_t, moving=Bj_t, accumulate=True)
                nisa.nc_matmul(dst=psum_Tt, stationary=Bj_t, moving=Bi_t, accumulate=True)

                # PSUM → SBUF: NKI 0.3.0 requires an explicit tensor_copy
                # into a fresh SBUF tile for downstream Vector Engine ops.
                t = nl.ndarray((NVIR, NVIR), dtype=B.dtype, buffer=nl.sbuf)
                t_T = nl.ndarray((NVIR, NVIR), dtype=B.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_T, dst=t)
                nisa.tensor_copy(src=psum_Tt, dst=t_T)

                # Δ_ab = (ε_i + ε_j) - ε_a - ε_b
                # Build the (NVIR, NVIR) denominator on the Vector Engine.
                # ev is already (NVIR, 1); reshape to (1, NVIR) for the second step.
                denom_rows = nl.subtract(eo_sum, ev)  # broadcasts (1,1) - (NVIR,1) → (NVIR,1)
                denom = nl.subtract(denom_rows, ev.reshape((1, NVIR)))  # (NVIR, NVIR)

                # term = T * (2T - T.T) / Δ. NKI 0.3.0 drops tensor-tensor
                # nl.divide; use multiply × reciprocal.
                two_t_minus_tT = nl.subtract(nl.multiply(t, 2.0), t_T)
                numerator = nl.multiply(t, two_t_minus_tT)
                term = nl.multiply(numerator, nl.reciprocal(denom))

                # Reduce to scalar via a persistent (1, 1) SBUF accumulator.
                # NKI rejects 0-D SBUF tiles, so we can't allocate one
                # directly from nl.sum; instead broadcast the 0-D sum into
                # a (1, 1) tile via nl.add. Matches trnblas's pattern.
                acc = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                acc[...] = nl.add(acc, nl.sum(term, axis=(0, 1)))
                nl.store(partial[i : i + 1, j : j + 1], value=acc)

        return partial

    @nki.jit
    def ao_to_mo_transform_kernel(eri, C_occ, C_vir):
        """Fused 4-index AO→MO integral transform with K-tiling over the basis index.

        Computes ``B[i, a, P] = Σ_{μ,ν} C_occ[μ, i] · C_vir[ν, a] · eri[μ, ν, P]``
        as one NKI program. Per auxiliary index P:

        1. **Step 1 matmul** (K-tiled over μ): ``intermediate(i, ν) = Σ_μ C_occ(μ,i) · eri(μ,ν)``
           — tiles over μ in chunks of ``TILE_K=128`` (the NKI partition-dim limit);
           PSUM accumulates the partial sums across tiles; result (NOCC, NBASIS) written
           to kernel-scratch HBM.
        2. **Step 2 matmul** (K-tiled over ν): ``B(i, a) = Σ_ν intermediate(ν,i) · C_vir(ν,a)``
           — tiles over ν in the same way; intermediate reloaded transposed so the
           ν chunk is the partition dim; result (NOCC, NVIR) written to B_out.

        The two-step fused structure means the per-P intermediate ``(i, ν)`` never
        appears as a user-visible tensor — it lives in kernel-scratch HBM between the
        two matmul steps.

        Constraints (enforced at the dispatch boundary, not here):
        - NBASIS must be a multiple of ``TILE_K``; dispatch pads if necessary.
        - NBASIS ≤ 512 (moving free dim of step-1 eri tile, the ν side).
        - NOCC ≤ 128 (stationary free dim = M dimension of both nc_matmul calls).
        - NVIR ≤ 512 (moving free dim of step-2 C_vir tile).

        Inputs
        ------
        eri   : (nbasis, nbasis, naux) — AO-basis density-fitted ERIs; nbasis is a
                multiple of TILE_K (dispatcher pads).
        C_occ : (nbasis, nocc)         — occupied MO coefficients.
        C_vir : (nbasis, nvir)         — virtual MO coefficients.

        Returns
        -------
        (nocc, nvir, naux) tensor of transformed coefficients B_ia^P.
        """
        NBASIS, _, NAUX = eri.shape
        _, NOCC = C_occ.shape
        _, NVIR = C_vir.shape

        B_out = nl.ndarray((NOCC, NVIR, NAUX), dtype=eri.dtype, buffer=nl.shared_hbm)
        # Kernel-scratch HBM: holds the post-step-1 intermediate (NOCC, NBASIS) for
        # each P. Reloaded in TILE_K-wide ν slices (transposed) in step 2.
        inter_hbm = nl.ndarray((NOCC, NBASIS, NAUX), dtype=eri.dtype, buffer=nl.shared_hbm)

        for p in nl.affine_range(NAUX):
            # ------------------------------------------------------------------
            # Step 1: intermediate(i, ν) = Σ_μ C_occ(μ,i) · eri(μ,ν,P)
            # K-tile over μ (partition dim ≤ TILE_K per nc_matmul call).
            # psum_1 accumulates the (NOCC, NBASIS) partial sums across μ-tiles.
            # ------------------------------------------------------------------
            psum_1 = nl.zeros((NOCC, NBASIS), dtype=nl.float32, buffer=nl.psum)
            for k_mu in nl.affine_range(NBASIS // TILE_K):
                k_off = k_mu * TILE_K
                # C_occ tile: (TILE_K, NOCC) — partition=μ_chunk, free=i
                C_occ_tile = nl.load(C_occ[k_off : k_off + TILE_K, 0:NOCC])
                # eri tile: (TILE_K, NBASIS) — partition=μ_chunk, free=ν
                eri_tile = nl.load(eri[k_off : k_off + TILE_K, 0:NBASIS, p])
                nisa.nc_matmul(dst=psum_1, stationary=C_occ_tile, moving=eri_tile, accumulate=True)

            # PSUM → SBUF → kernel-scratch HBM. Stored as (NOCC, NBASIS) so step 2
            # can reload ν-slices transposed to get partition=ν_chunk.
            inter_sbuf = nl.ndarray((NOCC, NBASIS), dtype=eri.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(src=psum_1, dst=inter_sbuf)
            nl.store(inter_hbm[0:NOCC, 0:NBASIS, p], value=inter_sbuf)

            # ------------------------------------------------------------------
            # Step 2: B(i, a) = Σ_ν intermediate(ν,i) · C_vir(ν,a)
            # K-tile over ν. Reload TILE_K-wide slices of the intermediate
            # transposed so partition=ν_chunk, free=i.
            # ------------------------------------------------------------------
            psum_2 = nl.zeros((NOCC, NVIR), dtype=nl.float32, buffer=nl.psum)
            for k_nu in nl.affine_range(NBASIS // TILE_K):
                k_off = k_nu * TILE_K
                # Load (NOCC, TILE_K) slice and transpose → (TILE_K, NOCC)
                # so partition=ν_chunk, free=i.
                inter_tile = nl.load_transpose2d(inter_hbm[0:NOCC, k_off : k_off + TILE_K, p])
                # C_vir tile: (TILE_K, NVIR) — partition=ν_chunk, free=a
                C_vir_tile = nl.load(C_vir[k_off : k_off + TILE_K, 0:NVIR])
                nisa.nc_matmul(
                    dst=psum_2, stationary=inter_tile, moving=C_vir_tile, accumulate=True
                )

            out_sbuf = nl.ndarray((NOCC, NVIR), dtype=eri.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(src=psum_2, dst=out_sbuf)
            nl.store(B_out[0:NOCC, 0:NVIR, p], value=out_sbuf)

        return B_out
