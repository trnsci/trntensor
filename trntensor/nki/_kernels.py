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
        eps_occ : (nocc,)           — occupied orbital energies.
        eps_vir : (nvir,)           — virtual orbital energies.

        Returns
        -------
        (nocc, nocc) float32 partial tensor. Host sums to scalar.
        """
        NOCC, NVIR, NAUX = B.shape
        partial = nl.ndarray((NOCC, NOCC), dtype=nl.float32, buffer=nl.shared_hbm)

        # Pre-load eps_vir once; reused for every (i, j).
        ev = nl.load(eps_vir[0:NVIR])  # shape (NVIR,)

        for i in nl.affine_range(NOCC):
            eo_i = nl.load(eps_occ[i : i + 1])  # (1,)
            # Load Bi once per i, transposed so partition dim = NAUX.
            # Shape becomes (NAUX, NVIR) in SBUF.
            Bi_t = nl.load_transpose2d(B[i, 0:NVIR, 0:NAUX])

            for j in nl.affine_range(NOCC):
                eo_j = nl.load(eps_occ[j : j + 1])  # (1,)
                eo_sum = nl.add(eo_i, eo_j)  # (1,)

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
                denom_rows = nl.subtract(eo_sum, ev.reshape((NVIR, 1)))  # (NVIR, 1)
                denom = nl.subtract(denom_rows, ev.reshape((1, NVIR)))  # (NVIR, NVIR)

                # term = T * (2T - T.T) / Δ. NKI 0.3.0 drops tensor-tensor
                # nl.divide; use multiply × reciprocal.
                two_t_minus_tT = nl.subtract(nl.multiply(t, 2.0), t_T)
                numerator = nl.multiply(t, two_t_minus_tT)
                term = nl.multiply(numerator, nl.reciprocal(denom))

                # Reduce to scalar and store.
                e_ij = nl.sum(term, axis=(0, 1))
                nl.store(partial[i : i + 1, j : j + 1], value=e_ij)

        return partial
