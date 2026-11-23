"""NKI kernels for the equivariant contractions (trnsci/trntensor#138).

Four kernels, one launch per direction per layer:

``conv_tp_fwd_kernel`` / ``conv_tp_bwd_kernel`` (A1, e3nn "uvu" tensor product)
    Edges on the 128 partitions, the channels ``u`` on the free axis. Each
    tile of 128 edges loads its rows of ``x``, ``y``, ``w`` (and ``g``) with one
    contiguous DMA per operand, computes every per-edge coupling block

        G[e, (p, i, k)] = Σ_j y[e, j] · α_p C_p[i, j, k]

    with one Tensor Engine matmul against the block-CG matrix ``cg`` (kept
    in SBUF for the whole kernel; ``[16, 88]`` at MACE-MP layer 1), and then
    works per path in SBUF:

        out_p[:, u, k] = Σ_i (x_i ⊙ w_p)[:, u] · G[:, (p, i, k)]

    ``G[:, (p, i, k)]`` is one value per edge, i.e. a per-partition scalar,
    so every term is a single ``[128, U]`` instruction: the ``x_i ⊙ w_p``
    products on GpSimd, the first term of each ``k`` on the Scalar Engine
    (``activation(copy, scale=G column)``) and the accumulating terms on the
    Vector Engine (``scalar_tensor_tensor``), so the three engines overlap.
    e3nn's mul-major layouts (``[U, 2l+1]`` per irrep block) are read and
    written through strided SBUF views; HBM traffic stays one contiguous row
    per edge (20 KiB per edge for the layer-1 output) and nothing but the
    inputs and the outputs ever touches HBM. The layer-1 forward is 110
    such instructions per 128 edges.

    The backward recomputes ``G`` (cheaper than storing it) and forms, per
    path and input component,

        h_{p,i} = Σ_k g_{p,k} · G[:, (p, i, k)]      (Scalar + Vector, per-partition scalars)
        ∂w_p   += x_i ⊙ h_{p,i}                     (GpSimd / Vector)
        ∂x_i   += w_p ⊙ h_{p,i}
        ∂G[:, (p, i, k)] = Σ_u (x_i ⊙ w_p) ⊙ g_{p,k}  (Vector, fused free-axis reduction)

    and finally ``∂y = ∂G @ cgᵀ``: a Tensor Engine transpose of ``∂G`` and one
    matmul accumulating in PSUM. This is the 128-channel reduction of the
    contract, done on the free axis by ``scalar_tensor_tensor``'s fused
    reduction instead of a transposed copy of the cotangent.

    SBUF per partition (fp32, MACE-MP layer 1): forward ``x`` 2 KiB, ``w`` 5 KiB,
    ``out`` 20 KiB, ``G`` 0.35 KiB, temporaries ≤ 2 KiB, ≈ 30 KiB per tile;
    backward ≈ 40 KiB per tile. Both leave room for the tile framework's
    multi-buffering in the 192 KiB partition. PSUM: ``G`` (≤ 512 columns per
    matmul) and the ``y`` / ``∂G`` transposes, one bank each. Edge counts are
    padded by the host to a multiple of 256 (and to a coarse bucket, see
    ``_equivariant.edge_bucket``) so both programs of an LNC=2 core get the
    same number of tiles; padded edges are zero and contribute zero.

``symcon_fwd_kernel`` / ``symcon_bwd_kernel`` (A2, MACE symmetric contraction)
    Rows ``r = (node, channel)`` on the free axis, 512 per block (4 nodes of
    128 channels, one PSUM bank of fp32), and the small axes on partitions:
    the 16 node features, the ``A`` symmetric monomials (410 at layer 0), the
    ``MK`` coupling columns ``(irrep, m, ν, k)`` (202 at layer 0) and the
    ``Σ d_out`` outputs (4 at layer 0). Every contraction is a Tensor Engine
    matmul accumulating in PSUM, with all constants resident in SBUF:

        xᵀ_aug [17, 512]   = Tensor Engine transposes of the 4 node tiles, plus a ones row
        X_j    [α, 512]    = sel_jᵀ @ xᵀ_aug     (0/1 selection of factor j of monomial α)
        mono   [α, 512]    = X_0 ⊙ X_1 ⊙ X_2      (Vector Engine)
        P      [mk, 512]  += usymᵀ @ mono         (PSUM accumulation over monomial chunks)
        out    [d, 512]   += Sᵀ @ (P ⊙ W_rows)    (W_rows: the per-node weight rows W[z_n])

    The backward runs the same chain in reverse (``Sᵀ``-expand the cotangent,
    multiply by ``W_rows``, ``usym``-contract to monomials, multiply by the
    other two factors, ``sel``-contract to features) and transposes ``∂xᵀ``
    back per node. Species enter only as the host-side row gather
    ``W_rows = W_mk[species]`` (one aten ``index_select``); nothing in the
    kernel depends on index values. PSUM: three selection products, the
    ``P`` accumulators (≤ 2 banks for MK ≤ 256), the output and a transpose:
    7 of the 8 banks. SBUF: the resident constants are < 16 KiB per partition.

Every kernel takes its structure as static arguments (``plan`` tuples or
shapes), never as closure or global state (see tests/test_trace_cache_safety.py).
"""

from __future__ import annotations

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

# Edge tile: one edge per partition.
CONV_TP_TILE = 128
# PSUM matmul output width (fp32 columns per bank).
CONV_TP_Q_CHUNK = 512
# Symmetric contraction: rows (node, channel) per block (one fp32 PSUM bank).
SYMCON_ROWS = 512
SYMCON_PART = 128
# Per-partition SBUF budget the host checks before choosing a kernel (bytes).
EQUIVARIANT_SBUF_BUDGET = 96 * 1024


def conv_tp_plan_paths(plan) -> list[tuple[int, int, int, int, int, int]]:
    """``plan`` → ``[(x_off, d1, q_off, d3, out_off, accumulate), ...]``."""
    body = plan[1:]
    return [tuple(body[i : i + 6]) for i in range(0, len(body), 6)]


def conv_tp_sbuf_bytes(d1: int, d2: int, pw: int, dout: int, q: int, u: int) -> int:
    """Per-partition SBUF bytes of one backward tile (the larger direction)."""
    return 4 * (dout + 2 * d1 + d2 + 2 * pw + 2 * q + 6 * u + q)


def symcon_psum_banks(mk: int) -> int:
    """PSUM banks the symmetric-contraction forward keeps live per block."""
    return 3 + -(-mk // SYMCON_PART) + 2


if HAS_NKI:

    def _programs2(n_tiles):
        """(n_prog, pid) splitting ``n_tiles`` over the logical core's programs."""
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        if n_prog > 1 and n_tiles % n_prog == 0:
            return n_prog, nl.program_id(0)
        return 1, 0

    def _coupling_blocks(yt, cg_sb, D2, Q):
        """``G = y_tile @ cg`` for one edge tile: [128, Q] in SBUF (fp32)."""
        T = CONV_TP_TILE
        yT_ps = nl.ndarray((D2, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=yT_ps, data=yt)
        yT = nl.ndarray((D2, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=yT, src=yT_ps)
        G = nl.ndarray((T, Q), dtype=nl.float32, buffer=nl.sbuf)
        for q0 in range(0, Q, CONV_TP_Q_CHUNK):
            qw = min(CONV_TP_Q_CHUNK, Q - q0)
            g_ps = nl.ndarray((T, qw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=g_ps, stationary=yT, moving=cg_sb[0:D2, q0 : q0 + qw])
            nisa.tensor_copy(dst=G[0:T, q0 : q0 + qw], src=g_ps)
        return G

    @nki.jit
    def conv_tp_fwd_kernel(x, y, w, cg, plan, zero_fill: int = 0):
        """A1 forward: ``out [E, Dout]`` for padded ``x [E, D1]``, ``y [E, D2]``,
        ``w [E, P·U]`` and the compact block-CG ``cg [D2, Q]`` (α folded).

        ``plan = (Dout, x_off, d1, q_off, d3, out_off, accumulate, ...)``, six
        entries per path in weight order. ``E`` is a multiple of 128 (of 256 for
        an even split at LNC=2). ``zero_fill`` zeroes output blocks no path writes.
        """
        E, D1 = x.shape
        _, D2 = y.shape
        _, PW = w.shape
        _, Q = cg.shape
        T = CONV_TP_TILE
        Dout = plan[0]
        paths = conv_tp_plan_paths(plan)
        U = PW // len(paths)
        out = nl.ndarray((E, Dout), dtype=x.dtype, buffer=nl.shared_hbm)
        cg_sb = nl.ndarray((D2, Q), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cg_sb, src=cg[0:D2, 0:Q])
        n_tiles = E // T
        n_prog, pid = _programs2(n_tiles)
        NT = n_tiles // n_prog
        for t in nl.affine_range(NT):
            e0 = (pid * NT + t) * T
            xt = nl.ndarray((T, D1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[e0 : e0 + T, 0:D1])
            yt = nl.ndarray((T, D2), dtype=y.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=yt, src=y[e0 : e0 + T, 0:D2])
            wt = nl.ndarray((T, PW), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt, src=w[e0 : e0 + T, 0:PW])
            G = _coupling_blocks(yt, cg_sb, D2, Q)
            ot = nl.ndarray((T, Dout), dtype=x.dtype, buffer=nl.sbuf)
            if zero_fill:
                nisa.memset(dst=ot, value=0.0)
            for p, (x_off, d1, q_off, d3, o_off, acc) in enumerate(paths):
                xv = xt[0:T, x_off : x_off + U * d1].reshape((T, U, d1))
                ov = ot[0:T, o_off : o_off + U * d3].reshape((T, U, d3))
                wp = wt[0:T, p * U : (p + 1) * U]
                xw = []
                for i in range(d1):
                    xwi = nl.ndarray((T, U), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=xwi,
                        data1=xv[0:T, 0:U, i],
                        data2=wp,
                        op=nl.multiply,
                        engine=nisa.engine.gpsimd,
                    )
                    xw.append(xwi)
                for k in range(d3):
                    for i in range(d1):
                        q = q_off + i * d3 + k
                        if i == 0 and not acc:
                            nisa.activation(
                                dst=ov[0:T, 0:U, k], op=nl.copy, data=xw[0], scale=G[0:T, q : q + 1]
                            )
                        else:
                            nisa.scalar_tensor_tensor(
                                dst=ov[0:T, 0:U, k],
                                data=xw[i],
                                op0=nl.multiply,
                                operand0=G[0:T, q : q + 1],
                                op1=nl.add,
                                operand1=ov[0:T, 0:U, k],
                            )
            nisa.dma_copy(dst=out[e0 : e0 + T, 0:Dout], src=ot)
        return out

    @nki.jit
    def conv_tp_bwd_kernel(
        g, x, y, w, cg, cgT, plan, need_dx: int = 1, need_dy: int = 1, need_dw: int = 1
    ):
        """A1 backward: ``[∂x | ∂y | ∂w]``, one ``[E, D1 + D2 + P·U]`` output, for
        the cotangent ``g [E, Dout]``. Operands as :func:`conv_tp_fwd_kernel`,
        plus ``cgT = cgᵀ [Q, D2]``. A gradient whose ``need_*`` flag is 0 is not
        computed (its columns are left unwritten; the host returns None for it).
        One output keeps the op schema single-tensor, like every other
        trntensor op.
        """
        E, D1 = x.shape
        _, D2 = y.shape
        _, PW = w.shape
        _, Q = cg.shape
        T = CONV_TP_TILE
        Dout = plan[0]
        paths = conv_tp_plan_paths(plan)
        U = PW // len(paths)
        grads = nl.ndarray((E, D1 + D2 + PW), dtype=x.dtype, buffer=nl.shared_hbm)
        cg_sb = nl.ndarray((D2, Q), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cg_sb, src=cg[0:D2, 0:Q])
        n_qc = (Q + T - 1) // T
        cgT_sb = nl.ndarray((T, n_qc, D2), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(n_qc):
            qw = min(T, Q - c * T)
            nisa.dma_copy(dst=cgT_sb[0:qw, c, 0:D2], src=cgT[c * T : c * T + qw, 0:D2])
        # Static bookkeeping: which (x block, component) gradients a path writes first.
        first_dx = {}
        for p, (x_off, d1, _q, _d3, _o, _a) in enumerate(paths):
            for i in range(d1):
                first_dx.setdefault((x_off, i), p)
        covered = sum(d1 * U for (_x_off, d1) in {(pp[0], pp[1]) for pp in paths})
        n_tiles = E // T
        n_prog, pid = _programs2(n_tiles)
        NT = n_tiles // n_prog
        for t in nl.affine_range(NT):
            e0 = (pid * NT + t) * T
            gt = nl.ndarray((T, Dout), dtype=g.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gt, src=g[e0 : e0 + T, 0:Dout])
            xt = nl.ndarray((T, D1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[e0 : e0 + T, 0:D1])
            yt = nl.ndarray((T, D2), dtype=y.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=yt, src=y[e0 : e0 + T, 0:D2])
            wt = nl.ndarray((T, PW), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt, src=w[e0 : e0 + T, 0:PW])
            G = _coupling_blocks(yt, cg_sb, D2, Q)
            dxt = nl.ndarray((T, D1), dtype=x.dtype, buffer=nl.sbuf)
            if need_dx and covered < D1:
                nisa.memset(dst=dxt, value=0.0)
            dwt = nl.ndarray((T, PW), dtype=w.dtype, buffer=nl.sbuf)
            dG = nl.ndarray((T, Q), dtype=nl.float32, buffer=nl.sbuf)
            for p, (x_off, d1, q_off, d3, o_off, _acc) in enumerate(paths):
                xv = xt[0:T, x_off : x_off + U * d1].reshape((T, U, d1))
                gv = gt[0:T, o_off : o_off + U * d3].reshape((T, U, d3))
                wp = wt[0:T, p * U : (p + 1) * U]
                dwp = dwt[0:T, p * U : (p + 1) * U]
                dxv = dxt[0:T, x_off : x_off + U * d1].reshape((T, U, d1))
                for i in range(d1):
                    if need_dx or need_dw:
                        # h = sum_k g_k * G[:, (p, i, k)]
                        h = nl.ndarray((T, U), dtype=nl.float32, buffer=nl.sbuf)
                        q = q_off + i * d3
                        nisa.activation(
                            dst=h, op=nl.copy, data=gv[0:T, 0:U, 0], scale=G[0:T, q : q + 1]
                        )
                        for k in range(1, d3):
                            nisa.scalar_tensor_tensor(
                                dst=h,
                                data=gv[0:T, 0:U, k],
                                op0=nl.multiply,
                                operand0=G[0:T, q + k : q + k + 1],
                                op1=nl.add,
                                operand1=h,
                            )
                        if need_dw:
                            if i == 0:
                                nisa.tensor_tensor(
                                    dst=dwp, data1=xv[0:T, 0:U, i], data2=h, op=nl.multiply
                                )
                            else:
                                tw = nl.ndarray((T, U), dtype=nl.float32, buffer=nl.sbuf)
                                nisa.tensor_tensor(
                                    dst=tw,
                                    data1=xv[0:T, 0:U, i],
                                    data2=h,
                                    op=nl.multiply,
                                    engine=nisa.engine.gpsimd,
                                )
                                nisa.tensor_tensor(dst=dwp, data1=dwp, data2=tw, op=nl.add)
                        if need_dx:
                            if first_dx[(x_off, i)] == p:
                                nisa.tensor_tensor(
                                    dst=dxv[0:T, 0:U, i], data1=wp, data2=h, op=nl.multiply
                                )
                            else:
                                tx = nl.ndarray((T, U), dtype=nl.float32, buffer=nl.sbuf)
                                nisa.tensor_tensor(
                                    dst=tx,
                                    data1=wp,
                                    data2=h,
                                    op=nl.multiply,
                                    engine=nisa.engine.gpsimd,
                                )
                                nisa.tensor_tensor(
                                    dst=dxv[0:T, 0:U, i],
                                    data1=dxv[0:T, 0:U, i],
                                    data2=tx,
                                    op=nl.add,
                                )
                    if need_dy:
                        xw = nl.ndarray((T, U), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(
                            dst=xw,
                            data1=xv[0:T, 0:U, i],
                            data2=wp,
                            op=nl.multiply,
                            engine=nisa.engine.gpsimd,
                        )
                        junk = nl.ndarray((T, U), dtype=nl.float32, buffer=nl.sbuf)
                        for k in range(d3):
                            q = q_off + i * d3 + k
                            nisa.scalar_tensor_tensor(
                                dst=junk,
                                data=xw,
                                op0=nl.multiply,
                                operand0=1.0,
                                op1=nl.multiply,
                                operand1=gv[0:T, 0:U, k],
                                reduce_op=nl.add,
                                reduce_res=dG[0:T, q : q + 1],
                            )
            if need_dy:
                dy_ps = nl.zeros((T, D2), dtype=nl.float32, buffer=nl.psum)
                for c in range(n_qc):
                    qw = min(T, Q - c * T)
                    dGT_ps = nl.ndarray((qw, T), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=dGT_ps, data=dG[0:T, c * T : c * T + qw])
                    dGT = nl.ndarray((qw, T), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dGT, src=dGT_ps)
                    nisa.nc_matmul(
                        dst=dy_ps, stationary=dGT, moving=cgT_sb[0:qw, c, 0:D2], accumulate=True
                    )
                dyt = nl.ndarray((T, D2), dtype=y.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dyt, src=dy_ps)
                nisa.dma_copy(dst=grads[e0 : e0 + T, D1 : D1 + D2], src=dyt)
            if need_dx:
                nisa.dma_copy(dst=grads[e0 : e0 + T, 0:D1], src=dxt)
            if need_dw:
                nisa.dma_copy(dst=grads[e0 : e0 + T, D1 + D2 : D1 + D2 + PW], src=dwt)
        return grads

    def _symcon_load_xT(x, n0, NB, C, L):
        """``[x_aug]ᵀ`` for the ``NB`` nodes from ``n0``: [L+1, NB·C], last row ones."""
        xT = nl.ndarray((L + 1, NB * C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=xT, value=1.0)
        for q in range(NB):
            xt = nl.ndarray((C, L), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[n0 + q, 0:C, 0:L])
            tp = nl.ndarray((L, C), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=xt)
            nisa.tensor_copy(dst=xT[0:L, q * C : (q + 1) * C], src=tp)
        return xT

    def _symcon_load_w(wr, n0, NB, C, mk0, mw):
        wt = nl.ndarray((mw, NB * C), dtype=nl.float32, buffer=nl.sbuf)
        for q in range(NB):
            nisa.dma_copy(dst=wt[0:mw, q * C : (q + 1) * C], src=wr[n0 + q, mk0 : mk0 + mw, 0:C])
        return wt

    def _symcon_factors(sel_sb, xT, L, a0, aw, RB, ja):
        """The three factor rows of monomials ``a0 : a0+aw`` (SBUF copies)."""
        Xs = []
        for j in range(3):
            Xp = nl.ndarray((aw, RB), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=Xp, stationary=sel_sb[0 : L + 1, j, a0 : a0 + aw], moving=xT)
            Xj = nl.ndarray((aw, RB), dtype=nl.float32, buffer=nl.sbuf)
            if j == 1:
                nisa.activation(dst=Xj, op=nl.copy, data=Xp)
            else:
                nisa.tensor_copy(dst=Xj, src=Xp)
            Xs.append(Xj)
        return Xs

    @nki.jit
    def symcon_fwd_kernel(x, wr, sel, usym, s):
        """A2 forward: ``out [D, N·C]`` (rows m-major; the host reorders to MACE's
        layout) for ``x [N, C, L]``, per-node weight rows ``wr [N, MK, C]``,
        selections ``sel [L+1, 3, A]``, folded couplings ``usym [A, MK]`` and the
        reduction ``s [MK, D]``. ``N·C`` is a multiple of 512 (and of 1024 for an
        even LNC=2 split); ``C`` divides 512 and is ≤ 128.
        """
        N, C, L = x.shape
        _, MK, _ = wr.shape
        A = usym.shape[0]
        D = s.shape[1]
        P = SYMCON_PART
        RB = SYMCON_ROWS
        NB = RB // C
        nA = (A + P - 1) // P
        nM = (MK + P - 1) // P
        out = nl.ndarray((D, N * C), dtype=nl.float32, buffer=nl.shared_hbm)
        sel_sb = nl.ndarray((L + 1, 3, A), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sel_sb, src=sel[0 : L + 1, 0:3, 0:A])
        usym_sb = nl.ndarray((P, nA, MK), dtype=nl.float32, buffer=nl.sbuf)
        for ja in range(nA):
            aw = min(P, A - ja * P)
            nisa.dma_copy(dst=usym_sb[0:aw, ja, 0:MK], src=usym[ja * P : ja * P + aw, 0:MK])
        s_sb = nl.ndarray((P, nM, D), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(nM):
            mw = min(P, MK - i * P)
            nisa.dma_copy(dst=s_sb[0:mw, i, 0:D], src=s[i * P : i * P + mw, 0:D])
        n_blk = (N * C) // RB
        n_prog, pid = _programs2(n_blk)
        NBK = n_blk // n_prog
        for b in nl.affine_range(NBK):
            r0 = (pid * NBK + b) * RB
            n0 = r0 // C
            xT = _symcon_load_xT(x, n0, NB, C, L)
            Pp = []
            for i in range(nM):
                mw = min(P, MK - i * P)
                Pp.append(nl.zeros((mw, RB), dtype=nl.float32, buffer=nl.psum))
            for ja in range(nA):
                a0 = ja * P
                aw = min(P, A - a0)
                X0, X1, X2 = _symcon_factors(sel_sb, xT, L, a0, aw, RB, ja)
                m01 = nl.ndarray((aw, RB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=m01, data1=X0, data2=X1, op=nl.multiply, engine=nisa.engine.gpsimd
                )
                mono = nl.ndarray((aw, RB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=mono, data1=m01, data2=X2, op=nl.multiply)
                for i in range(nM):
                    mw = min(P, MK - i * P)
                    nisa.nc_matmul(
                        dst=Pp[i],
                        stationary=usym_sb[0:aw, ja, i * P : i * P + mw],
                        moving=mono,
                        accumulate=True,
                    )
            o_ps = nl.zeros((D, RB), dtype=nl.float32, buffer=nl.psum)
            for i in range(nM):
                mw = min(P, MK - i * P)
                wt = _symcon_load_w(wr, n0, NB, C, i * P, mw)
                qi = nl.ndarray((mw, RB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=qi, data1=Pp[i], data2=wt, op=nl.multiply)
                nisa.nc_matmul(dst=o_ps, stationary=s_sb[0:mw, i, 0:D], moving=qi, accumulate=True)
            o_sb = nl.ndarray((D, RB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_sb, src=o_ps)
            nisa.dma_copy(dst=out[0:D, r0 : r0 + RB], src=o_sb)
        return out

    @nki.jit
    def symcon_bwd_kernel(g, x, wr, sel, selT, usymT, sT):
        """A2 backward: ``∂x [N, C, L]`` for the m-major cotangent ``g [D, N·C]``.

        ``selT [A, 3, L+1]``, ``usymT [MK, A]`` and ``sT [D, MK]`` are the
        transposed constants of :func:`symcon_fwd_kernel`.
        """
        N, C, L = x.shape
        _, MK, _ = wr.shape
        A = usymT.shape[1]
        D = sT.shape[0]
        P = SYMCON_PART
        RB = SYMCON_ROWS
        NB = RB // C
        nA = (A + P - 1) // P
        nM = (MK + P - 1) // P
        dx = nl.ndarray((N, C, L), dtype=x.dtype, buffer=nl.shared_hbm)
        sel_sb = nl.ndarray((L + 1, 3, A), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sel_sb, src=sel[0 : L + 1, 0:3, 0:A])
        selT_sb = nl.ndarray((P, nA, 3, L + 1), dtype=nl.float32, buffer=nl.sbuf)
        for ja in range(nA):
            aw = min(P, A - ja * P)
            nisa.dma_copy(
                dst=selT_sb[0:aw, ja, 0:3, 0 : L + 1],
                src=selT[ja * P : ja * P + aw, 0:3, 0 : L + 1],
            )
        usymT_sb = nl.ndarray((P, nM, A), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(nM):
            mw = min(P, MK - i * P)
            nisa.dma_copy(dst=usymT_sb[0:mw, i, 0:A], src=usymT[i * P : i * P + mw, 0:A])
        sT_sb = nl.ndarray((D, MK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sT_sb, src=sT[0:D, 0:MK])
        n_blk = (N * C) // RB
        n_prog, pid = _programs2(n_blk)
        NBK = n_blk // n_prog
        for b in nl.affine_range(NBK):
            r0 = (pid * NBK + b) * RB
            n0 = r0 // C
            xT = _symcon_load_xT(x, n0, NB, C, L)
            gt = nl.ndarray((D, RB), dtype=g.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gt, src=g[0:D, r0 : r0 + RB])
            GW = []
            for i in range(nM):
                mw = min(P, MK - i * P)
                ps = nl.ndarray((mw, RB), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps, stationary=sT_sb[0:D, i * P : i * P + mw], moving=gt)
                wt = _symcon_load_w(wr, n0, NB, C, i * P, mw)
                gw = nl.ndarray((mw, RB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=gw, data1=ps, data2=wt, op=nl.multiply)
                GW.append(gw)
            dxT_ps = nl.zeros((L + 1, RB), dtype=nl.float32, buffer=nl.psum)
            for ja in range(nA):
                a0 = ja * P
                aw = min(P, A - a0)
                X0, X1, X2 = _symcon_factors(sel_sb, xT, L, a0, aw, RB, ja)
                R_ps = nl.zeros((aw, RB), dtype=nl.float32, buffer=nl.psum)
                for i in range(nM):
                    mw = min(P, MK - i * P)
                    nisa.nc_matmul(
                        dst=R_ps,
                        stationary=usymT_sb[0:mw, i, a0 : a0 + aw],
                        moving=GW[i],
                        accumulate=True,
                    )
                R = nl.ndarray((aw, RB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=R, op=nl.copy, data=R_ps)
                for j, (fa, fb) in enumerate(((X1, X2), (X0, X2), (X0, X1))):
                    tj = nl.ndarray((aw, RB), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=tj, data1=fa, data2=fb, op=nl.multiply, engine=nisa.engine.gpsimd
                    )
                    dj = nl.ndarray((aw, RB), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=dj, data1=tj, data2=R, op=nl.multiply)
                    nisa.nc_matmul(
                        dst=dxT_ps,
                        stationary=selT_sb[0:aw, ja, j, 0 : L + 1],
                        moving=dj,
                        accumulate=True,
                    )
            dxT = nl.ndarray((L, RB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dxT, src=dxT_ps[0:L, 0:RB])
            for q in range(NB):
                tps = nl.ndarray((C, L), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=dxT[0:L, q * C : (q + 1) * C])
                dxs = nl.ndarray((C, L), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dxs, src=tps)
                nisa.dma_copy(dst=dx[n0 + q, 0:C, 0:L], src=dxs)
        return dx
