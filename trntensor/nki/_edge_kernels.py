"""NKI kernels for the per-edge MLIP ops (trnsci/trnsci#178).

Both ops are elementwise in the edges, so the edges are laid out
*partition-major*: the host pads ``E`` to ``128·M`` and views each per-edge
array ``[E, F]`` as ``[128, M·F]`` (a free view), so partition ``p`` holds the
``M`` consecutive edges ``p·M … p·M + M − 1``. Every elementwise step is then
one instruction over a ``[128, J]`` plane (``J`` columns of edges at a time),
not over ``[128, 1]`` columns, and every DMA moves a contiguous run per
partition. Column ``j`` of a plane is a tile of 128 edges with one edge per
partition, which is the layout the matmuls of the edge chain need.

``sh_fwd_kernel`` / ``sh_bwd_kernel`` (spherical harmonics, ``l ≤ 3``)
    The unit vector, its degree-≤3 monomials and every output component are
    planes: ``Y_k = c0_k + Σ_α C[α, k] m_α`` is a short chain of
    ``tensor_scalar`` / ``scalar_tensor_tensor`` instructions per component
    (zero coefficients are skipped at trace time). The backward forms
    ``∂Y/∂n`` from the derivative tables the same way, projects out the
    radial part and scales by ``1 / max(|v|, eps)``. Coefficient tables are
    static arguments.

``edge_chain_fwd_kernel`` / ``edge_chain_bwd_kernel`` (radial basis + MLPs)
    Per chunk of ``J`` columns: the cutoff ``c(r)``, the Agnesi transform
    ``x(r)`` and the Bessel basis ``B_n = prefactor · sin(w_n x) / x`` as
    planes on the Vector and Scalar engines. ``sin`` runs as
    ``activation(sin, scale=−w_n, bias=π)``, i.e. ``sin(π − w_n x)``, which
    keeps the argument inside the Scalar Engine's accurate ``[−π, π]`` range
    for ``0 < w_n x ≤ 2π`` (the host routes only specs that satisfy it: with
    the Agnesi transform ``x ≤ 1``). Then per group of four columns (512
    edges, one PSUM bank) the radial MLPs run on the Tensor Engine with the
    features on partitions: the basis tile is transposed once, each hidden
    layer is one matmul ``[h_out, 512] = Wᵀ a`` with ``silu`` applied on the way
    out of PSUM, and the last layer is one matmul per 128-edge column with the
    activations stationary, so its result lands edges-on-partitions, the
    layout of the output rows. Every weight matrix stays in SBUF for the
    whole kernel (one packed ``[128, Σ cols]`` operand, 8.7 KiB per partition
    for MACE-MP-0b2 medium).

    The backward recomputes the chain (cheaper than storing it), transposes
    the cotangent of each head to the features-on-partitions layout, runs
    the MLPs backwards (``silu_dx`` on the Scalar Engine, one matmul per layer
    against the transposed weights), accumulates ``∂E/∂(basis)`` of all heads
    in PSUM, and reduces ``Σ_n ∂E/∂f_n · ∂f_n/∂r`` on the Vector Engine. The
    single output is ``∂E/∂r``.

Every kernel takes its structure as static arguments (``list[int]`` /
``list[float]``), never as closure or global state (see
tests/test_trace_cache_safety.py).
"""

from __future__ import annotations

import math

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

EDGE_PART = 128
# Columns (edges per partition) per chunk, and per MLP group (one PSUM bank).
EDGE_CHUNK = 32
EDGE_GROUP = 4
# PSUM matmul output width (fp32 columns per bank).
EDGE_PSUM_COLS = 512

POST_CODES = {None: 0, "tanh_sq": 1}


def edge_heads(dims):
    """``dims = [n_heads, (n_layers, post, h_0, …, h_L) per head]`` → ``[(post, (h_0, …, h_L)), …]``."""
    dims = tuple(int(d) for d in dims)
    heads, i = [], 1
    for _ in range(dims[0]):
        n, post = dims[i], dims[i + 1]
        heads.append((post, dims[i + 2 : i + 3 + n]))
        i += 3 + n
    return heads


def edge_pack_layout(dims):
    """Column layout of the packed weights.

    Forward pack ``[128, WF]``: matrix ``W_l [h_{l-1}, h_l]`` of each head in
    rows ``0:h_{l-1}``, columns ``fwd[h][l] : + h_l``.

    Backward pack ``[128, WB]``: ``W_lᵀ [h_l, h_{l-1}]`` cut into 128-row chunks
    placed side by side (chunk ``c`` at ``bwd[h][l] + c·h_{l-1}``); a
    single-layer head with one output stores its row ``W_1ᵀ`` replicated
    over the 128 partitions instead (``[128, h_0]``, for a per-partition
    scalar multiply).
    """
    fwd, bwd, cf, cb = [], [], 0, 0
    for _post, hs in edge_heads(dims):
        f, b = [], []
        for ell in range(1, len(hs)):
            f.append(cf)
            cf += hs[ell]
            b.append(cb)
            if len(hs) == 2 and hs[1] == 1:
                cb += hs[0]
            else:
                cb += -(-hs[ell] // EDGE_PART) * hs[ell - 1]
        fwd.append(f)
        bwd.append(b)
    return fwd, cf, bwd, cb


def edge_out_offsets(dims):
    offs, o = [], 0
    for _post, hs in edge_heads(dims):
        offs.append(o)
        o += hs[-1]
    return offs, o


def edge_chain_sbuf_bytes(dims, nb):
    """Per-partition SBUF bytes of one edge-chain backward chunk (the larger direction)."""
    _f, wf, _b, wb = edge_pack_layout(dims)
    _o, do = edge_out_offsets(dims)
    # weights, cotangent / output / head tiles per group, chunk planes, MLP activations
    return 4 * (
        wf + wb + 3 * EDGE_GROUP * do + EDGE_CHUNK * (4 * nb + 40) + 10 * EDGE_PART * EDGE_GROUP
    )


def _cols_per_program(M, n_prog):
    return M // n_prog if (n_prog > 1 and M % n_prog == 0) else M


if HAS_NKI:

    def _programs(M):
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        if n_prog > 1 and M % n_prog == 0:
            return n_prog, nl.program_id(0)
        return 1, 0

    def _plane(J):
        return nl.ndarray((EDGE_PART, J), dtype=nl.float32, buffer=nl.sbuf)

    def _ts(src, op0, a0, op1=None, a1=None):
        out = _plane(src.shape[1])
        nisa.tensor_scalar(dst=out, data=src, op0=op0, operand0=a0, op1=op1, operand1=a1)
        return out

    def _tt(a, b, op):
        out = _plane(a.shape[1])
        nisa.tensor_tensor(dst=out, data1=a, data2=b, op=op)
        return out

    def _act(src, op, scale=1.0, bias=None):
        out = _plane(src.shape[1])
        nisa.activation(dst=out, op=op, data=src, scale=scale, bias=bias)
        return out

    def _lincomb(dst, terms, planes, const):
        """``dst = const + Σ coef · planes[i]`` over ``terms = ((i, coef), …)``."""
        if not terms:
            nisa.memset(dst=dst, value=float(const))
            return
        (i0, c0), rest = terms[0], terms[1:]
        if const:
            nisa.tensor_scalar(
                dst=dst,
                data=planes[i0],
                op0=nl.multiply,
                operand0=float(c0),
                op1=nl.add,
                operand1=float(const),
            )
        else:
            nisa.tensor_scalar(dst=dst, data=planes[i0], op0=nl.multiply, operand0=float(c0))
        for i, c in rest:
            nisa.scalar_tensor_tensor(
                dst=dst,
                data=planes[i],
                op0=nl.multiply,
                operand0=float(c),
                op1=nl.add,
                operand1=dst,
            )

    # ── spherical harmonics ──────────────────────────────────────────────

    def _sh_unit(vt, J, normalize, eps):
        """Planes ``nx, ny, nz``, ``inv = 1/max(|v|, eps)`` and ``|v|²``."""
        comps = [vt[0:EDGE_PART, 0:J, c] for c in range(3)]
        if not normalize:
            return [_ts(c, nl.multiply, 1.0) for c in comps], None, None
        sq = nl.ndarray((EDGE_PART, J, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=vt, data2=vt, op=nl.multiply)
        r2 = _plane(J)
        nisa.tensor_reduce(dst=r2, op=nl.add, data=sq, axis=2)
        # max(|v|, eps) = sqrt(max(|v|², eps²)): keeps sqrt off 0 (its accurate
        # range starts at 2^-116; padded edges are zero vectors)
        rc = _act(_ts(r2, nl.maximum, float(eps) * float(eps)), nl.sqrt)
        inv = _plane(J)
        nisa.reciprocal(dst=inv, data=rc)
        return [_tt(c, inv, nl.multiply) for c in comps], inv, r2

    def _sh_monomials(n, deg):
        """Planes of the monomials of degree 1..deg (order of ``edge._monomial_exps``)."""
        x, y, z = n
        m = [x, y, z]
        if deg >= 2:
            m2 = [
                _tt(a, b, nl.multiply) for a, b in ((x, x), (x, y), (x, z), (y, y), (y, z), (z, z))
            ]
            m += m2
            if deg >= 3:
                xx, xy, xz, yy, yz, zz = m2
                m += [
                    _tt(a, b, nl.multiply)
                    for a, b in (
                        (xx, x),
                        (xx, y),
                        (xx, z),
                        (xy, y),
                        (xy, z),
                        (xz, z),
                        (yy, y),
                        (yy, z),
                        (yz, z),
                        (zz, z),
                    )
                ]
        return m

    @nki.jit
    def sh_fwd_kernel(v, lmax: int, normalize: int, eps: float, table):
        """Spherical harmonics ``[128, M·D]`` of ``v [128, M·3]`` (partition-major edges).

        ``table = c0 (D) + C (M_mono × D, row-major)`` from ``edge.sh_coefficients``.
        """
        P = EDGE_PART
        _, M3 = v.shape
        M = M3 // 3
        D = (lmax + 1) * (lmax + 1)
        NM = len(table) // D - 1
        c0 = table[0:D]
        C = table[D:]
        out = nl.ndarray((P, M * D), dtype=v.dtype, buffer=nl.shared_hbm)
        n_prog, pid = _programs(M)
        MP = _cols_per_program(M, n_prog)
        J = min(MP, EDGE_CHUNK)
        for j0 in range(0, MP, J):
            Jc = min(J, MP - j0)
            col = pid * MP + j0
            vt = nl.ndarray((P, Jc, 3), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=vt, src=v[0:P, col * 3 : (col + Jc) * 3])
            ot = nl.ndarray((P, Jc, D), dtype=nl.float32, buffer=nl.sbuf)
            if lmax == 0:
                nisa.memset(dst=ot, value=float(c0[0]))
            else:
                n, _inv, _r = _sh_unit(vt, Jc, normalize, eps)
                mono = _sh_monomials(n, lmax)
                for k in range(D):
                    terms = tuple((a, C[a * D + k]) for a in range(NM) if C[a * D + k] != 0.0)
                    _lincomb(ot[0:P, 0:Jc, k], terms, mono, c0[k])
            nisa.dma_copy(dst=out[0:P, col * D : (col + Jc) * D], src=ot)
        return out

    @nki.jit
    def sh_bwd_kernel(g, v, lmax: int, normalize: int, eps: float, dtable):
        """``∂/∂v [128, M·3]`` for the cotangent ``g [128, M·D]``.

        ``dtable = dC (D × 3 × (1 + M'), row-major)`` from ``edge.sh_coefficients``.
        """
        P = EDGE_PART
        _, M3 = v.shape
        M = M3 // 3
        D = (lmax + 1) * (lmax + 1)
        NL = len(dtable) // (3 * D)
        dv = nl.ndarray((P, M * 3), dtype=v.dtype, buffer=nl.shared_hbm)
        n_prog, pid = _programs(M)
        MP = _cols_per_program(M, n_prog)
        J = min(MP, EDGE_CHUNK)
        for j0 in range(0, MP, J):
            Jc = min(J, MP - j0)
            col = pid * MP + j0
            vt = nl.ndarray((P, Jc, 3), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=vt, src=v[0:P, col * 3 : (col + Jc) * 3])
            gt = nl.ndarray((P, Jc, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gt, src=g[0:P, col * D : (col + Jc) * D])
            dvt = nl.ndarray((P, Jc, 3), dtype=nl.float32, buffer=nl.sbuf)
            if lmax == 0:
                nisa.memset(dst=dvt, value=0.0)
            else:
                n, inv, r = _sh_unit(vt, Jc, normalize, eps)
                low = [None] + (_sh_monomials(n, lmax - 1) if lmax >= 2 else [])
                gk = [gt[0:P, 0:Jc, k] for k in range(D)]
                gn = []
                for c in range(3):
                    acc = None
                    for b in range(NL):
                        terms = tuple(
                            (k, dtable[(k * 3 + c) * NL + b])
                            for k in range(D)
                            if dtable[(k * 3 + c) * NL + b] != 0.0
                        )
                        if not terms:
                            continue
                        h = _plane(Jc)
                        _lincomb(h, terms, gk, 0.0)
                        if b > 0:
                            h = _tt(h, low[b], nl.multiply)
                        if acc is None:
                            acc = h
                        else:
                            nisa.tensor_tensor(dst=acc, data1=acc, data2=h, op=nl.add)
                    if acc is None:
                        acc = _plane(Jc)
                        nisa.memset(dst=acc, value=0.0)
                    gn.append(acc)
                if not normalize:
                    for c in range(3):
                        nisa.tensor_copy(dst=dvt[0:P, 0:Jc, c], src=gn[c])
                else:
                    # dv = (gn − [r ≥ eps] (gn·n) n) / max(r, eps)
                    s = _tt(gn[0], n[0], nl.multiply)
                    for c in (1, 2):
                        t = _tt(gn[c], n[c], nl.multiply)
                        nisa.tensor_tensor(dst=s, data1=s, data2=t, op=nl.add)
                    mask = _ts(r, nl.greater_equal, float(eps) * float(eps))  # r holds |v|²
                    nisa.tensor_tensor(dst=s, data1=s, data2=mask, op=nl.multiply)
                    for c in range(3):
                        t = _tt(s, n[c], nl.multiply)
                        nisa.tensor_tensor(dst=t, data1=gn[c], data2=t, op=nl.subtract)
                        nisa.tensor_tensor(
                            dst=dvt[0:P, 0:Jc, c], data1=t, data2=inv, op=nl.multiply
                        )
            nisa.dma_copy(dst=dv[0:P, col * 3 : (col + Jc) * 3], src=dvt)
        return dv

    # ── edge chain ───────────────────────────────────────────────────────

    def _radial_planes(rt, r0t, consts, w, J, want_grad):
        """Planes of the radial chain for ``J`` columns.

        ``consts = [r_max, prefactor, p, a, q, pp, apply_cutoff]``. Returns
        ``c``, ``rx = 1/x``, the basis sines ``S [128, J, nb]`` (``sin(w_n x)``)
        and, with ``want_grad``, ``dc/dr``, ``dx/dr`` and ``C [128, J, nb]``
        (``w_n cos(w_n x)``).
        """
        r_max, _pf, p, a, q, pp = (float(v) for v in consts[0:6])
        nb = len(w)
        pi = int(round(p))
        u = _ts(rt, nl.multiply, 1.0 / r_max)
        mask = _ts(rt, nl.less, r_max)
        # u^(p-1), u^p by repeated products (integer p >= 1)
        upm1 = None
        for _ in range(pi - 1):
            upm1 = _ts(u, nl.multiply, 1.0) if upm1 is None else _tt(upm1, u, nl.multiply)
        up = u if upm1 is None else _tt(upm1, u, nl.multiply)
        # c = (1 − u^p (A − Bc u + Cc u²)) · mask
        A, Bc, Cc = (p + 1.0) * (p + 2.0) / 2.0, p * (p + 2.0), p * (p + 1.0) / 2.0
        t = _ts(u, nl.multiply, Cc, nl.add, -Bc)
        t = _tt(t, u, nl.multiply)
        t = _ts(t, nl.add, A)
        t = _tt(t, up, nl.multiply)
        c = _ts(t, nl.multiply, -1.0, nl.add, 1.0)
        nisa.tensor_tensor(dst=c, data1=c, data2=mask, op=nl.multiply)
        # Agnesi: x = 1 / (1 + N), N = a t^q / (1 + t^(q−pp)), t = r / r0
        rr0 = _plane(J)
        nisa.reciprocal(dst=rr0, data=r0t)
        tt = _tt(rt, rr0, nl.multiply)
        lt = _act(tt, nl.log)
        s1 = _act(lt, nl.exp, scale=q)
        s2 = _act(lt, nl.exp, scale=q - pp)
        Dn = _ts(s2, nl.add, 1.0)
        rD = _plane(J)
        nisa.reciprocal(dst=rD, data=Dn)
        N = _plane(J)
        nisa.scalar_tensor_tensor(
            dst=N, data=s1, op0=nl.multiply, operand0=a, op1=nl.multiply, operand1=rD
        )
        xp1 = _ts(N, nl.add, 1.0)
        x = _plane(J)
        nisa.reciprocal(dst=x, data=xp1)
        rx = xp1  # 1 / x = 1 + N
        S = nl.ndarray((EDGE_PART, J, nb), dtype=nl.float32, buffer=nl.sbuf)
        for n in range(nb):
            # sin(w x) = sin(π − w x), argument in [−π, π] for 0 < w x ≤ 2π
            nisa.activation(
                dst=S[0:EDGE_PART, 0:J, n], op=nl.sin, data=x, scale=-float(w[n]), bias=math.pi
            )
        res = {"c": c, "x": x, "rx": rx, "S": S, "mask": mask}
        if want_grad:
            # dc/dr = −K u^(p−1) (1 − u)² · mask / r_max, K = p(p+1)(p+2)/2
            K = p * (p + 1.0) * (p + 2.0) / 2.0
            omu = _ts(u, nl.multiply, -1.0, nl.add, 1.0)
            dc = _tt(omu, omu, nl.multiply)
            if upm1 is not None:
                nisa.tensor_tensor(dst=dc, data1=dc, data2=upm1, op=nl.multiply)
            nisa.scalar_tensor_tensor(
                dst=dc,
                data=dc,
                op0=nl.multiply,
                operand0=-K / r_max,
                op1=nl.multiply,
                operand1=mask,
            )
            # dx/dr = x² (N / r) ((q − pp) s2 / D − q)
            k2 = _tt(s2, rD, nl.multiply)
            k2 = _ts(k2, nl.multiply, q - pp, nl.add, -q)
            nisa.tensor_tensor(dst=k2, data1=k2, data2=N, op=nl.multiply)
            rr = _plane(J)
            nisa.reciprocal(dst=rr, data=rt)
            nisa.tensor_tensor(dst=k2, data1=k2, data2=rr, op=nl.multiply)
            x2 = _tt(x, x, nl.multiply)
            nisa.tensor_tensor(dst=k2, data1=k2, data2=x2, op=nl.multiply)
            # cos(w x) = 1 − 2 sin²(w x / 2), with sin(w x / 2) = sin(π − w x / 2)
            Cs = nl.ndarray((EDGE_PART, J, nb), dtype=nl.float32, buffer=nl.sbuf)
            for n in range(nb):
                h = _act(x, nl.sin, scale=-0.5 * float(w[n]), bias=math.pi)
                nisa.activation(dst=h, op=nl.square, data=h)
                # w_n cos(w_n x) = w_n − 2 w_n sin²(w_n x / 2)
                nisa.tensor_scalar(
                    dst=Cs[0:EDGE_PART, 0:J, n],
                    data=h,
                    op0=nl.multiply,
                    operand0=-2.0 * float(w[n]),
                    op1=nl.add,
                    operand1=float(w[n]),
                )
            res.update(dc=dc, dx=k2, C=Cs)
        return res

    def _basis_features(rad, consts, J, nb):
        """``f [128, J, nb]``: ``prefactor · S / x`` (times ``c`` when the cutoff is applied)."""
        pf, apply = float(consts[1]), int(consts[6])
        qv = _plane(J)
        if apply:
            nisa.scalar_tensor_tensor(
                dst=qv,
                data=rad["c"],
                op0=nl.multiply,
                operand0=pf,
                op1=nl.multiply,
                operand1=rad["rx"],
            )
        else:
            nisa.tensor_scalar(dst=qv, data=rad["rx"], op0=nl.multiply, operand0=pf)
        f = nl.ndarray((EDGE_PART, J, nb), dtype=nl.float32, buffer=nl.sbuf)
        for n in range(nb):
            nisa.tensor_tensor(
                dst=f[0:EDGE_PART, 0:J, n],
                data1=rad["S"][0:EDGE_PART, 0:J, n],
                data2=qv,
                op=nl.multiply,
            )
        return f

    def _features_T(f, j0, G, nb):
        """``fᵀ [nb, 128·G]`` for columns ``j0 … j0+G−1`` (Tensor Engine transposes)."""
        P = EDGE_PART
        ps = nl.ndarray((nb, P * G), dtype=nl.float32, buffer=nl.psum)
        for g in range(G):
            nisa.nc_transpose(dst=ps[0:nb, g * P : (g + 1) * P], data=f[0:P, j0 + g, 0:nb])
        fT = nl.ndarray((nb, P * G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=fT, src=ps)
        return fT

    def _hidden_forward(fT, wf, hs, fo, G, keep_z):
        """Hidden activations ``a_l [h_l, 128·G]`` (and pre-activations) of one head."""
        P = EDGE_PART
        acts, zs, a = [], [], fT
        for ell in range(1, len(hs) - 1):
            ps = nl.ndarray((hs[ell], P * G), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=ps,
                stationary=wf[0 : hs[ell - 1], fo[ell - 1] : fo[ell - 1] + hs[ell]],
                moving=a,
            )
            if keep_z:
                z = nl.ndarray((hs[ell], P * G), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=z, src=ps)
                zs.append(z)
            a = nl.ndarray((hs[ell], P * G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=a, op=nl.silu, data=ps)
            acts.append(a)
        return acts, zs

    def _last_layer(a_col, wf, hs, fo, dst):
        """``dst [128, h_L] = a_colᵀ @ W_L`` (edges on partitions), in PSUM chunks."""
        P = EDGE_PART
        L = len(hs) - 1
        h_in, h_out, o = hs[L - 1], hs[L], fo[L - 1]
        for c0 in range(0, h_out, EDGE_PSUM_COLS):
            cw = min(EDGE_PSUM_COLS, h_out - c0)
            ps = nl.ndarray((P, cw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps, stationary=a_col, moving=wf[0:h_in, o + c0 : o + c0 + cw])
            nisa.tensor_copy(dst=dst[0:P, c0 : c0 + cw], src=ps)

    @nki.jit
    def edge_chain_fwd_kernel(r, r0, wpack, dims, consts, bessel):
        """Edge-chain forward: ``out [128, M·Σd]`` (per edge, the heads side by side).

        ``r``, ``r0 [128, M]`` (partition-major edges), ``wpack [128, WF]``
        (:func:`edge_pack_layout`), ``dims`` (:func:`edge_heads`),
        ``consts = [r_max, prefactor, p, a, q, pp, apply_cutoff]``,
        ``bessel = [w_1 … w_nb]``.
        """
        P = EDGE_PART
        _, M = r.shape
        heads = edge_heads(dims)
        fwd, WF, _bwd, _WB = edge_pack_layout(dims)
        offs, DO = edge_out_offsets(dims)
        nb = len(bessel)
        apply = int(consts[6])
        out = nl.ndarray((P, M * DO), dtype=r.dtype, buffer=nl.shared_hbm)
        wf = nl.ndarray((P, WF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wf, src=wpack[0:P, 0:WF])
        n_prog, pid = _programs(M)
        MP = _cols_per_program(M, n_prog)
        J = min(MP, EDGE_CHUNK)
        for j0 in range(0, MP, J):
            Jc = min(J, MP - j0)
            col = pid * MP + j0
            rt = _plane(Jc)
            nisa.dma_copy(dst=rt, src=r[0:P, col : col + Jc])
            r0t = _plane(Jc)
            nisa.dma_copy(dst=r0t, src=r0[0:P, col : col + Jc])
            rad = _radial_planes(rt, r0t, consts, bessel, Jc, False)
            f = _basis_features(rad, consts, Jc, nb)
            for g0 in range(0, Jc, EDGE_GROUP):
                G = min(EDGE_GROUP, Jc - g0)
                fT = _features_T(f, g0, G, nb)
                ot = nl.ndarray((P, G, DO), dtype=nl.float32, buffer=nl.sbuf)
                for h, (post, hs) in enumerate(heads):
                    acts, _zs = _hidden_forward(fT, wf, hs, fwd[h], G, False)
                    a_last = acts[-1] if acts else fT
                    for g in range(G):
                        dst = ot[0:P, g, offs[h] : offs[h] + hs[-1]]
                        _last_layer(a_last[0 : hs[-2], g * P : (g + 1) * P], wf, hs, fwd[h], dst)
                        if post == 1:  # tanh(z²)
                            nisa.activation(dst=dst, op=nl.square, data=dst)
                            nisa.activation(dst=dst, op=nl.tanh, data=dst)
                        if not apply:
                            cc = rad["c"][0:P, g0 + g : g0 + g + 1]
                            nisa.tensor_scalar(dst=dst, data=dst, op0=nl.multiply, operand0=cc)
                c1 = col + g0
                nisa.dma_copy(dst=out[0:P, c1 * DO : (c1 + G) * DO], src=ot)
        return out

    @nki.jit
    def edge_chain_bwd_kernel(g, r, r0, wpack, wtpack, dims, consts, bessel):
        """Edge-chain backward: ``∂/∂r [128, M]`` for the cotangent ``g [128, M·Σd]``.

        ``wtpack [128, WB]``: the transposed weights (:func:`edge_pack_layout`).
        Other operands as :func:`edge_chain_fwd_kernel`.
        """
        P = EDGE_PART
        _, M = r.shape
        heads = edge_heads(dims)
        fwd, WF, bwd, WB = edge_pack_layout(dims)
        offs, DO = edge_out_offsets(dims)
        nb = len(bessel)
        pf, apply = float(consts[1]), int(consts[6])
        dr = nl.ndarray((P, M), dtype=r.dtype, buffer=nl.shared_hbm)
        wf = nl.ndarray((P, WF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wf, src=wpack[0:P, 0:WF])
        wb = nl.ndarray((P, WB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wb, src=wtpack[0:P, 0:WB])
        n_prog, pid = _programs(M)
        MP = _cols_per_program(M, n_prog)
        J = min(MP, EDGE_CHUNK)
        for j0 in range(0, MP, J):
            Jc = min(J, MP - j0)
            col = pid * MP + j0
            rt = _plane(Jc)
            nisa.dma_copy(dst=rt, src=r[0:P, col : col + Jc])
            r0t = _plane(Jc)
            nisa.dma_copy(dst=r0t, src=r0[0:P, col : col + Jc])
            rad = _radial_planes(rt, r0t, consts, bessel, Jc, True)
            f = _basis_features(rad, consts, Jc, nb)
            dcp = _plane(Jc)  # Σ over heads / basis of the cutoff's cotangent
            nisa.memset(dst=dcp, value=0.0)
            dxp = _plane(Jc)  # ∂E/∂x
            for g0 in range(0, Jc, EDGE_GROUP):
                G = min(EDGE_GROUP, Jc - g0)
                fT = _features_T(f, g0, G, nb)
                c1 = col + g0
                gt = nl.ndarray((P, G, DO), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=gt, src=g[0:P, c1 * DO : (c1 + G) * DO])
                dF = nl.ndarray((P, G, nb), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=dF, value=0.0)
                for h, (post, hs) in enumerate(heads):
                    L = len(hs) - 1
                    acts, zs = _hidden_forward(fT, wf, hs, fwd[h], G, True)
                    a_last = acts[-1] if acts else fT
                    d_out = hs[-1]
                    # cotangent of the head's matmul output, per column [128, d_out]
                    gz = nl.ndarray((P, G, d_out), dtype=nl.float32, buffer=nl.sbuf)
                    for gi in range(G):
                        gv = gt[0:P, gi, offs[h] : offs[h] + d_out]
                        gzv = gz[0:P, gi, 0:d_out]
                        need_z = post == 1 or not apply
                        if need_z:
                            z = nl.ndarray((P, d_out), dtype=nl.float32, buffer=nl.sbuf)
                            _last_layer(
                                a_last[0 : hs[-2], gi * P : (gi + 1) * P], wf, hs, fwd[h], z
                            )
                        nisa.tensor_copy(dst=gzv, src=gv)
                        if post == 1:
                            y = nl.ndarray((P, d_out), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.activation(dst=y, op=nl.square, data=z)
                            nisa.activation(dst=y, op=nl.tanh, data=y)
                        if not apply:
                            yv = y if post == 1 else z
                            junk = nl.ndarray((P, d_out), dtype=nl.float32, buffer=nl.sbuf)
                            red = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.scalar_tensor_tensor(
                                dst=junk,
                                data=gv,
                                op0=nl.multiply,
                                operand0=1.0,
                                op1=nl.multiply,
                                operand1=yv,
                                reduce_op=nl.add,
                                reduce_res=red,
                            )
                            cslot = dcp[0:P, g0 + gi : g0 + gi + 1]
                            nisa.tensor_tensor(dst=cslot, data1=cslot, data2=red, op=nl.add)
                            nisa.tensor_scalar(
                                dst=gzv,
                                data=gzv,
                                op0=nl.multiply,
                                operand0=rad["c"][0:P, g0 + gi : g0 + gi + 1],
                            )
                        if post == 1:
                            # g · (1 − y²) · 2z
                            t = nl.ndarray((P, d_out), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.tensor_tensor(dst=t, data1=y, data2=y, op=nl.multiply)
                            nisa.tensor_scalar(
                                dst=t,
                                data=t,
                                op0=nl.multiply,
                                operand0=-2.0,
                                op1=nl.add,
                                operand1=2.0,
                            )
                            nisa.tensor_tensor(dst=t, data1=t, data2=z, op=nl.multiply)
                            nisa.tensor_tensor(dst=gzv, data1=gzv, data2=t, op=nl.multiply)
                    if L == 1:
                        # single-layer head, one output: dF += g_z · W_1ᵀ (row replicated)
                        wrep = wb[0:P, bwd[h][0] : bwd[h][0] + nb]
                        for gi in range(G):
                            nisa.scalar_tensor_tensor(
                                dst=dF[0:P, gi, 0:nb],
                                data=wrep,
                                op0=nl.multiply,
                                operand0=gz[0:P, gi, 0:1],
                                op1=nl.add,
                                operand1=dF[0:P, gi, 0:nb],
                            )
                        continue
                    # ∂a_{L-1} [h_{L-1}, 128·G] = W_L (g_z)ᵀ, over 128-row chunks of d_out
                    h_in = hs[L - 1]
                    n_oc = -(-d_out // P)
                    da_ps = nl.ndarray((h_in, P * G), dtype=nl.float32, buffer=nl.psum)
                    for oc in range(n_oc):
                        ow = min(P, d_out - oc * P)
                        gT_ps = nl.ndarray((ow, P * G), dtype=nl.float32, buffer=nl.psum)
                        for gi in range(G):
                            nisa.nc_transpose(
                                dst=gT_ps[0:ow, gi * P : (gi + 1) * P],
                                data=gz[0:P, gi, oc * P : oc * P + ow],
                            )
                        gT = nl.ndarray((ow, P * G), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=gT, src=gT_ps)
                        b0 = bwd[h][L - 1] + oc * h_in
                        nisa.nc_matmul(
                            dst=da_ps,
                            stationary=wb[0:ow, b0 : b0 + h_in],
                            moving=gT,
                            accumulate=(oc > 0),
                        )
                    da = nl.ndarray((h_in, P * G), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=da, src=da_ps)
                    for ell in range(L - 1, 0, -1):
                        # dz = da · silu'(z_ell)
                        sd = nl.ndarray((hs[ell], P * G), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.activation(dst=sd, op=nl.silu_dx, data=zs[ell - 1])
                        dz = nl.ndarray((hs[ell], P * G), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=dz, data1=da, data2=sd, op=nl.multiply)
                        b0 = bwd[h][ell - 1]
                        if ell > 1:
                            ps = nl.ndarray((hs[ell - 1], P * G), dtype=nl.float32, buffer=nl.psum)
                            nisa.nc_matmul(
                                dst=ps, stationary=wb[0 : hs[ell], b0 : b0 + hs[ell - 1]], moving=dz
                            )
                            da = nl.ndarray((hs[ell - 1], P * G), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.tensor_copy(dst=da, src=ps)
                        else:
                            # ∂f per column, edges on partitions: dzᵀ W_1ᵀ
                            for gi in range(G):
                                ps = nl.ndarray((P, nb), dtype=nl.float32, buffer=nl.psum)
                                nisa.nc_matmul(
                                    dst=ps,
                                    stationary=dz[0 : hs[1], gi * P : (gi + 1) * P],
                                    moving=wb[0 : hs[1], b0 : b0 + nb],
                                )
                                nisa.tensor_tensor(
                                    dst=dF[0:P, gi, 0:nb],
                                    data1=dF[0:P, gi, 0:nb],
                                    data2=ps,
                                    op=nl.add,
                                )
                # per column: dc += Σ dF·B (cutoff applied), dx = Σ dF·c^α·∂B/∂x
                for gi in range(G):
                    j = g0 + gi
                    dFv = dF[0:P, gi, 0:nb]
                    Sv, Cv = rad["S"][0:P, j, 0:nb], rad["C"][0:P, j, 0:nb]
                    rxj = rad["rx"][0:P, j : j + 1]
                    # B = pf · S / x ;  ∂B/∂x = pf (w cos − S / x) / x
                    B = nl.ndarray((P, nb), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=B, data=Sv, op0=nl.multiply, operand0=rxj, op1=nl.multiply, operand1=pf
                    )
                    dB = nl.ndarray((P, nb), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=dB,
                        data=Sv,
                        op0=nl.multiply,
                        operand0=rxj,
                        op1=nl.subtract,
                        operand1=Cv,
                        reverse1=True,
                    )
                    nisa.tensor_scalar(
                        dst=dB, data=dB, op0=nl.multiply, operand0=rxj, op1=nl.multiply, operand1=pf
                    )
                    junk = nl.ndarray((P, nb), dtype=nl.float32, buffer=nl.sbuf)
                    if apply:
                        red = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.scalar_tensor_tensor(
                            dst=junk,
                            data=dFv,
                            op0=nl.multiply,
                            operand0=1.0,
                            op1=nl.multiply,
                            operand1=B,
                            reduce_op=nl.add,
                            reduce_res=red,
                        )
                        cslot = dcp[0:P, j : j + 1]
                        nisa.tensor_tensor(dst=cslot, data1=cslot, data2=red, op=nl.add)
                        nisa.scalar_tensor_tensor(
                            dst=junk,
                            data=dFv,
                            op0=nl.multiply,
                            operand0=rad["c"][0:P, j : j + 1],
                            op1=nl.multiply,
                            operand1=dB,
                            reduce_op=nl.add,
                            reduce_res=dxp[0:P, j : j + 1],
                        )
                    else:
                        nisa.scalar_tensor_tensor(
                            dst=junk,
                            data=dFv,
                            op0=nl.multiply,
                            operand0=1.0,
                            op1=nl.multiply,
                            operand1=dB,
                            reduce_op=nl.add,
                            reduce_res=dxp[0:P, j : j + 1],
                        )
            # dr = dx · dx/dr + dc · dc/dr
            drt = _tt(dxp, rad["dx"], nl.multiply)
            t = _tt(dcp, rad["dc"], nl.multiply)
            nisa.tensor_tensor(dst=drt, data1=drt, data2=t, op=nl.add)
            nisa.dma_copy(dst=dr[0:P, col : col + Jc], src=drt)
        return dr
