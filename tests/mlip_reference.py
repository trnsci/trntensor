"""Test oracle for trntensor.equivariant: the MLIP exemplar's reference blocks.

``conv_tp_reference`` and ``symmetric_contraction_reference`` are the A1 / A2
blocks of trnsci/trnsci ``examples/mlip/blocks.py`` (the MLIP exemplar,
trnsci/trnsci#170), unchanged apart from names: plain PyTorch, one einsum per
path (the e3nn formulation), no e3nn or MACE import. They reproduce MACE-MP-0b2
medium (mace-torch 0.3.15) to <= 3e-7 relative in fp32.

``fixtures/mace_mp0b2_tiny.pt`` is that exemplar's committed CI fixture: an
exact slice (8 channels per irrep, 24 edges for A1) of the 16-atom capture of
the public MACE-MP-0b2 medium checkpoint, with the inputs, outputs, cotangents
and input gradients of every block and the w3j / U / W constants.

``mace_conv_tp_irreps`` reproduces MACE's ``tp_out_irreps_with_instructions``
(and its output sort) for hidden irreps ``Cx0e`` (layer 0) or ``Cx0e + Cx1o``
(layer 1) and spherical harmonics ``0e + 1o + 2e + 3o``.
"""

from __future__ import annotations

from pathlib import Path

import torch

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mace_mp0b2_tiny.pt"


def load_fixture() -> dict:
    return torch.load(FIXTURE, map_location="cpu", weights_only=True)


def _offsets(irreps):
    out, o = [], 0
    for mul, ell in irreps:
        d = mul * (2 * ell + 1)
        out.append((o, o + d))
        o += d
    return out, o


def conv_tp_reference(x, y, w, spec: dict):
    in1, in2, out_irreps = spec["irreps_in1"], spec["irreps_in2"], spec["irreps_out"]
    s1, _ = _offsets(in1)
    s2, _ = _offsets(in2)
    E = x.shape[0]
    outs: list = [None] * len(out_irreps)
    woff = 0
    for i1, i2, io, alpha in spec["instructions"]:
        (mul1, l1), (mul2, l2), (mulo, l3) = in1[i1], in2[i2], out_irreps[io]
        x1 = x[:, s1[i1][0] : s1[i1][1]].reshape(E, mul1, 2 * l1 + 1)
        x2 = y[:, s2[i2][0] : s2[i2][1]].reshape(E, mul2, 2 * l2 + 1)
        wp = w[:, woff : woff + mul1 * mul2].reshape(E, mul1, mul2)
        woff += mul1 * mul2
        C = spec["w3j"][f"{l1},{l2},{l3}"].to(x.device, x.dtype)
        r = alpha * torch.einsum("euv,ijk,eui,evj->euk", wp, C, x1, x2)
        r = r.reshape(E, mulo * (2 * l3 + 1))
        outs[io] = r if outs[io] is None else outs[io] + r
    for io, (mulo, l3) in enumerate(out_irreps):
        if outs[io] is None:
            outs[io] = x.new_zeros(E, mulo * (2 * l3 + 1))
    return torch.cat(outs, dim=1)


def _dense_u(u) -> torch.Tensor:
    if torch.is_tensor(u):
        return u
    dense = torch.zeros(u["shape"], dtype=u["values"].dtype)
    dense[tuple(u["indices"].long())] = u["values"]
    return dense


def symmetric_contraction_reference(x, species, spec: dict, chunk: int = 4096):
    N, C, L = x.shape
    blocks = []
    for con in spec["contractions"]:
        d = con["d_out"]
        parts = []
        for n0 in range(0, N, max(1, chunk // C)):
            n1 = min(N, n0 + max(1, chunk // C))
            xb, zb = x[n0:n1], species[n0:n1]
            acc = None
            mono = None
            for nu in range(1, con["correlation"] + 1):
                U = _dense_u(con["U"][str(nu)]).to(x.device, x.dtype)
                if d == 1:
                    U = U.unsqueeze(0)
                K = U.shape[-1]
                mono = (
                    xb
                    if mono is None
                    else (mono.unsqueeze(-1) * xb.reshape(xb.shape[:2] + (1,) * (nu - 1) + (L,)))
                )
                q = mono.reshape(-1, L**nu) @ U.reshape(d, L**nu, K).permute(1, 0, 2).reshape(
                    L**nu, d * K
                )
                q = q.reshape(n1 - n0, C, d, K)
                wz = con["W"][str(nu)].to(x.device, x.dtype)[zb]  # [n, K, C]
                t = torch.einsum("ncdk,nkc->ncd", q, wz)
                acc = t if acc is None else acc + t
            parts.append(acc.reshape(n1 - n0, C * d))
        blocks.append(torch.cat(parts, 0))
    return torch.cat(blocks, dim=1)


def mace_conv_tp_irreps(layer: int, channels: int):
    """``(irreps_in1, irreps_in2, irreps_out, instructions)`` as MACE builds them.

    Parity is (-1)^l for every irrep here (0e, 1o, 2e, 3o), so a product
    ``l1 x l2 -> l3`` is allowed when ``l1 + l2 + l3`` is even. Outputs are
    sorted by l (stably, as e3nn's ``Irreps.sort``), then the instructions by
    output index, which is also the weight order.
    """
    in1 = [(channels, 0)] if layer == 0 else [(channels, 0), (channels, 1)]
    in2 = [(1, ell) for ell in range(4)]
    outs, ins = [], []
    for i1, (_, l1) in enumerate(in1):
        for i2, (_, l2) in enumerate(in2):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                if l3 <= 3 and (l1 + l2 + l3) % 2 == 0:
                    ins.append((i1, i2, len(outs)))
                    outs.append((channels, l3))
    order = sorted(range(len(outs)), key=lambda k: outs[k][1])
    perm = {k: j for j, k in enumerate(order)}
    irreps_out = [outs[k] for k in order]
    ins = sorted(((i1, i2, perm[k]) for i1, i2, k in ins), key=lambda t: t[2])
    return in1, in2, irreps_out, ins
