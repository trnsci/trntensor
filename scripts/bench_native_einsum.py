"""Warm-latency sweep: trntensor einsum vs aten ``torch.einsum`` on TorchNeuron.

Measures, per contraction and size, the median wall-clock latency of:

    cpu        torch.einsum on host tensors
    aten_off   (unpinned) host operands -> device, torch.einsum, result -> host
    aten_dev   torch.einsum on device-resident ``neuron`` tensors
    tt_auto    trntensor, default dispatch (thresholds as configured)
    tt_nki     trntensor, NKI forced (thresholds 0, TRNTENSOR_REQUIRE_NKI=1)
    tt_pt      trntensor, set_backend("pytorch") (aten on device)

Device variants take device-resident operands and are timed to completion
with ``torch.neuron.synchronize()``. ``--family unpinned`` instead feeds host
operands, so the NKI variant includes the host->device and device->host
transfers (the case the unpinned threshold governs).

Native path only (TorchNeuron, Trn2/Trn3). One family per process:

    python scripts/bench_native_einsum.py --family matmul --dtype float32 \
        --json out/matmul_f32.jsonl

Families: matmul, bmm, ijk, df, mp2, multi, unpinned.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")
os.environ["TRNTENSOR_REQUIRE_NKI"] = "1"  # surface kernel failures, never fall back silently

import torch  # noqa: E402

import trntensor  # noqa: E402
from trntensor import quantum  # noqa: E402
from trntensor.nki import dispatch  # noqa: E402

DEV = "neuron"
_DEFAULTS = (
    dispatch._MIN_NKI_FLOPS_NATIVE,
    dispatch._MIN_NKI_FLOPS_NATIVE_PINNED,
)


def _sync():
    torch.neuron.synchronize()


def _mode(mode: str) -> None:
    """Configure trntensor dispatch for one variant."""
    dispatch._MIN_NKI_FLOPS_NATIVE, dispatch._MIN_NKI_FLOPS_NATIVE_PINNED = _DEFAULTS
    if mode in ("tt_auto", "tt_auto_cpu"):
        trntensor.set_backend("auto")
    elif mode in ("tt_nki", "tt_nki_cpu"):
        trntensor.set_backend("nki")
        dispatch._MIN_NKI_FLOPS_NATIVE = 0
        dispatch._MIN_NKI_FLOPS_NATIVE_PINNED = 0
    elif mode == "tt_pt":
        trntensor.set_backend("pytorch")


def _time(fn, device: bool, warm: int, reps: int) -> float:
    for _ in range(warm):
        fn()
    if device:
        _sync()
    # Shrink reps for slow calls so a single variant stays bounded.
    t0 = time.perf_counter()
    fn()
    if device:
        _sync()
    one = time.perf_counter() - t0
    if one > 0.05:
        reps = max(3, min(reps, int(1.0 / one)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if device:
            _sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def _relerr(x, ref) -> float:
    if isinstance(x, list | tuple):
        return max(_relerr(a, b) for a, b in zip(x, ref, strict=True))
    x = x.detach().to("cpu", torch.float64)
    ref = ref.detach().to("cpu", torch.float64)
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


# ── cases ────────────────────────────────────────────────────────────────────
# Each case: (label, flops, make(dtype) -> host operands, aten(ops), tt(ops))
# flops = multiply-add count M·K·N-style (what dispatch thresholds compare).


def _rand(*shape, dtype):
    return torch.randn(*shape, dtype=torch.float32).to(dtype)


def cases_matmul(dtype):
    shapes = [(n, n, n) for n in (128, 256, 512, 768, 1024, 1536, 2048, 4096)]
    shapes += [(64, 512, 64), (128, 512, 128), (256, 512, 256), (512, 2048, 512), (128, 4096, 128)]
    out = []
    for M, K, N in shapes:
        out.append(
            (
                f"ij,jk->ik {M}x{K}x{N}",
                M * K * N,
                lambda M=M, K=K, N=N: (_rand(M, K, dtype=dtype), _rand(K, N, dtype=dtype)),
                lambda a, b: torch.einsum("ij,jk->ik", a, b),
                lambda a, b: trntensor.einsum("ij,jk->ik", a, b),
            )
        )
    return out


def cases_bmm(dtype):
    shapes = [
        (8, 128, 128, 128),
        (32, 128, 128, 128),
        (8, 256, 256, 256),
        (32, 256, 256, 256),
        (8, 512, 512, 512),
        (32, 512, 512, 512),
        (64, 64, 64, 64),
        (8, 1024, 1024, 1024),
    ]
    out = []
    for Bz, M, K, N in shapes:
        out.append(
            (
                f"bij,bjk->bik {Bz}x{M}x{K}x{N}",
                Bz * M * K * N,
                lambda Bz=Bz, M=M, K=K, N=N: (
                    _rand(Bz, M, K, dtype=dtype),
                    _rand(Bz, K, N, dtype=dtype),
                ),
                lambda a, b: torch.einsum("bij,bjk->bik", a, b),
                lambda a, b: trntensor.einsum("bij,bjk->bik", a, b),
            )
        )
    return out


def cases_ijk(dtype):
    shapes = [(16, 64, 64, 64), (32, 128, 128, 128), (64, 128, 256, 256), (64, 256, 512, 512)]
    out = []
    for i, j, k, n in shapes:
        out.append(
            (
                f"ijk,kl->ijl {i}x{j}x{k}x{n}",
                i * j * k * n,
                lambda i=i, j=j, k=k, n=n: (_rand(i, j, k, dtype=dtype), _rand(k, n, dtype=dtype)),
                lambda a, b: torch.einsum("ijk,kl->ijl", a, b),
                lambda a, b: trntensor.einsum("ijk,kl->ijl", a, b),
            )
        )
    return out


def cases_df(dtype):
    # (nbasis, nocc, naux); nvir = nbasis - nocc
    shapes = [(64, 8, 128), (128, 16, 256), (256, 32, 512)]
    out = []
    for nb, no, nx in shapes:
        nv = nb - no
        fl = no * nb * nb * nx + no * nv * nb * nx  # two-step cost

        def make(nb=nb, no=no, nv=nv, nx=nx):
            return (
                _rand(nb, no, dtype=dtype),
                _rand(nb, nv, dtype=dtype),
                _rand(nb, nb, nx, dtype=dtype),
            )

        out.append(
            (
                f"mi,na,mnP->iaP nb={nb} no={no} nx={nx} (einsum)",
                fl,
                make,
                lambda c, v, e: torch.einsum("mi,na,mnP->iaP", c, v, e),
                lambda c, v, e: trntensor.einsum("mi,na,mnP->iaP", c, v, e),
            )
        )
        out.append(
            (
                f"ao_to_mo_transform nb={nb} no={no} nx={nx}",
                fl,
                make,
                lambda c, v, e: torch.einsum("mi,na,mnP->iaP", c, v, e),
                lambda c, v, e: quantum.ao_to_mo_transform(e, c, v),
            )
        )
    return out


def _aten_mp2(B, eo, ev):
    T = torch.einsum("iap,jbp->ijab", B, B)
    d = (
        eo[:, None, None, None]
        + eo[None, :, None, None]
        - ev[None, None, :, None]
        - ev[None, None, None, :]
    )
    return (T * (2 * T - T.transpose(2, 3)) / d).sum()


def cases_mp2(dtype):
    shapes = [(8, 32, 128), (16, 64, 256), (16, 128, 512), (32, 128, 512)]
    out = []
    for no, nv, nx in shapes:

        def make(no=no, nv=nv, nx=nx):
            B = _rand(no, nv, nx, dtype=dtype) * 0.1
            eo = -torch.rand(no).to(dtype) - 1.0
            ev = torch.rand(nv).to(dtype) + 0.5
            return (B, eo, ev)

        out.append(
            (
                f"mp2_energy no={no} nv={nv} nx={nx}",
                no * no * nv * nv * nx * 2,
                make,
                _aten_mp2,
                lambda B, eo, ev: quantum.mp2_energy(B, eo, ev),
            )
        )
    return out


def cases_multi(dtype):
    out = []
    for n in (256, 512, 1024):
        nb = 8

        def make(n=n, nb=nb):
            return tuple(_rand(n, n, dtype=dtype) for _ in range(nb)) + (_rand(n, n, dtype=dtype),)

        def aten(*ops):
            *As, B = ops
            return [torch.einsum("ij,jk->ik", a, B).cpu() for a in As]

        def tt(*ops):
            *As, B = ops
            return trntensor.multi_einsum(*(("ij,jk->ik", a, B) for a in As))

        out.append((f"multi_einsum 8x ij,jk->ik {n}^3 (shared B)", nb * n**3, make, aten, tt))

    for no, nv, nx in ((8, 64, 256), (16, 128, 512)):

        def make(no=no, nv=nv, nx=nx):
            return (_rand(no, nv, nx, dtype=dtype),)

        def aten(B):
            return [
                torch.einsum("iap,jbp->ijab", B, B).cpu(),
                torch.einsum("ibp,jap->ijab", B, B).cpu(),
            ]

        def tt(B):
            return trntensor.multi_einsum(("iap,jbp->ijab", B, B), ("ibp,jap->ijab", B, B))

        out.append(
            (
                f"multi_einsum MP2 J/K no={no} nv={nv} nx={nx}",
                2 * no * no * nv * nv * nx,
                make,
                aten,
                tt,
            )
        )

    for n in (256, 1024, 2048):

        def make(n=n):
            return (_rand(n, n, dtype=dtype), _rand(n, n, dtype=dtype), _rand(n, n, dtype=dtype))

        out.append(
            (
                f"chain ij,jk,kl->il {n}^3",
                2 * n**3,
                make,
                lambda a, b, c: torch.einsum("ij,jk,kl->il", a, b, c),
                lambda a, b, c: trntensor.einsum("ij,jk,kl->il", a, b, c),
            )
        )
    return out


FAMILIES = {
    "matmul": cases_matmul,
    "bmm": cases_bmm,
    "ijk": cases_ijk,
    "df": cases_df,
    "mp2": cases_mp2,
    "multi": cases_multi,
}

DEVICE_MODES = ("aten_dev", "tt_auto", "tt_nki", "tt_pt")


def run_case(case, modes, warm, reps, host_ops_for_tt=False):
    label, flops, make, aten, tt = case
    torch.manual_seed(0)
    host = make()
    dev = tuple(t.to(DEV) for t in host)
    _sync()
    row = {"case": label, "flops": flops}
    ref = None
    try:
        row["cpu"] = _time(lambda: aten(*host), False, 1, max(3, reps // 4))
        ref = aten(*(t.double() for t in host))
    except Exception as exc:  # noqa: BLE001
        row["cpu"] = None
        row["cpu_err"] = repr(exc)[:200]
    for mode in modes:
        _mode(mode)
        ops = host if (host_ops_for_tt and mode.startswith("tt_")) else dev
        if mode == "aten_dev":
            fn = lambda f=aten, o=ops: f(*o)  # noqa: E731
        elif mode == "aten_off":  # host operands, offload to aten on device, result back
            fn = lambda f=aten, o=host: f(*(t.to(DEV) for t in o)).cpu()  # noqa: E731
        else:
            fn = lambda f=tt, o=ops: f(*o)  # noqa: E731
        try:
            row[mode] = _time(fn, True, warm, reps)
            if ref is not None:
                row[f"{mode}_relerr"] = _relerr(fn(), ref)
        except Exception as exc:  # noqa: BLE001
            row[mode] = None
            row[f"{mode}_err"] = repr(exc)[:300]
        _mode("tt_auto")
    return row


def _fmt(v):
    return "ERR" if v is None else f"{v:9.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--family", required=True, choices=[*FAMILIES, "unpinned"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--only", default="", help="substring filter on case labels")
    ap.add_argument("--json", default="", help="append JSON lines here")
    args = ap.parse_args()

    if dispatch._get_dispatch_path() != "native":
        sys.exit("native TorchNeuron path not available")
    dtype = getattr(torch, args.dtype)

    if args.family == "unpinned":
        cases = [c for c in cases_matmul(dtype)]
        modes = ("tt_auto_cpu", "tt_nki_cpu", "aten_off")
        host_tt = True
    else:
        cases = FAMILIES[args.family](dtype)
        modes = DEVICE_MODES
        host_tt = False
    cases = [c for c in cases if args.only in c[0]]

    print(f"# family={args.family} dtype={args.dtype} defaults={_DEFAULTS}", flush=True)
    print(
        f"{'case':52s} {'MFLOP':>9s} {'cpu':>9s} " + " ".join(f"{m:>11s}" for m in modes),
        flush=True,
    )
    for case in cases:
        row = run_case(case, modes, args.warm, args.reps, host_ops_for_tt=host_tt)
        row["dtype"] = args.dtype
        row["family"] = args.family
        print(
            f"{row['case']:52s} {row['flops'] / 1e6:9.1f} {_fmt(row.get('cpu'))} "
            + " ".join(f"{_fmt(row.get(m)):>11s}" for m in modes),
            flush=True,
        )
        for m in modes:
            if row.get(f"{m}_err"):
                print(f"    {m}: {row[f'{m}_err']}", flush=True)
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
