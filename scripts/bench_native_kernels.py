"""Kernel-level A/B on TorchNeuron: trntensor NKI kernels vs aten (#51).

Times, on device-resident ``neuron`` tensors created on the host and copied
over (aws-neuron-sdk#1439), the warm median latency of:

    aten      the aten formulation trntensor's native ``auto`` uses
    nki_trn1  the original kernel (the XLA-path kernel, as on main)
    nki_trn2  the native Trn2 kernel (program_id split, no DMA transposes)

Families: matmul, ao2mo, mp2. ``--hold-gb G`` keeps G GB allocated on the
device during the run (the memory-pressure regime of aws-neuron-sdk#1447).

    python scripts/bench_native_kernels.py --family mp2 --hold-gb 12
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")

import torch  # noqa: E402

from trntensor import quantum  # noqa: E402
from trntensor.nki import dispatch  # noqa: E402

DEV = "neuron"


def _sync():
    torch.neuron.synchronize()


def _time(fn, warm, reps, budget_s=20.0):
    for _ in range(warm):
        fn()
    _sync()
    t0 = time.perf_counter()
    fn()
    _sync()
    one = time.perf_counter() - t0
    if one > 0.05:
        reps = max(3, min(reps, int(budget_s / one)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def _relerr(x, ref):
    x = x.detach().cpu().double()
    ref = ref.double()
    return float((x - ref).norm() / ref.norm().clamp_min(1e-300))


def _ops():
    return dispatch._get_native_ops()


def cases_matmul(dtype, sizes):
    sizes = sizes or [(n, n, n) for n in (512, 1024, 2048, 4096)] + [(512, 2048, 512)]
    for M, K, N in sizes:

        def make(M=M, K=K, N=N):
            g = torch.Generator().manual_seed(0)
            a = torch.randn(M, K, generator=g).to(dtype)
            b = torch.randn(K, N, generator=g).to(dtype)
            return (a, b), (a.double() @ b.double())

        yield (
            f"matmul {M}x{K}x{N} {str(dtype)[6:]}",
            make,
            {
                "aten": lambda a, b: torch.matmul(a, b),
                "nki_trn1": lambda a, b: _ops()["matmul"](a, b),
                "nki_trn2": lambda a, b: _ops()["matmul_trn2"](a, b),
            },
        )


def cases_ao2mo(dtype, sizes):
    # (nbasis, nocc, naux), nvir = nbasis - nocc; nbasis/nvir/naux tile-padded below.
    sizes = sizes or [(128, 16, 256), (256, 32, 512), (512, 32, 1024), (768, 48, 1536)]
    for nb, no, nx in sizes:
        nv = nb - no

        def make(nb=nb, no=no, nv=nv, nx=nx):
            g = torch.Generator().manual_seed(0)
            co = torch.randn(nb, no, generator=g).to(dtype)
            cv = torch.randn(nb, nv, generator=g).to(dtype)
            eri = torch.randn(nb, nb, nx, generator=g).to(dtype)
            ref = torch.einsum(
                "na,inP->iaP", cv.double(), torch.einsum("mi,mnP->inP", co.double(), eri.double())
            )
            return (eri, co, cv), ref

        modes = {
            "aten": lambda eri, co, cv: quantum._cpu_ao_to_mo_transform(eri, co, cv),
            "nki_trn2": lambda eri, co, cv: dispatch._nki_ao_to_mo_trn2(eri, co, cv),
        }
        if nb <= 128:  # the original kernel compiles for many minutes beyond this
            modes["nki_trn1"] = lambda eri, co, cv: dispatch._nki_ao_to_mo_transform(eri, co, cv)
        yield (f"ao2mo nb={nb} no={no} nv={nv} nx={nx}", make, modes)


def cases_mp2(dtype, sizes):
    sizes = sizes or [
        (16, 128, 512),
        (32, 128, 512),
        (32, 256, 768),
        (48, 384, 1152),
        (64, 448, 1536),
        (96, 672, 2304),
    ]
    for no, nv, nx in sizes:

        def make(no=no, nv=nv, nx=nx):
            g = torch.Generator().manual_seed(0)
            B = (torch.randn(no, nv, nx, generator=g) * 0.05).to(dtype)
            eo = -torch.rand(no, generator=g) - 1.0
            ev = torch.rand(nv, generator=g) + 0.5
            ref = quantum._cpu_mp2_energy(B.double(), eo.double(), ev.double())
            return (B, eo, ev), ref

        modes = {
            "aten": lambda B, eo, ev: quantum._cpu_mp2_energy(B, eo, ev),
            "nki_trn2": lambda B, eo, ev: dispatch._nki_mp2_energy_rows(B, eo, ev),
        }
        if no <= 32:
            modes["nki_trn1"] = lambda B, eo, ev: dispatch._nki_mp2_energy(B, eo, ev)
        yield (f"mp2 no={no} nv={nv} nx={nx}", make, modes)


FAMILIES = {"matmul": cases_matmul, "ao2mo": cases_ao2mo, "mp2": cases_mp2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--modes", default="aten,nki_trn1,nki_trn2")
    ap.add_argument("--sizes", default="", help="semicolon list of comma tuples")
    ap.add_argument("--hold-gb", type=float, default=0.0)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    sizes = [tuple(int(x) for x in s.split(",")) for s in args.sizes.split(";") if s]
    want = args.modes.split(",")

    held = []
    if args.hold_gb > 0:
        # Hold the working set in 1 GiB host-made blocks (device randn is
        # shape-sensitive, #1439; zeros are enough to occupy the memory).
        for _ in range(int(args.hold_gb)):
            held.append(torch.ones(256 * 1024 * 1024, dtype=torch.float32).to(DEV))
        _sync()
    print(f"# family={args.family} dtype={args.dtype} hold_gb={args.hold_gb}", flush=True)
    for label, make, modes in FAMILIES[args.family](dtype, sizes):
        host, ref = make()
        dev = tuple(t.to(DEV) for t in host)
        _sync()
        row = {"case": label, "hold_gb": args.hold_gb}
        for m in want:
            if m not in modes:
                continue
            fn = lambda f=modes[m], o=dev: f(*o)  # noqa: E731
            try:
                t0 = time.perf_counter()
                out = fn()
                _sync()
                row[f"{m}_first_s"] = round(time.perf_counter() - t0, 2)
                row[f"{m}_relerr"] = _relerr(out, ref)
                del out
                row[m] = _time(fn, args.warm, args.reps)
            except Exception as exc:  # noqa: BLE001
                row[m] = None
                row[f"{m}_err"] = repr(exc)[:400]
        del dev
        print(json.dumps(row), flush=True)
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
