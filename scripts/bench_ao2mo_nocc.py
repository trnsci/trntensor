"""AO→MO on TorchNeuron across nocc, full and packed eri (#80).

Times, on device-resident ``neuron`` tensors made on the host and copied over
(aws-neuron-sdk#1439), the warm median latency of:

    aten     the aten route: two einsums (packed input: unpack_tril first)
    trn2     the Trn2 kernel in one launch (occupied blocks inside, #80)
    blk128   the nocc <= 128 kernel launched once per 128-column block of
             C_occ, results concatenated: what the kernel could do before #80
             (each launch reads all of eri again). Equal to trn2 at nocc <= 128.
    auto     trntensor.ao_to_mo_transform under the default routing

``--ref`` also checks each route against a float64 host reference (slow at
large sizes); otherwise the kernels are compared with aten on the device.

    python scripts/bench_ao2mo_nocc.py --sizes "768,160,2304;768,384,2304" --forms full,packed
"""

from __future__ import annotations

import argparse
import gc
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
    ref = ref.detach().cpu().double()
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1e-300))


def _blk128(trn2, eri, co, cv):
    nocc = co.shape[1]
    if nocc <= 128:
        return trn2(eri, co, cv)
    outs = [trn2(eri, co[:, o : o + 128].contiguous(), cv) for o in range(0, nocc, 128)]
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", required=True, help="semicolon list of nbasis,nocc,naux")
    ap.add_argument("--forms", default="full,packed")
    ap.add_argument("--modes", default="aten,trn2,blk128,auto")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--ref", action="store_true", help="float64 host reference")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    sizes = [tuple(int(x) for x in s.split(",")) for s in args.sizes.split(";") if s]
    want = args.modes.split(",")
    tag = {
        "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
        "trace_cache": os.environ.get("NKI_ENABLE_TRACE_CACHE"),
    }
    print(f"# {tag}", flush=True)
    for nb, no, nx in sizes:
        nv = nb - no
        g = torch.Generator().manual_seed(0)
        co_h = torch.randn(nb, no, generator=g) / nb**0.5
        cv_h = torch.randn(nb, nv, generator=g) / nb**0.5
        npair = nb * (nb + 1) // 2
        pk_h = torch.randn(npair, nx, generator=g)
        ref = None
        if args.ref:
            full = quantum.unpack_tril(pk_h.double(), nb)
            ref = torch.einsum(
                "na,inP->iaP", cv_h.double(), torch.einsum("mi,mnP->inP", co_h.double(), full)
            )
            del full
        co, cv, pk = co_h.to(DEV), cv_h.to(DEV), pk_h.to(DEV)
        del pk_h
        _sync()
        forms = args.forms.split(",")
        for form in forms:
            eri = pk if form == "packed" else quantum.unpack_tril(pk, nb)
            _sync()
            if "packed" not in forms:
                pk = None  # hold only the full square (device-memory pressure, #1447)
            if form == "packed":
                trn2 = dispatch._nki_ao_to_mo_trn2_packed

                def aten(e, c, v, nb=nb):
                    return quantum._cpu_ao_to_mo_transform(quantum.unpack_tril(e, nb), c, v)
            else:
                trn2 = dispatch._nki_ao_to_mo_trn2
                aten = quantum._cpu_ao_to_mo_transform
            modes = {
                "aten": aten,
                "trn2": trn2,
                "blk128": lambda e, c, v, f=trn2: _blk128(f, e, c, v),
                "auto": quantum.ao_to_mo_transform,
            }
            row = {
                "nbasis": nb,
                "nocc": no,
                "nvir": nv,
                "naux": nx,
                "form": form,
                "macs": no * nb * nb * nx,
                "route": dispatch._ao_to_mo_route(eri, no, nv, nb),
                **tag,
            }
            base = None
            for m in want:
                if m not in modes or (m == "blk128" and no <= 128):
                    continue
                fn = lambda f=modes[m], e=eri, c=co, v=cv: f(e, c, v)  # noqa: E731
                try:
                    t0 = time.perf_counter()
                    out = fn()
                    _sync()
                    row[f"{m}_first_s"] = round(time.perf_counter() - t0, 2)
                    if ref is not None:
                        row[f"{m}_err64"] = _relerr(out, ref)
                    if base is None:
                        base = out.cpu()
                    else:
                        row[f"{m}_vs_{want[0]}"] = _relerr(out, base)
                    del out
                    gc.collect()
                    row[m] = round(_time(fn, args.warm, args.reps), 3)
                except Exception as exc:  # noqa: BLE001
                    row[m] = None
                    row[f"{m}_err"] = repr(exc)[:300]
                gc.collect()
            del eri, base
            gc.collect()
            print(json.dumps(row), flush=True)
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(row) + "\n")
        del co, cv, pk, ref
        gc.collect()


if __name__ == "__main__":
    main()
