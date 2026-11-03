"""Host cost of one native kernel call: the `trntensor::` custom op against the
cached eager handle (`TRNTENSOR_NKI_EAGER_HANDLE`, trnblas#92), with aten for
scale, on device-resident inputs.

``issue`` is the host time per call of n back-to-back calls; ``e2e`` the
median of one call plus a synchronize; ms. ``same_as_op`` compares the
handle's result with the op's bit for bit. Inputs are created on the host
and copied to the device (aws-neuron/aws-neuron-sdk#1439).

    python scripts/bench_eager_handle.py
"""

from __future__ import annotations

import statistics
import time

import torch


def _sync():
    import torch_neuronx

    sync = getattr(torch_neuronx, "synchronize", None)
    if sync is None:
        sync = torch_neuronx.neuron.synchronize
    sync()


def _issue_e2e(fn, n=200, e2e_reps=100):
    for _ in range(10):
        fn()
    _sync()
    issue = []
    for _ in range(5):
        t0 = time.perf_counter()
        rs = [fn() for _ in range(n)]
        issue.append((time.perf_counter() - t0) / n * 1e3)
        _sync()
        del rs
    e2e = []
    for _ in range(e2e_reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        e2e.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(issue), statistics.median(e2e)


def _host(out):
    if isinstance(out, tuple | list):
        return tuple(_host(o) for o in out)
    return out.detach().cpu() if isinstance(out, torch.Tensor) else out


def _same(a, b):
    if isinstance(a, tuple):
        return all(_same(x, y) for x, y in zip(a, b, strict=True))
    return torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b


def run(D, cases, n=200):
    """cases: (name, fn, aten) — aten=True runs once (no op/handle toggle)."""
    print(f"{'case':44s} {'path':7s} {'issue':>8s} {'e2e':>8s}  same_as_op", flush=True)
    for name, fn, aten, *kw in cases:
        reps = kw[0] if kw else n
        paths = ("aten",) if aten else ("op", "handle")
        ref = None
        for path in paths:
            D._EAGER_HANDLE_ENABLED = path != "op"
            out = _host(fn())
            same = ""
            if path == "op":
                ref = out
            elif path == "handle":
                same = "yes" if _same(out, ref) else "NO"
            del out
            issue, e2e = _issue_e2e(fn, n=reps, e2e_reps=min(100, 5 * reps))
            print(f"{name:44s} {path:7s} {issue:8.4f} {e2e:8.4f}  {same}", flush=True)
    D._EAGER_HANDLE_ENABLED = True


def main():
    import trntensor
    from trntensor.nki import dispatch as D

    g = torch.Generator().manual_seed(0)
    A, B = torch.randn(256, 128, generator=g), torch.randn(128, 512, generator=g)
    a, b = A.to("neuron"), B.to("neuron")
    Bm = (torch.randn(8, 128, 256, generator=g) * 0.05).to("neuron")
    eo = (-torch.rand(8, generator=g) - 1.0).to("neuron")
    ev = (torch.rand(128, generator=g) + 0.5).to("neuron")
    eri = torch.randn(300, 300, 600, generator=g) * 0.1
    eri = 0.5 * (eri + eri.transpose(0, 1))
    pk = trntensor.pack_tril(eri).to("neuron")
    C = (torch.randn(300, 33, generator=g) / 300**0.5).to("neuron")
    del eri
    run(
        D,
        [
            ("aten a@b 256x128x512", lambda: a @ b, True),
            (
                "matmul 256x128x512 (use_sr=False)",
                lambda: D._launch(None, "matmul", a, b, use_sr=False),
                False,
            ),
            (
                "matmul 256x128x512 (use_sr=True)",
                lambda: D._launch(None, "matmul", a, b, use_sr=True),
                False,
            ),
            (
                "_nki_mp2_energy_rows nocc=8 (8 launches)",
                lambda: D._nki_mp2_energy_rows(Bm, eo, ev),
                False,
                20,
            ),
            (
                "_nki_df_exchange 300/33/600 (nub/tp/tail, banks)",
                lambda: D._nki_df_exchange(pk, C),
                False,
                20,
            ),
        ],
    )


if __name__ == "__main__":
    main()
