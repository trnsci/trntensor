"""Dispatch threshold calibration for trntensor NKI matmul dispatch (#33).

Sweeps a range of (M, K, N) shapes, times NKI vs torch.matmul for each,
and reports the FLOP count at which NKI first beats PyTorch. The result
informs the TRNTENSOR_MIN_NKI_FLOPS constant in dispatch.py.

Background:
    nki_matmul has ~1ms per-call overhead from XLA graph compile + NEFF
    dispatch. For small contractions the torch.matmul fallback finishes
    in microseconds; NKI only wins when the kernel work exceeds the
    overhead. TRNTENSOR_MIN_NKI_FLOPS gates which path is taken; this
    script finds the empirical crossover on the current hardware.

Run on trn1 (via SSM or directly):
    TRNTENSOR_FORCE_BACKEND=nki python scripts/autotune_dispatch.py
    TRNTENSOR_FORCE_BACKEND=nki python scripts/autotune_dispatch.py --write-cache

The script is dispatch-path aware. FLOPs below are multiply-add counts
M·K·N, the quantity dispatch.py compares against the thresholds.

On the XLA lineage it times the kernel on the XLA device against host
torch.matmul and writes ``trntensor_min_nki_flops`` as before.

On TorchNeuron (native, Trn2/Trn3) it times five executors per shape:
host torch.matmul, aten offload (host operands -> ``neuron`` -> aten ->
host), the ``trntensor::matmul`` op with the same transfers, and aten and
the op on device-resident operands (synchronized). It then derives both
native thresholds:

    trntensor_min_nki_flops_native         first M·K·N where the device (best
                                           of aten / NKI, with transfers)
                                           beats host torch
    trntensor_min_nki_flops_native_pinned  first M·K·N where the NKI op beats
                                           aten on device-resident operands,
                                           or "never"

An existing cache file is updated, not replaced, so one file can hold both
paths' values.

The --write-cache flag writes the recommendation to
/var/tmp/trntensor-autotune/threshold.json so dispatch.py can read it
on next start (pass TRNTENSOR_AUTOTUNE_CACHE to override the path).

Exit code 1 on the XLA path if the NKI crossover was never observed (NKI
loses at all tested sizes — indicates a regression or configuration
problem). On the native path NKI losing to aten everywhere is a valid
result (recorded as "never"); exit 1 only if the device never beats host.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Shape sweep: (label, M, K, N) — representative chemistry / einsum sizes.
# Sorted ascending by FLOPs so we can find the crossover efficiently.
# ---------------------------------------------------------------------------
SHAPES: list[tuple[str, int, int, int]] = [
    # below expected crossover
    ("sq128", 128, 128, 128),
    ("sq256", 256, 256, 256),
    ("sq512", 512, 512, 512),
    # around expected crossover
    ("sq768", 768, 768, 768),
    ("sq1024", 1024, 1024, 1024),
    ("rect_df_small", 64, 512, 64),  # DF-MP2 pair contraction, small active space
    ("rect_df_med", 128, 512, 128),  # medium active space
    ("rect_df_large", 256, 512, 256),  # large active space
    # well above crossover
    ("sq1536", 1536, 1536, 1536),
    ("sq2048", 2048, 2048, 2048),
    ("sq4096", 4096, 4096, 4096),
]
SHAPES.sort(key=lambda s: s[1] * s[2] * s[3])

WARMS = 5  # warm-up passes
REPEATS = 10  # timed passes per shape


def time_pytorch(A: torch.Tensor, B: torch.Tensor) -> float:
    """Return mean wall-clock seconds for torch.matmul over REPEATS runs."""
    # One extra cold pass
    _ = torch.matmul(A, B)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        _ = torch.matmul(A, B)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def _native() -> bool:
    from trntensor.nki.dispatch import _get_dispatch_path

    return _get_dispatch_path() == "native"


def time_nki(A: torch.Tensor, B: torch.Tensor) -> float:
    """Return mean wall-clock seconds for nki_matmul over REPEATS runs."""
    from trntensor.nki.dispatch import _to_xla

    (a_x, b_x), orig = _to_xla(A.contiguous(), B.contiguous())

    from trntensor.nki._kernels import matmul_kernel

    # Warm passes (compile + cache)
    for _ in range(WARMS):
        c = matmul_kernel(a_x, b_x)
        _ = c.to(orig)

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        c = matmul_kernel(a_x, b_x)
        _ = c.to(orig)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def flops(M: int, K: int, N: int) -> int:
    """Multiply-add count M·K·N for (M,K)×(K,N) — what dispatch compares."""
    return M * K * N


def _timed(fn, sync=None) -> float:
    """Median wall-clock ms over REPEATS runs, after WARMS warm-up runs."""
    import statistics

    for _ in range(WARMS):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def _recommend(rows: list[dict], key_win: str) -> tuple[int | None, int | None]:
    """(first winning FLOPs, recommended threshold) for rows sorted by FLOPs.

    Conservative: the threshold sits just above the last losing shape before
    the first win, rounded up to a clean number.
    """
    import math

    first = next((r["flops"] for r in rows if r[key_win]), None)
    if first is None:
        return None, None
    prev = None
    for r in rows:
        if r["flops"] == first:
            break
        prev = r
    rec = prev["flops"] if prev else first
    magnitude = 10 ** int(math.log10(rec))
    return first, (rec // magnitude + 1) * magnitude


def native_main(args) -> None:
    """Native (TorchNeuron) sweep: derive both native thresholds."""
    from trntensor.nki.dispatch import _get_native_ops

    op = _get_native_ops()["matmul"]
    sync = torch.neuron.synchronize
    dev = "neuron"
    hdr = ("host", "aten_off", "nki_off", "aten_dev", "nki_dev")
    print(f"{'Shape':16s} {'M·K·N':>14s} " + " ".join(f"{h + ' ms':>11s}" for h in hdr))
    print("-" * 80)
    rows: list[dict] = []
    for label, M, K, N in SHAPES:
        torch.manual_seed(0)
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        a_d, b_d = A.to(dev), B.to(dev)
        sync()
        t = {
            "host": _timed(lambda A=A, B=B: torch.matmul(A, B)),
            "aten_off": _timed(lambda A=A, B=B: torch.matmul(A.to(dev), B.to(dev)).cpu()),
            "nki_off": _timed(lambda A=A, B=B: op(A.to(dev), B.to(dev)).cpu()),
            "aten_dev": _timed(lambda a=a_d, b=b_d: torch.matmul(a, b), sync),
            "nki_dev": _timed(lambda a=a_d, b=b_d: op(a, b), sync),
        }
        f = flops(M, K, N)
        row = {"label": label, "M": M, "K": K, "N": N, "flops": f}
        row.update({f"{k}_ms": v for k, v in t.items()})
        row["offload_wins"] = min(t["aten_off"], t["nki_off"]) < t["host"]
        row["nki_wins_pinned"] = t["nki_dev"] < t["aten_dev"]
        rows.append(row)
        print(f"  {label:14s} {f:14,d} " + " ".join(f"{t[h]:11.3f}" for h in hdr))

    from trntensor.nki import dispatch

    first_off, rec_off = _recommend(rows, "offload_wins")
    first_nki, rec_nki = _recommend(rows, "nki_wins_pinned")
    print()
    print("Dispatch path:           native")
    print(f"Device beats host from:  {first_off}")
    print(f"NKI beats aten from:     {first_nki if first_nki else 'never'}")
    print(f"Recommended TRNTENSOR_MIN_NKI_FLOPS_NATIVE={rec_off}")
    print(f"Recommended TRNTENSOR_MIN_NKI_FLOPS_NATIVE_PINNED={rec_nki or 'never'}")
    cur = (dispatch._MIN_NKI_FLOPS_NATIVE, dispatch._MIN_NKI_FLOPS_NATIVE_PINNED)
    print(f"Current (unpinned, pinned): {cur}")

    if args.write_cache:
        cache_path = Path(os.environ.get("TRNTENSOR_AUTOTUNE_CACHE", args.cache_path))
        try:
            cache = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            cache = {}
        if rec_off is not None:
            cache["trntensor_min_nki_flops_native"] = rec_off
        cache["trntensor_min_nki_flops_native_pinned"] = rec_nki if rec_nki else "never"
        cache["shapes_measured_native"] = rows
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"Wrote calibration cache to: {cache_path}")
    if first_off is None:
        print("The device never beat host torch — investigate.")
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--write-cache",
        action="store_true",
        help="Write calibrated threshold to autotune cache file",
    )
    ap.add_argument(
        "--cache-path",
        default="/var/tmp/trntensor-autotune/threshold.json",
        help="Cache file path (default: /var/tmp/trntensor-autotune/threshold.json)",
    )
    args = ap.parse_args()

    try:
        from trntensor.nki.dispatch import HAS_NKI
    except ImportError:
        HAS_NKI = False

    if not HAS_NKI:
        print(
            "HAS_NKI is False — NKI runtime not available. "
            "Run on a Trainium instance with the Neuron SDK installed."
        )
        raise SystemExit(1)

    if _native():
        native_main(args)
        return

    forced = os.environ.get("TRNTENSOR_FORCE_BACKEND", "").lower()
    if forced != "nki":
        print(
            "WARNING: TRNTENSOR_FORCE_BACKEND is not 'nki'. "
            "Set TRNTENSOR_FORCE_BACKEND=nki to ensure NKI path is exercised."
        )

    print(
        f"{'Shape':20s}  {'M':5s} {'K':5s} {'N':5s}  {'FLOPs':12s}  "
        f"{'PyTorch ms':12s}  {'NKI ms':10s}  {'Winner':8s}  Speedup"
    )
    print("-" * 90)

    first_nki_win_flops: int | None = None
    rows: list[dict] = []

    for label, M, K, N in SHAPES:
        torch.manual_seed(0)
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        f = flops(M, K, N)

        try:
            pt_ms = time_pytorch(A, B) * 1000
        except Exception as exc:
            print(f"  {label:20s}  PyTorch ERROR: {exc}")
            continue

        try:
            nki_ms = time_nki(A, B) * 1000
        except Exception as exc:
            print(f"  {label:20s}  NKI ERROR: {exc}")
            continue

        winner = "NKI" if nki_ms < pt_ms else "PyTorch"
        speedup = pt_ms / nki_ms

        print(
            f"  {label:20s}  {M:5d} {K:5d} {N:5d}  {f:12,d}  "
            f"{pt_ms:12.3f}  {nki_ms:10.3f}  {winner:8s}  {speedup:.2f}x"
        )

        rows.append(
            {
                "label": label,
                "M": M,
                "K": K,
                "N": N,
                "flops": f,
                "pytorch_ms": pt_ms,
                "nki_ms": nki_ms,
                "winner": winner,
            }
        )

        if winner == "NKI" and first_nki_win_flops is None:
            first_nki_win_flops = f

    print()
    if first_nki_win_flops is None:
        print(
            "NKI never won — all sizes below crossover. Increase max shape or investigate overhead."
        )
        raise SystemExit(1)

    # Conservative: back off one step to avoid borderline cases
    prev_row = None
    for row in rows:
        if row["flops"] == first_nki_win_flops:
            break
        prev_row = row

    recommended = prev_row["flops"] if prev_row else first_nki_win_flops
    # Round up to a clean number
    import math

    magnitude = 10 ** int(math.log10(recommended))
    recommended = (recommended // magnitude + 1) * magnitude

    from trntensor.nki import dispatch

    env_var = "TRNTENSOR_MIN_NKI_FLOPS"
    cache_key = "trntensor_min_nki_flops"
    current = dispatch._MIN_NKI_FLOPS
    print("Dispatch path:           xla")
    print(f"First NKI win at:        {first_nki_win_flops:,d} FLOPs")
    print(f"Recommended threshold:   {recommended:,d} FLOPs")
    print(f"Current {env_var}: {current:,d}")
    if recommended != current:
        delta_pct = (recommended - current) / current * 100
        sign = "+" if delta_pct > 0 else ""
        print(f"  → {sign}{delta_pct:.0f}% vs current value")
    else:
        print("  → matches current value (no change needed)")

    print()
    print(f"To apply:  export {env_var}={recommended}")

    if args.write_cache:
        cache_path = Path(os.environ.get("TRNTENSOR_AUTOTUNE_CACHE", args.cache_path))
        suffix = ""
        try:
            cache = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            cache = {}
        cache.update(
            {
                cache_key: recommended,
                f"first_nki_win_flops{suffix}": first_nki_win_flops,
                f"shapes_measured{suffix}": rows,
            }
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"Wrote calibration cache to: {cache_path}")


if __name__ == "__main__":
    main()
