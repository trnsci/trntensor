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

The --write-cache flag writes the recommendation to
/var/tmp/trntensor-autotune/threshold.json so dispatch.py can read it
on next start (pass TRNTENSOR_AUTOTUNE_CACHE to override the path).

Exit code 1 if the NKI crossover was never observed (NKI loses at all
tested sizes — indicates a regression or configuration problem).
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
]

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
    """Multiply-add count for (M,K)×(K,N) matmul."""
    return 2 * M * K * N


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

    current = int(os.environ.get("TRNTENSOR_MIN_NKI_FLOPS", "2000000000"))
    print(f"First NKI win at:        {first_nki_win_flops:,d} FLOPs")
    print(f"Recommended threshold:   {recommended:,d} FLOPs")
    print(f"Current TRNTENSOR_MIN_NKI_FLOPS: {current:,d}")
    if recommended != current:
        delta_pct = (recommended - current) / current * 100
        sign = "+" if delta_pct > 0 else ""
        print(f"  → {sign}{delta_pct:.0f}% vs current value")
    else:
        print("  → matches current value (no change needed)")

    print()
    print(f"To apply:  export TRNTENSOR_MIN_NKI_FLOPS={recommended}")

    if args.write_cache:
        cache = {
            "trntensor_min_nki_flops": recommended,
            "first_nki_win_flops": first_nki_win_flops,
            "shapes_measured": rows,
        }
        cache_path = Path(os.environ.get("TRNTENSOR_AUTOTUNE_CACHE", args.cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"Wrote calibration cache to: {cache_path}")


if __name__ == "__main__":
    main()
