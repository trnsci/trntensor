#!/usr/bin/env python
"""Multi-rank reduce-parallel einsum check for trntensor (Phase 4).

Runs ``trntensor.einsum("ij,jk->ik", scatter(A, dim=1, n_shards=world), B)``
with one process per rank. The contracted index ``j`` is sharded across
ranks. Rank ``r`` computes only its partial ``A[:, J_r] @ B[J_r, :]`` on its
own NeuronCore, and the partials are summed by the all-reduce backend that
:func:`trntensor.parallel.allreduce_backend` selects (``TRNTENSOR_ALLREDUCE``):

  mock   single-process sum (every rank computes every partial)
  torch  torch.distributed.all_reduce(SUM) through the c10d backend
  nki    nki.collectives.all_reduce(add) on shared_hbm (trntensor.nki.collectives)
  auto   nki under a "neuron" group, torch under gloo, mock at world 1

Each check, on every rank:
  * rel_err = ||C - C_single||_F / ||C_single||_F, where C_single is the dense
    single-device einsum on the same core. PASS when rel_err <= 1e-5 (fp32).
  * bit-exactness against a fixed-order sum ((p0 + p1) + p2) + p3 of the
    per-shard partials, and against torch.stack(partials).sum(0) (the mock).
    Reported, not gated: the all-reduce may sum in a different order.
  * rel error against an fp64 host reference (reported).
  * sha256 of C, so the launcher can confirm every rank holds the same bytes
    (gated: ranks must agree).
Timings: cold (includes NEFF compile) and warm median of the sharded einsum,
the bare all-reduce of one partial, and the dense single-device einsum.

Usage (repo root; the script is its own launcher and worker):

    # Trn2, LNC=2: world 2 on cores 2-3, world 4 on cores 0-3
    python scripts/run_reduce_parallel_multirank.py --world 1,2 --core-start 2
    python scripts/run_reduce_parallel_multirank.py --world 4 --core-start 0
    # Mac / CPU pre-flight (gloo; nki mode is skipped)
    python scripts/run_reduce_parallel_multirank.py --cpu --world 1,2,4

Launch pattern (proven by trnsci spike B4, SDK 2.33 / NKI 0.7.0): each rank
sets ``NEURON_RT_VISIBLE_CORES=<core_start + local_rank>`` before importing
torch, the parent's RANK/WORLD_SIZE/MASTER_*/NEURON_RT_* variables are
stripped, ``NEURON_LOGICAL_NC_CONFIG`` defaults to 2,
``dist.init_process_group("neuron")``, ``torch.neuron.set_device(rank)``,
device ``neuron:<rank>``. Rank cores must form an aligned pair or quad
(0-1, 2-3, 0-3 work; 1-2 fails with "no_hier no_mesh"). The launcher refuses
unaligned layouts unless ``--allow-unaligned`` is given.

Each (world, mode) job is a fresh set of processes with a hard timeout.
Per-rank logs and JSON results go to ``--logdir``.
"""

import os
import sys

_WORKER_FLAG = "TRNTENSOR_RP_WORKER"
_IS_WORKER = os.environ.get(_WORKER_FLAG) == "1"
_CPU = "--cpu" in sys.argv or os.environ.get("TRNTENSOR_RP_DEVICE") == "cpu"
if _IS_WORKER and not _CPU:
    # Pin this rank to its own logical core BEFORE torch / torch_neuronx import.
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(
        int(os.environ.get("TRNTENSOR_RP_CORE_START", "0")) + int(os.environ["LOCAL_RANK"])
    )

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import timedelta  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL_TOL = 1e-5
SUBSCRIPTS = "ij,jk->ik"


# ------------------------------------------------------------------ worker
class Worker:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.mode = os.environ["TRNTENSOR_ALLREDUCE"]
        self.sizes = [int(s) for s in os.environ["TRNTENSOR_RP_SIZES"].split(",")]
        self.reps = int(os.environ.get("TRNTENSOR_RP_REPS", "5"))
        self.result_file = os.environ["TRNTENSOR_RP_RESULT"]
        self.results = []
        self.dev = "cpu"

    def log(self, msg):
        print(f"[r{self.rank}/{self.world} {self.mode}] {msg}", flush=True)

    def flush(self):
        with open(self.result_file, "w") as fh:
            json.dump(
                dict(rank=self.rank, world=self.world, mode=self.mode, results=self.results), fh
            )

    def record(self, **res):
        self.results.append(res)
        self.log(json.dumps(res))
        self.flush()

    def sync(self):
        if not _CPU:
            import torch_neuronx

            torch_neuronx.synchronize()

    def timed(self, fn):
        t0 = time.perf_counter()
        out = fn()
        self.sync()
        cold = (time.perf_counter() - t0) * 1e3
        warm = []
        for _ in range(self.reps):
            t0 = time.perf_counter()
            out = fn()
            self.sync()
            warm.append((time.perf_counter() - t0) * 1e3)
        return out, cold, (statistics.median(warm) if warm else None)

    def setup(self):
        import torch
        import torch.distributed as dist

        if _CPU:
            dist.init_process_group("gloo", timeout=timedelta(seconds=120))
        else:
            import torch_neuronx  # registers the "neuron" c10d backend

            dist.init_process_group("neuron", timeout=timedelta(seconds=120))
            torch.neuron.set_device(self.rank)
            self.dev = f"neuron:{self.rank}"
            from torch_neuronx.utils import get_logical_neuron_cores

            self.log(
                f"torch={torch.__version__} torch_neuronx={torch_neuronx.__version__} "
                f"lnc={get_logical_neuron_cores()} "
                f"NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES')}"
            )
        assert dist.get_rank() == self.rank and dist.get_world_size() == self.world
        self.log(f"process group up: backend={dist.get_backend()} device={self.dev}")

    def check_size(self, n):
        import torch

        import trntensor
        from trntensor.parallel import (
            ShardedTensor,
            _dist_allreduce,
            allreduce_backend,
            scatter,
            shard_boundaries,
        )

        g = torch.Generator().manual_seed(1234 + n)
        A = torch.randn(n, n, generator=g, dtype=torch.float32)
        B = torch.randn(n, n, generator=g, dtype=torch.float32)
        A_dev, B_dev = A.to(self.dev), B.to(self.dev)

        backend = allreduce_backend(self.world)
        st = scatter(A, dim=1, n_shards=self.world)
        rank_local = backend in ("torch", "nki")
        # Rank-local backends read only chunk `rank`; the rest stay on the host.
        chunks = [
            c.to(self.dev) if (not rank_local or i == self.rank) else c
            for i, c in enumerate(st.chunks)
        ]
        A_sh = ShardedTensor(chunks, partition_dim=1)

        C, cold, warm = self.timed(lambda: trntensor.einsum(SUBSCRIPTS, A_sh, B_dev))
        C_single, s_cold, s_warm = self.timed(lambda: trntensor.einsum(SUBSCRIPTS, A_dev, B_dev))

        # Per-shard partials, all on this device, for the fixed-order references.
        bounds = shard_boundaries(st)
        partials = [
            trntensor.einsum(SUBSCRIPTS, st.chunks[i].to(self.dev), B_dev[s:e])
            for i, (s, e) in enumerate(bounds)
        ]
        fixed = partials[0].clone()
        for p in partials[1:]:
            fixed = fixed + p
        mock = torch.stack(partials).sum(0)

        ar_cold = ar_warm = None
        if backend == "torch":
            _, ar_cold, ar_warm = self.timed(lambda: _dist_allreduce(partials[self.rank]))
        elif backend == "nki":
            from trntensor.nki.collectives import nki_all_reduce_sum

            _, ar_cold, ar_warm = self.timed(lambda: nki_all_reduce_sum(partials[self.rank]))

        c = C.detach().to("cpu")
        cs = C_single.detach().to("cpu")
        ref64 = A.double() @ B.double()

        def rel(x, ref):
            ref = ref.double()
            return float((x.double() - ref).norm() / ref.norm())

        rel_single = rel(c, cs)
        res = dict(
            n=n,
            backend=backend,
            shape=list(c.shape),
            rel_err_vs_single=rel_single,
            rel_err_vs_fp64=rel(c, ref64),
            single_rel_err_vs_fp64=rel(cs, ref64),
            bitexact_vs_fixed_order=bool(torch.equal(c, fixed.detach().to("cpu"))),
            bitexact_vs_mock=bool(torch.equal(c, mock.detach().to("cpu"))),
            bitexact_vs_single=bool(torch.equal(c, cs)),
            max_abs_diff_vs_fixed=float(
                (c.double() - fixed.detach().to("cpu").double()).abs().max()
            ),
            sha256=hashlib.sha256(c.contiguous().numpy().tobytes()).hexdigest()[:16],
            cold_ms=cold,
            warm_ms=warm,
            allreduce_cold_ms=ar_cold,
            allreduce_warm_ms=ar_warm,
            single_cold_ms=s_cold,
            single_warm_ms=s_warm,
            status=("PASS" if rel_single <= REL_TOL and tuple(c.shape) == (n, n) else "FAIL"),
        )
        self.record(**res)

    def run(self):
        self.flush()
        try:
            self.setup()
        except Exception as e:
            traceback.print_exc()
            self.record(n=None, status="ERROR", detail=f"setup: {type(e).__name__}: {str(e)[:300]}")
            return 1
        rc = 0
        for n in self.sizes:
            try:
                self.check_size(n)
            except Exception as e:
                traceback.print_exc()
                first = (str(e).strip().splitlines() or [type(e).__name__])[0]
                self.record(n=n, status="ERROR", detail=f"{type(e).__name__}: {first[:300]}")
                rc = 1
                # A rank that errored skips later collectives; peers would hang on
                # them, so stop here and let the launcher's peer grace clean up.
                break
        try:
            import torch.distributed as dist

            dist.destroy_process_group()
        except Exception as e:
            self.log(f"destroy_process_group: {type(e).__name__}: {e}")
        return rc


# ------------------------------------------------------------------ launcher
_STRIP_CHILD_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "NEURON_RT_VISIBLE_CORES",
    "NEURON_RT_NUM_CORES",
    "NEURON_RT_ROOT_COMM_ID",
)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill(procs):
    for p in procs:
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, signal.SIGKILL)
    for p in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(timeout=10)


def run_job(args, world, mode):
    env_base = {k: v for k, v in os.environ.items() if k not in _STRIP_CHILD_ENV}
    if _CPU:
        env_base["TRNTENSOR_RP_DEVICE"] = "cpu"
    else:
        env_base.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
    if args.backend != "auto":
        env_base["TRNTENSOR_FORCE_BACKEND"] = args.backend
    env_base["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT] + ([env_base["PYTHONPATH"]] if env_base.get("PYTHONPATH") else [])
    )
    env_base.update(
        {
            _WORKER_FLAG: "1",
            "TRNTENSOR_ALLREDUCE": mode,
            "TRNTENSOR_RP_SIZES": args.sizes,
            "TRNTENSOR_RP_REPS": str(args.reps),
            "TRNTENSOR_RP_CORE_START": str(args.core_start),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(free_port()),
            "WORLD_SIZE": str(world),
            "LOCAL_WORLD_SIZE": str(world),
        }
    )
    procs, logs, rfiles = [], [], []
    t_start = time.time()
    for r in range(world):
        tag = f"w{world}_{mode}_r{r}"
        rfile = os.path.join(args.logdir, tag + ".json")
        lfile = os.path.join(args.logdir, tag + ".log")
        if os.path.exists(rfile):
            os.remove(rfile)
        env = dict(env_base, RANK=str(r), LOCAL_RANK=str(r), TRNTENSOR_RP_RESULT=rfile)
        cmd = [sys.executable, os.path.abspath(__file__)] + (["--cpu"] if _CPU else [])
        with open(lfile, "w") as fh:
            procs.append(
                subprocess.Popen(
                    cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
                )
            )
        logs.append(lfile)
        rfiles.append(rfile)

    deadline = t_start + args.job_timeout
    note = ""
    while True:
        rcs = [p.poll() for p in procs]
        if all(rc is not None for rc in rcs):
            break
        now = time.time()
        bad = [rc for rc in rcs if rc not in (None, 0)]
        if bad and deadline > now + args.peer_grace:
            deadline = now + args.peer_grace
            note = f"a rank exited rc={bad[0]}; peers given {args.peer_grace:.0f}s"
        if now > deadline:
            hung = [r for r, rc in enumerate(rcs) if rc is None]
            note = (note + "; " if note else "") + (
                f"TIMEOUT after {now - t_start:.0f}s, killed ranks {hung}"
            )
            _kill(procs)
            break
        time.sleep(0.25)
    elapsed = time.time() - t_start

    per_rank = []
    for r, (p, lfile, rfile) in enumerate(zip(procs, logs, rfiles, strict=True)):
        data = None
        if os.path.exists(rfile):
            try:
                with open(rfile) as fh:
                    data = json.load(fh)
            except Exception:
                data = None
        per_rank.append(dict(rank=r, rc=p.returncode, data=data))
        with open(lfile, errors="replace") as fh:
            lines = fh.read().splitlines()
        if args.log_tail and (p.returncode != 0 or args.verbose):
            print(f"--- w{world} {mode} rank {r} rc={p.returncode} log tail ({lfile})")
            for line in lines[-args.log_tail :]:
                print(f"  [r{r}] {line}")
    return per_rank, note, elapsed


def _fmt(x, spec=".3f"):
    return "-" if x is None else format(x, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", default="1,2,4", help="comma list of world sizes (1, 2, 4)")
    ap.add_argument(
        "--core-start", type=int, default=0, help="first NeuronCore id (rank r -> core_start + r)"
    )
    ap.add_argument("--sizes", default="512,1024,2048", help="comma list of n for n x n x n")
    ap.add_argument(
        "--modes",
        default=None,
        help="comma list of TRNTENSOR_ALLREDUCE modes for world > 1 "
        "(default: neuron 'auto,torch,nki'; --cpu 'auto,torch,mock')",
    )
    ap.add_argument(
        "--backend",
        default="pytorch",
        choices=("pytorch", "nki", "auto"),
        help="TRNTENSOR_FORCE_BACKEND for the local partial einsum (default pytorch: "
        "torch.matmul on the rank's device, so only the all-reduce is under test)",
    )
    ap.add_argument("--cpu", action="store_true", help="gloo on CPU (pre-flight); nki mode skipped")
    ap.add_argument("--reps", type=int, default=5, help="warm repetitions (default 5)")
    ap.add_argument("--job-timeout", type=float, default=600.0, help="hard kill per job, s")
    ap.add_argument("--budget", type=float, default=2400.0, help="skip jobs after this many s")
    ap.add_argument("--peer-grace", type=float, default=60.0, help="kill peers s after a rank dies")
    ap.add_argument("--logdir", default="rp_multirank_logs")
    ap.add_argument("--log-tail", type=int, default=40)
    ap.add_argument("--verbose", action="store_true", help="echo rank log tails on success too")
    ap.add_argument("--allow-unaligned", action="store_true", help="permit e.g. cores 1-2")
    args = ap.parse_args()
    args.logdir = os.path.abspath(args.logdir)
    os.makedirs(args.logdir, exist_ok=True)

    worlds = [int(w) for w in args.world.split(",") if w]
    modes = [
        m for m in (args.modes or ("auto,torch,mock" if _CPU else "auto,torch,nki")).split(",") if m
    ]
    if _CPU and "nki" in modes:
        print("note: nki mode needs a 'neuron' process group; skipped under --cpu")
        modes = [m for m in modes if m != "nki"]
    if not _CPU and not args.allow_unaligned:
        for w in worlds:
            if w > 1 and args.core_start % w != 0:
                sys.exit(
                    f"cores {args.core_start}-{args.core_start + w - 1} are not an aligned "
                    f"{'pair' if w == 2 else 'group'} (core_start % world != 0); replica groups "
                    "like cores 1-2 fail ('no_hier no_mesh'). Use --core-start "
                    f"{args.core_start - args.core_start % w} or --allow-unaligned."
                )

    print(f"trntensor reduce-parallel multi-rank  device={'cpu (gloo)' if _CPU else 'neuron'}")
    print(
        f"einsum={SUBSCRIPTS!r} (j sharded) sizes={args.sizes} fp32 worlds={worlds} modes={modes} "
        f"local_backend={args.backend} reps={args.reps} core_start={args.core_start}"
    )
    print(f"job_timeout={args.job_timeout:.0f}s budget={args.budget:.0f}s logdir={args.logdir}")
    print(f"python={sys.executable} utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    t0 = time.time()
    rows = []  # summary rows
    failed = False
    for world in worlds:
        job_modes = ["auto"] if world == 1 else modes
        for mode in job_modes:
            name = f"w{world}/{mode}"
            if time.time() - t0 > args.budget:
                print(f"RESULT [{name}]: SKIP total budget exhausted")
                rows.append((name, "SKIP", "budget", None))
                failed = True
                continue
            print(f"\n===== {name} =====", flush=True)
            per_rank, note, elapsed = run_job(args, world, mode)
            by_n = {}
            for pr in per_rank:
                for res in (pr["data"] or {}).get("results", []):
                    by_n.setdefault(res.get("n"), {})[pr["rank"]] = res
            sizes = [int(s) for s in args.sizes.split(",")]
            for n in sizes:
                ranks = by_n.get(n, {})
                sts = [ranks.get(r, {}).get("status", "ERROR") for r in range(world)]
                shas = {ranks[r]["sha256"] for r in ranks if "sha256" in ranks[r]}
                agree = len(shas) == 1 and len(ranks) == world
                for r in range(world):
                    res = ranks.get(r)
                    if res is None:
                        print(
                            f"RESULT [{name}/n={n}/r{r}]: ERROR no result (rc={per_rank[r]['rc']}) {note}"
                        )
                        continue
                    if "rel_err_vs_single" not in res:
                        print(
                            f"RESULT [{name}/n={n}/r{r}]: {res['status']} {res.get('detail', '')}"
                        )
                        continue
                    print(
                        f"RESULT [{name}/n={n}/r{r}]: {res['status']} backend={res['backend']} "
                        f"rel_err_vs_single={res['rel_err_vs_single']:.3e} "
                        f"vs_fp64={res['rel_err_vs_fp64']:.3e} "
                        f"(single vs_fp64={res['single_rel_err_vs_fp64']:.3e}) "
                        f"bitexact[fixed_order={res['bitexact_vs_fixed_order']} "
                        f"mock={res['bitexact_vs_mock']} single={res['bitexact_vs_single']}] "
                        f"max|C-fixed|={res['max_abs_diff_vs_fixed']:.3e} "
                        f"einsum cold={_fmt(res['cold_ms'], '.1f')}ms warm={_fmt(res['warm_ms'])}ms "
                        f"allreduce warm={_fmt(res['allreduce_warm_ms'])}ms "
                        f"single warm={_fmt(res['single_warm_ms'])}ms sha={res['sha256']}"
                    )
                ok = all(s == "PASS" for s in sts) and agree
                agg = (
                    "PASS"
                    if ok
                    else ("FAIL" if "FAIL" in sts or (ranks and not agree) else "ERROR")
                )
                failed |= not ok
                detail = f"ranks={sts} ranks_agree={agree} job={elapsed:.0f}s" + (
                    f" {note}" if note else ""
                )
                print(f"RESULT [{name}/n={n}]: {agg} {detail}")
                r0 = ranks.get(0) if ranks.get(0, {}).get("warm_ms") is not None else None
                warm_max = (
                    max(ranks[r]["warm_ms"] for r in ranks if ranks[r].get("warm_ms") is not None)
                    if r0
                    else None
                )
                rows.append((f"{name}/n={n}", agg, detail, (r0, warm_max)))

    print("\n===== SUMMARY =====")
    print(
        f"{'job':24s} {'status':6s} {'backend':7s} {'rel_err':>9s} {'fixed=':>6s} "
        f"{'einsum_ms':>10s} {'allred_ms':>10s} {'single_ms':>10s}"
    )
    for name, st, _det, extra in rows:
        r0, warm_max = extra if extra else (None, None)
        if r0 is None:
            print(f"{name:24s} {st:6s}")
            continue
        print(
            f"{name:24s} {st:6s} {r0['backend']:7s} {r0['rel_err_vs_single']:9.2e} "
            f"{str(r0['bitexact_vs_fixed_order'])[0]:>6s} {_fmt(warm_max):>10s} "
            f"{_fmt(r0['allreduce_warm_ms']):>10s} {_fmt(r0['single_warm_ms']):>10s}"
        )
    print(
        "(einsum_ms = slowest rank's warm median of the sharded einsum; allred_ms = rank 0's "
        "bare all-reduce of one partial; single_ms = rank 0's dense single-device einsum)"
    )
    print(f"total {time.time() - t0:.0f}s; per-rank logs in {args.logdir}")
    print("OVERALL:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    if _IS_WORKER:
        sys.exit(Worker().run())
    sys.exit(main())
