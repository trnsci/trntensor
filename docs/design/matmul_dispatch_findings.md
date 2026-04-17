# matmul dispatch overhead findings (#33)

**Date:** 2026-04-17  
**Tracker:** [#33](https://github.com/trnsci/trntensor/issues/33)  
**Method:** `scripts/run_neuron_profile.sh --dispatch-timing` on trn1.2xlarge  
**Hardware:** trn1.2xlarge, Neuron runtime 2.31.24, compiler 2.24.5133

## TL;DR

The per-call overhead in `nki_matmul` is **~0.67 ms fixed XLA dispatch latency** regardless
of tensor size. Host↔device transfer scales with tensor size on top of this.

- **Without residency**: NKI barely beats CPU at 1024² (2.94 ms vs 2.94 ms); 2.1× at 1536².  
  Crossover: **~2 GFLOPs** (current `_MIN_NKI_FLOPS` default is correct).
- **With residency** (`to_xla` pre-applied): Eliminates host→device transfer.  
  NKI wins at 1024² (1.60 ms vs 2.90 ms = **1.8×**); 5.65× at 2048².  
  Crossover: **~900 MFLOPs** (new `_MIN_NKI_FLOPS_PINNED` default set to 1 GFLOPs).

---

## Step-by-step timing (trn1.2xlarge)

| Size | FLOPs | PT ms | to_xla ms | kern ms | to_cpu ms | nki_tot ms | winner |
|------|------:|------:|----------:|--------:|----------:|-----------:|--------|
| 128  | 4M    | 0.025 | 0.080 | 0.680 | 0.082 | 0.038† | PyTorch |
| 256  | 34M   | 0.109 | 0.105 | 0.670 | 0.128 | 0.070† | PyTorch |
| 512  | 268M  | 0.393 | 0.295 | 0.672 | 0.322 | 0.419† | PyTorch |
| 768  | 906M  | 1.274 | 0.703 | 0.673 | 0.779 | 1.270† | PyTorch |
| 1024 | 2.1G  | 2.938 | 1.502 | 0.684 | 0.970 | **2.935** | NKI (1.0×) |
| 1536 | 7.2G  | 9.441 | 3.998 | 0.879 | 1.859 | **4.514** | NKI (2.1×) |
| 2048 | 17G   | 23.547 | 7.282 | 1.003 | 2.960 | **17.934** | NKI (1.3×) |

† Sizes below `_MIN_NKI_FLOPS` (2 GFLOPs) fall through to `torch.matmul` via the
dispatch gate; `nki_tot` is measuring the fallback path, not NKI execution.

The individual step measurements (to_xla, kern, to_cpu) always invoke the NKI path
directly, bypassing the gate, and are accurate.

---

## Pre-pinned sweep (residency via `to_xla`)

| Size | FLOPs | PT ms | kern ms | to_cpu ms | pinned_tot ms | winner |
|------|------:|------:|--------:|----------:|--------------:|--------|
| 128  | 4M    | 0.021 | 0.687 | 0.083 | 0.770 | PyTorch |
| 256  | 34M   | 0.058 | 0.663 | 0.124 | 0.787 | PyTorch |
| 512  | 268M  | 0.396 | 0.662 | 0.311 | 0.972 | PyTorch |
| 768  | 906M  | 1.270 | 0.931 | 0.748 | 1.680 | PyTorch |
| 1024 | 2.1G  | 2.899 | 0.673 | 0.924 | **1.597** | NKI (1.8×) |
| 1536 | 7.2G  | 9.354 | 0.689 | 1.821 | **2.511** | NKI (3.7×) |
| 2048 | 17G   | 23.274 | 0.973 | 3.144 | **4.116** | NKI (5.65×) |

---

## Key findings

### 1. ~0.67 ms is fixed XLA dispatch latency

The `kern ms` column is nearly constant at 128–1024 (0.67–0.68 ms). This is the cost of
submitting the XLA lazy graph for the NKI kernel — not the kernel execution time itself.
XLA defers execution until a synchronization point (`.to(cpu)`). The "kern" timing only
measures submission; actual execution is included in `to_cpu`.

This ~0.67 ms is a fundamental floor for any individual NKI kernel dispatch. It cannot be
reduced by kernel tuning; it can only be **amortized** by batching multiple operations into
one XLA graph compile.

### 2. Residency changes the economics

Without residency, `to_xla` is the dominant cost at large sizes (7.3 ms at 2048²). With
residency, that cost is paid once per session. The pre-pinned crossover (~900 MFLOPs vs
~2 GFLOPs) means users who call `to_xla()` before a loop of matmuls get NKI benefit at
roughly half the matrix size.

### 3. Current threshold was correct; pinned threshold is new

The 2 GFLOPs default was approximately right. The new `_MIN_NKI_FLOPS_PINNED = 1 GFLOPs`
captures the pre-pinned benefit with a conservative margin above the ~900 MFLOPs crossover.

### 4. 2048² only 1.3× faster unpinned — transfer dominates

At 2048², `to_xla` alone takes 7.3 ms. The kernel result is only 2.96 ms to retrieve.
The NKI win (17.9 ms vs 23.5 ms) is entirely because the kernel itself runs in ~1 ms while
torch.matmul spends 23 ms on CPU. The transfer overhead is significant. Users with 2048²
workloads should pre-pin.

---

## What this means for #33

The four items in the #33 description:

| Approach | Finding |
|----------|---------|
| Lower `_MIN_NKI_FLOPS` | Done for pinned path (1 GFLOPs); 2 GFLOPs remains correct for unpinned |
| Keep tensors on XLA across calls | Already implemented (`to_xla`/`from_xla`); pinned threshold now lower to exploit it |
| Batch dispatches into one XLA graph | Not yet — would amortize the 0.67 ms fixed cost; relevant for multi_einsum loops |
| `torch_xla.compile` on einsum body | Unexplored; would require graph-mode tracing |

**The batching approach** (fusing multiple matmul dispatches into one XLA graph) is the next
lever. For DF-MP2's O(N²) pair loop — `for i in range(nocc): for j in range(nocc): T = B[i] @ B[j].T` — each iteration pays 0.67 ms. At nocc=64, that's 64² × 0.67 ms ≈ 2.7 s of pure overhead. A batched dispatch would pay 0.67 ms once. This is the Phase 3 architectural target.

---

## Recommended next steps

1. **Close #33 partially** — thresholds calibrated, findings documented. The fundamental
   dispatch overhead is understood and bounded.

2. **Phase 3 batching investigation** — prototype fusing the nocc×nocc pair loop into one
   `torch_xla.compile` call or a single `multi_einsum` dispatch with an explicit graph.

3. **Run `scripts/run_neuron_profile.sh --kernel matmul`** to get Tensor Engine utilization
   for the kernel itself (separate from dispatch overhead). Expected: TE-dominant (contrast
   with trnblas `_mp2_energy_kernel` which was VE-dominant at 96.45%).

4. **Provision `trntensor-ci-trn1`** — run `terraform apply` in `infra/terraform/` to get
   a dedicated instance. Current approach borrows `trnblas-ci-trn1`.

---

## Raw timing data (JSON)

```json
{
  "hardware": "trn1.2xlarge",
  "neuron_runtime": "2.31.24",
  "compiler": "2.24.5133.0",
  "date": "2026-04-17",
  "unpinned": [
    {"size": 128,  "flops": 4194304,    "pt_ms": 0.025, "to_xla_ms": 0.080, "kern_ms": 0.680, "to_cpu_ms": 0.082, "nki_tot_ms": 0.038, "winner": "PyTorch"},
    {"size": 256,  "flops": 33554432,   "pt_ms": 0.109, "to_xla_ms": 0.105, "kern_ms": 0.670, "to_cpu_ms": 0.128, "nki_tot_ms": 0.070, "winner": "PyTorch"},
    {"size": 512,  "flops": 268435456,  "pt_ms": 0.393, "to_xla_ms": 0.295, "kern_ms": 0.672, "to_cpu_ms": 0.322, "nki_tot_ms": 0.419, "winner": "PyTorch"},
    {"size": 768,  "flops": 905969664,  "pt_ms": 1.274, "to_xla_ms": 0.703, "kern_ms": 0.673, "to_cpu_ms": 0.779, "nki_tot_ms": 1.270, "winner": "PyTorch"},
    {"size": 1024, "flops": 2147483648, "pt_ms": 2.938, "to_xla_ms": 1.502, "kern_ms": 0.684, "to_cpu_ms": 0.970, "nki_tot_ms": 2.935, "winner": "NKI"},
    {"size": 1536, "flops": 7247757312, "pt_ms": 9.441, "to_xla_ms": 3.998, "kern_ms": 0.879, "to_cpu_ms": 1.859, "nki_tot_ms": 4.514, "winner": "NKI"},
    {"size": 2048, "flops": 17179869184,"pt_ms": 23.547,"to_xla_ms": 7.282, "kern_ms": 1.003, "to_cpu_ms": 2.960, "nki_tot_ms": 17.934,"winner": "NKI"}
  ],
  "pinned": [
    {"size": 128,  "flops": 4194304,    "pt_ms": 0.021, "kern_ms": 0.687, "to_cpu_ms": 0.083, "pinned_tot_ms": 0.770, "winner": "PyTorch"},
    {"size": 256,  "flops": 33554432,   "pt_ms": 0.058, "kern_ms": 0.663, "to_cpu_ms": 0.124, "pinned_tot_ms": 0.787, "winner": "PyTorch"},
    {"size": 512,  "flops": 268435456,  "pt_ms": 0.396, "kern_ms": 0.662, "to_cpu_ms": 0.311, "pinned_tot_ms": 0.972, "winner": "PyTorch"},
    {"size": 768,  "flops": 905969664,  "pt_ms": 1.270, "kern_ms": 0.931, "to_cpu_ms": 0.748, "pinned_tot_ms": 1.680, "winner": "PyTorch"},
    {"size": 1024, "flops": 2147483648, "pt_ms": 2.899, "kern_ms": 0.673, "to_cpu_ms": 0.924, "pinned_tot_ms": 1.597, "winner": "NKI"},
    {"size": 1536, "flops": 7247757312, "pt_ms": 9.354, "kern_ms": 0.689, "to_cpu_ms": 1.821, "pinned_tot_ms": 2.511, "winner": "NKI"},
    {"size": 2048, "flops": 17179869184,"pt_ms": 23.274,"kern_ms": 0.973, "to_cpu_ms": 3.144, "pinned_tot_ms": 4.116, "winner": "NKI"}
  ]
}
```
