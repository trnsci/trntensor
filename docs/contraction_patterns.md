# Contraction patterns: cuTENSOR ↔ trntensor

This page maps cuTENSOR concepts to their trntensor equivalents and explains
where the two libraries differ architecturally. The short version: cuTENSOR
hides the kernel boundary behind an opaque `Plan` handle; trntensor exposes it
as the primary design surface.

## Concept map

| cuTENSOR | trntensor |
|---|---|
| `cutensorInitContractionDescriptor` + `cutensorContractionExecute` | `einsum(subscripts, A, B)` |
| `cutensorPlan` (opaque per-contraction handle) | `ContractionPlan` (inspectable dataclass) |
| `cutensorContractionFind` (workspace-size query) | `plan_contraction()` + `.backend` |
| Multiple `Plan` objects with HBM intermediates | `multi_einsum()` with shared-operand pre-pinning |
| `cutensorDecompositionCreate` (Tucker HOSVD) | `tucker_decompose()` |
| Custom THC / Cholesky-factored ERIs | `cp_decompose()` (tensor hypercontraction) |
| CUDA stream + `cudaMemcpy` to pin device memory | `to_device(tensor)` / `from_device(tensor)` (formerly `to_xla` / `from_xla`) |

## 2-Operand contractions — the generic path

```python
import trntensor

# Matrix multiply: aten on the native path (Trn2/Trn3); NKI matmul kernel on
# the XLA path (Trn1) when FLOPs ≥ 2 GFLOPs
C = trntensor.einsum("ij,jk->ik", A, B)

# Batched matmul (only this exact index order plans as bmm)
C = trntensor.einsum("bij,bjk->bik", A, B)

# Transposed / permuted outputs run through torch.einsum. Releases through
# 0.21.0 returned the wrong layout for these on every backend (#64).
Ct = trntensor.einsum("ij,jk->ki", A, B)

# DF-MP2 pair contraction (ap,bp->ab  ≡  B[i] @ B[j].T)
T = trntensor.einsum("ap,bp->ab", B_i, B_j)

# Anything else falls through to torch.einsum
X = trntensor.einsum("ijk,klm->ijlm", A, B)
```

### Inspecting the plan

`ContractionPlan` is a regular dataclass — not an opaque handle:

```python
plan = trntensor.plan_contraction("ij,jk->ik", A, B)
print(plan.strategy)  # "matmul" | "bmm" | "torch" | "path"
print(plan.backend)  # "nki" | "pytorch"
print(trntensor.estimate_flops("ij,jk->ik", A, B))  # multiply-add pairs
print(plan.contraction_path)  # [(i,j), ...] for 3+ operand einsums
```

`plan.backend` reflects what will actually run — not what the algorithm
prefers. A 64×64 matmul is `strategy="matmul"` but `backend="pytorch"` because
the per-dispatch overhead exceeds the kernel work at that size. On the native
path every matmul/bmm reports `"pytorch"` by default: aten beat the NKI matmul
kernels at every measured size.

### Backend overrides

```bash
# Force PyTorch path regardless of size (benchmarking baseline)
TRNTENSOR_FORCE_BACKEND=pytorch python script.py

# XLA path: lower the FLOP threshold (default 2 GFLOPs)
TRNTENSOR_MIN_NKI_FLOPS=500000000 python script.py
```

## Named fused primitives — the Trainium-native path

cuTENSOR's model is one `Plan` per contraction. For a DF-MP2 energy computation
that chains five operations, that means five kernel launches with HBM
intermediates between them.

trntensor's named primitives span a contraction DAG in a single `@nki.jit`
program:

```python
# Full DF-MP2 pipeline from AO integrals to correlation energy — two dispatches,
# no HBM intermediate visible to Python.

B = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
# Inside: two GEMM stages, kernel-scratch HBM for the intermediate.
# Accepts packed (μ≥ν) integrals.

E = trntensor.mp2_energy(B, eps_occ, eps_vir)
# Inside: T in PSUM/SBUF, Vector Engine elementwise denominator, SBUF
# accumulator. One launch (XLA) or one launch per occupied row (native).
```

Neither `T_ijab` nor the per-pair `(i,ν)` intermediate appear as Python
tensors. In cuTENSOR terms, this is three plans fused into one — a pattern
cuTENSOR's `Plan` abstraction cannot name.

The same idea covers DF-SCF: `df_exchange(B, C_occ)` and `df_coulomb(B, D)`
build K and J straight from packed integrals without forming the
`(n, n, naux)` square.

### Current shape constraints

The `ao_to_mo_transform` kernels need
`nocc ≤ 128`; `"auto"` also keeps aten (native) for problems too small to
beat the launch floor. See [Quantum-chemistry primitives](api/quantum.md) for
the current routing and limits of each primitive.

## Multi-contraction: shared-operand pre-pinning

When the same tensor appears as input to multiple contractions, `multi_einsum`
detects it and transfers it to the Trainium device once:

```python
# Without multi_einsum: B transferred once per call (N transfers total)
results = [trntensor.einsum("ap,bp->ab", B[i], B[j]) for i in range(nocc) for j in range(nocc)]

# With multi_einsum: shared B[i], B[j] pre-pinned; each appears in only
# one contraction here, but B itself is the shared operand across a loop:
results = trntensor.multi_einsum(
    ("ap,bp->ab", B_i, B_j),  # direct term
    ("ap,bp->ab", B_j, B_i),  # exchange term — B_i and B_j are reused
)
```

`multi_einsum` inspects tensor object identity (`id()`). If the same tensor
object appears in two or more contractions, it is transferred once and the
device-resident copy is used for all subsequent contractions in the batch.

On CPU (no NKI), `multi_einsum` behaves exactly like a loop over `einsum()`.

## Device residency: `to_device` / `from_device`

The NKI dispatch path always transfers operands to the Trainium device
(`neuron` on TorchNeuron, the XLA device on Trn1) before kernel launch. For a
single large matmul this cost is negligible; for a tight loop of small
contractions it dominates. `to_device` / `from_device` expose the transfer
explicitly so callers can control when it happens. `to_xla` / `from_xla` are
the older names for the same thing and still work on both lineages:

```python
# Cold path: B transferred on every call (3 transfers for 3 calls)
for _ in range(3):
    result = trntensor.einsum("ij,jk->ik", A, B)

# Residency path: B transferred once, result stays on device until needed
A_x = trntensor.to_device(A)
B_x = trntensor.to_device(B)
for _ in range(3):
    result_x = trntensor.einsum("ij,jk->ik", A_x, B_x)  # zero transfer cost
result = trntensor.from_device(result_x)
```

The residency API maps to the CUDA concept of a device-pinned buffer, but with
explicit Python-level transfer rather than stream-implicit staging.

`to_device` (like `to_xla`) raises `RuntimeError` on hosts without `nki` installed.

## 3+ operand einsums: contraction-path search

For einsums with three or more operands, trntensor uses a greedy FLOP-cost path
search to choose the contraction order:

```python
# Naive: A(100,200) @ B(200,5) @ C(5,50) evaluated left-to-right costs
# A@B = 100×200×5 = 100k FLOPs → then ×C = 100×5×50 = 25k → 125k total.
# Greedy: B@C = 200×5×50 = 50k → then A×(BC) = 100×200×50 = 1M FLOPs.
# Hmm — actually left-to-right is better here. The planner finds the min.

C = trntensor.einsum("ij,jk,kl->il", A, B, C)

plan = trntensor.plan_contraction("ij,jk,kl->il", A, B, C)
print(plan.strategy)  # "path"
print(plan.contraction_path)  # [(0,1),(0,1)] — contract A@B first, then ×C
```

Each binary step in the path calls `einsum()` recursively and gets its own
backend routing (on the XLA path NKI if large enough, PyTorch otherwise; on
the native path aten).

## Decompositions

```python
# CP / PARAFAC — tensor hypercontraction (THC) of two-electron integrals.
# Reduces O(N^4) storage to O(N^2 R).
factors, weights = trntensor.cp_decompose(tensor, rank=10)
reconstructed = trntensor.cp_reconstruct(factors, weights)

# Tucker / HOSVD — low-rank approximation of DF coefficient tensors.
# Reduces memory for large auxiliary basis sets.
core, factors = trntensor.tucker_decompose(tensor, ranks=(5, 5, 5))
reconstructed = trntensor.tucker_reconstruct(core, factors)
```

Both run on CPU. NKI-accelerated paths for large ranks are a follow-up.
