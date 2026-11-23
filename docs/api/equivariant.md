# Equivariant contractions

Fixed-shape kernels for the two contractions that dominate a MACE-style
machine-learned interatomic potential (MLIP exemplar,
[trnsci/trnsci#170](https://github.com/trnsci/trnsci/issues/170); contract
[#138](https://github.com/trnsci/trntensor/issues/138)). Conservative MLIPs
compute forces as F = −∂E/∂positions, so each op has a forward and a
registered input gradient.

| op | shares of a CPU energy + forces step (MACE-MP-0b2 medium, 216 atoms) |
|---|---|
| `conv_tp` (interaction tensor product) | 56% |
| `symmetric_contraction` (product basis) | 29% |

## `conv_tp(x, y, w, spec) -> Tensor`

The e3nn `TensorProduct` with "uvu" instructions and per-edge weights. For
edge `e`, channel `u` and path `p = (l1 ⊗ l2 → l3, α_p)`:

```
out_p[e, u, k] = α_p · w[e, p, u] · Σ_ij C^{l1 l2 l3}_ijk · x_{l1}[e, u, i] · y_{l2}[e, j]
```

`C` are the real Wigner-3j symbols (e3nn basis and sign convention). Layouts
are e3nn's mul-major: an irreps block `mul × l` is `[mul, 2l+1]` row-major.
Gradients: `∂x`, `∂y`, `∂w`.

| | MACE-MP layer 0 | layer 1 |
|---|---|---|
| `x` | `[E, 128]` (128x0e) | `[E, 512]` (128x0e + 128x1o) |
| `y` | `[E, 16]` (0e + 1o + 2e + 3o) | `[E, 16]` |
| `w` | `[E, 512]`, 4 paths | `[E, 1280]`, 10 paths |
| out | `[E, 2048]` | `[E, 5120]` |

### `conv_tp_spec(irreps_in1=None, irreps_in2=None, irreps_out=None, instructions=None, *, module=None, w3j=None, irrep_normalization="component", path_normalization="element") -> ConvTPSpec`

Built once per layer.

- `module=`: an e3nn `TensorProduct`, or a MACE interaction block (its
  `conv_tp` is read). The irreps, instruction order (which is also the
  weight order) and e3nn's normalisation (`path_weight`) are read from it.
- Explicit irreps: e3nn strings (`"128x0e+128x1o"`) or `[(mul, l), ...]`, and
  instructions `(i1, i2, io)` (α from e3nn's "uvu" normalisation),
  `(i1, i2, io, α)`, or e3nn's `(i1, i2, io, "uvu", True)`.
- The exemplar's capture dict is accepted wherever a spec is
  (`ConvTPSpec.from_dict`, `spec.as_dict()`).

```python
ins = [(0, 0, 0), (1, 1, 1), (0, 1, 2), (1, 0, 3), (1, 2, 4)]
ins += [(0, 2, 5), (1, 1, 6), (1, 3, 7), (0, 3, 8), (1, 2, 9)]
irreps_out = "128x0e+128x0e+128x1o+128x1o+128x1o+128x2e+128x2e+128x2e+128x3o+128x3o"
spec = trntensor.conv_tp_spec("128x0e+128x1o", "1x0e+1x1o+1x2e+1x3o", irreps_out, ins)
out = trntensor.conv_tp(x, y, w, spec)  # [E, 5120]
# or from the model: trntensor.conv_tp_spec(module=model.interactions[1].conv_tp)
```

## `symmetric_contraction(x, species, spec) -> Tensor`

MACE's `SymmetricContraction`, correlation ν ≤ 3:

```
out[n, c, m] = Σ_ν Σ_k W_ν[z_n, k, c] · Σ_{a_1..a_ν} U_ν[m, a_1..a_ν, k] · Π_j x[n, c, a_j]
```

`x [N, C, 16]`, `species [N]` (int64 indices into the weight tables).
Returns `[N, Σ C·d_out]` in MACE's layout (output irreps in order, each
`[C, d_out]` row-major). Gradient: `∂x`.

### `symmetric_contraction_spec(*, module=None, contractions=None) -> SymmetricContractionSpec`

- `module=`: a MACE `SymmetricContraction` or product block. Each
  `Contraction`'s `U_matrix_ν` buffers and weights are read (`weights_max` is
  ν = correlation, `weights[j]` is ν = correlation − 1 − j).
- `contractions=`: one mapping per output irrep with `d_out`, `correlation`,
  `U` (`{ν: [d_out, 16, …, 16, K_ν]}`, dense or COO) and `W` (`{ν: [Z, K_ν, C]}`).

## `wigner_3j(l1, l2, l3)`, `wigner_D(l, alpha, beta, gamma)`

e3nn-compatible real Wigner-3j symbols (`l ≤ 3`, checked against e3nn 0.4.4 to
1e-15) and Wigner D matrices (Euler `Y(α) X(β) Y(γ)`). `C` is invariant under
`D` on all three indices; the tests use this for the equivariance checks.

## Backends

| `set_backend` | what runs |
|---|---|
| `"pytorch"` | the fused PyTorch formulation on the operands' device |
| `"nki"` | the NKI kernels, one launch per direction per layer; float64 is computed in fp32 with a one-time warning |
| `"auto"` | the fused PyTorch path. With `TRNTENSOR_EQUIVARIANT_NKI=1`, fp32 operands on the `neuron` device use the kernels. float64 on the `neuron` device is computed on the host. |

The kernels are opt-in under `"auto"` until they have been validated on
Trainium (#138). `TRNTENSOR_REQUIRE_NKI=1` raises instead of falling back.

## Design

### The fused PyTorch path

- **conv_tp.** One matmul `y @ cg` builds every per-edge coupling block
  `G[e] ∈ R^{D1×K}` from a dense block-CG matrix (`[16, 4·40]` at layer 1,
  α folded in). Here `D1 = Σ (2 l1 + 1)` (4 at layer 1) and `K = Σ_p (2 l3 + 1)` (40).
  One batched product `x_cat @ G` follows, then one broadcast multiply by
  `w` per *run* of paths. A run is a group of consecutive paths with the same
  `2 l3 + 1` and consecutive output blocks; MACE-MP has 4 runs per layer. The
  multiply writes straight into strided views of the output in e3nn's layout.
  The forward is 7 aten launches at layer 0 and 8 at layer 1, the backward
  about 29. e3nn's codegen is 34 forward and 144 forward + backward at layer 1.
  The backward recomputes the intermediates rather than saving them, and it is
  built from differentiable ops, so double backward (training on forces) works.
- **symmetric_contraction.** The coupling tensors are folded onto the
  *symmetric monomial basis*: products `x_{a1} … x_{aν}` with `a1 ≤ … ≤ aν`
  and a non-zero coupling, 410 of the 968 at layer 0 and 94 at layer 1. Per
  row `(node, channel)`, `out = ((mono @ usym) ⊙ W[z_n, c]) @ S`. That is one
  `[rows, 410] × [410, 202]` matmul at layer 0, where MACE runs
  `[rows, 768] × [768, 816]` plus `bmm`.

### The NKI kernels

**conv_tp**:

- Edges sit on the 128 partitions and channels on the free axis. Each operand
  row is one contiguous DMA per tile of 128 edges.
- `G = y_tile @ cg` is one Tensor Engine matmul. The compact `cg`
  (`[16, Q]`, `Q = Σ_p d1·d3` = 88 at layer 1) stays in SBUF for the whole kernel.
- Each column of `G` is a per-partition scalar. So every term of
  `out_p[:, u, k] = Σ_i (x_i ⊙ w_p) · G[:, (p, i, k)]` is one `[128, U]`
  instruction:
  - the `x_i ⊙ w_p` products run on GpSimd;
  - the first term of each `k` runs on the Scalar Engine;
  - the accumulating terms run on the Vector Engine.
- e3nn's layouts are read and written through strided SBUF views.
- Per 128-edge tile that is 110 instructions forward at layer 1, and about
  30 KiB (forward) or 40 KiB (backward) of the 192 KiB per partition.
- The backward reduces the 128 channels for `∂G` with the Vector Engine's
  fused free-axis reduction. `∂y = ∂G @ cgᵀ` is then a Tensor Engine
  transpose and one matmul.

**symmetric_contraction**:

- Rows `(node, channel)` are on the free axis, 512 per block. The small axes
  are on partitions: features, monomials, coupling columns `(irrep, m, ν, k)`
  and outputs.
- Every contraction is a Tensor Engine matmul accumulating in PSUM:
  - 0/1 selections that form the monomial factors;
  - the folded couplings;
  - the reduction over `(ν, k)`.
- All constants stay in SBUF. PSUM use is 7 of 8 banks.
- Species enter only as a host-side row gather of the folded weight table.

**Launches.** One kernel per direction per layer. Edge counts are padded to
coarse buckets (a multiple of 256; 10,392 and 10,378 edges both run at 11,264),
so the small changes in E between MD steps reuse one compiled kernel. No index
is compiled in.

**Autograd.** On the native TorchNeuron path, `trntensor::conv_tp` and
`trntensor::symcon` are `nki_op` custom ops with `register_autograd`. Their
backward ops are `trntensor::conv_tp_bwd` and `trntensor::symcon_bwd`. On the
XLA path and the CPU simulator, `torch.autograd.Function` wrappers run the
same kernels. The kernel backward is first order only: forces work, double
backward does not.

## Tests

- `tests/test_equivariant.py` runs on CPU:
  - against MACE's captured outputs and gradients (the exemplar's fixture);
  - against the per-path reference;
  - fp64 `gradcheck` and `gradgradcheck`;
  - equivariance under random rotations;
  - spec builders from explicit irreps, e3nn and MACE modules.
- `tests/test_equivariant_native.py` runs on CPU. It covers autograd and op
  registration with emulated kernels.
- `tests/test_equivariant_nki.py` needs `nki`. It runs MLIR `verify()` for
  trn1/trn2 at LNC 1 and 2, and `nki.simulate` against MACE.
- `tests/test_equivariant_neuron.py` runs on device. It checks correctness at
  the 216- and 1728-atom shapes, checks captures, and has the timing harness.
