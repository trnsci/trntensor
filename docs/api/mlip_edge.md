# MLIP edge and node ops

The MACE-MP-0b2 medium energy + forces step on Trn2 (216 atoms, 10,392 edges)
took 118 ms with the equivariant contractions and the graph gather / segment
sum on trntensor and trnsparse kernels. About 92 ms of that was MACE's
remaining 1,503 aten ops, each launched separately
([trnsci/trnsci#178](https://github.com/trnsci/trnsci/issues/178)). These
ops replace the largest groups. Each one is one autograd node with an
analytic backward, so forces (`-dE/dpositions`) still come from autograd.

| op | replaces (MACE / e3nn) | gradient |
|---|---|---|
| `spherical_harmonics(vectors, lmax=3)` | `o3.SphericalHarmonics(..., normalize=True, normalization="component")` | ∂vectors |
| `edge_chain(lengths, spec, r0)` | `RadialEmbeddingBlock` (Agnesi, Bessel, polynomial cutoff), every interaction's `conv_tp_weights` and `tanh(density_fn(·)²)` | ∂lengths |
| `equivariant_linear(x, spec, species=None)` | `o3.Linear`; the one-hot `FullyConnectedTensorProduct` (`skip_tp`) | ∂x |

## `spherical_harmonics(vectors, lmax=3, normalize=True, normalization="component", eps=1e-12)`

This uses e3nn 0.4's real basis, component order and normalisation for
`l ≤ 3`. `normalize=True` divides by `max(|v|, eps)`, as
`torch.nn.functional.normalize` does. Every component is a polynomial of
degree `≤ l` in the unit vector, so the forward is one matmul of the
degree-1…3 monomials against a fixed `[19, 16]` coefficient matrix
(`sh_coefficients`). The backward is one matmul against the derivative
tables plus the projection of the normalisation.

## `edge_chain_spec(...)` and `edge_chain(lengths, spec, r0=None)`

`edge_chain_spec(module=model)` reads a MACE model. It takes the radial
embedding and, per interaction, `conv_tp_weights` (head `tp_w{i}`) and
`density_fn` with `post="tanh_sq"` (head `density{i}`). Explicit
construction takes `r_max`, `num_basis`, `cutoff_p`, `agnesi=(a, q, p)`,
`apply_cutoff` and `EdgeMLP` heads. Each head holds effective matrices with
e3nn's `1/√fan_in` and `normalize2mom` constants folded in.

`edge_chain` returns one `[E, d_h]` tensor per head. The Agnesi species
constant `r0 = (R[z_s] + R[z_r]) / 2` depends only on the neighbour list.
MACE rebuilds it from integer gathers on every call; here it is an input,
`spec.agnesi_r0(z_sender, z_receiver)`, built once per neighbour list. An
edge at or past `r_max` contributes exactly zero, and so does its gradient.
Pad edge buckets with such edges.

## `equivariant_linear_spec(module=None, *, elements=None, ...)` and `equivariant_linear(x, spec, species=None)`

The spec probes the module's dense map once in float64. It splits the map
per irrep type `(l, p)` into `K_l = W_l ⊗ I_{2l+1}` and checks the split
exactly. The op is then one matmul per irrep type and a concatenation, where
e3nn runs 62 aten ops for one 128x0e+…+128x3o linear. For the one-hot skip
tensor product, `elements` lists the species present; each node keeps its
own species' block. `mode="dense"` uses one matmul against the full matrix.

## Backends

- **PyTorch path** (default everywhere). This is a short chain of aten ops
  per direction. Against MACE-MP-0b2 at the six exemplar captures, outputs
  match MACE fp32 to ≤ 4.6e-7 relative. Input gradients for the real
  force-step cotangents are at most as far from fp64 as MACE's own fp32
  gradients (`examples/mlip/check_fused.py`).
- **Compiled regions**: `set_region_backend("neuron")` (or
  `TRNTENSOR_REGION_BACKEND`). The forward and the backward bodies each run
  as one `torch.compile(dynamic=False)` graph per call site. Pad edge counts
  to a bucket to avoid a recompile per geometry.
- **NKI kernels** for `spherical_harmonics` and `edge_chain`
  (`trntensor/nki/_edge_kernels.py`), one launch per direction. Edges are
  laid out partition-major, so each elementwise step is one instruction over
  a `[128, J]` plane of edges. The radial MLPs run on the Tensor Engine with
  the weights resident in SBUF and PSUM accumulation. `sin` runs as
  `sin(π − w x)` to stay in the Scalar Engine's accurate `[−π, π]` range; the
  host routes only specs where `max(w)·x ≤ 2π` (Agnesi transform). The
  kernels pass MLIR verification for trn1 and trn2 (LNC 1 and 2) and match
  the fp64 PyTorch path in `nki.simulate`. They are not yet validated on a
  device, so `"auto"` does not route to them; use `set_backend("nki")` or
  `TRNTENSOR_EDGE_NKI=1`.

## Tests

- `tests/test_edge.py` (CPU): the MACE fixture `fixtures/mace_mp0b2_edge.pt`,
  e3nn and MACE modules when installed, fp64 `gradcheck` / `gradgradcheck`,
  equivariance, padding and compiled regions.
- `tests/test_edge_native.py` (CPU): registered ops, autograd, routing and
  weight packing, with the kernels emulated by their contracts.
- `tests/test_edge_nki.py` (`nki`): MLIR verification and `nki.simulate`.
- `tests/test_edge_neuron.py` (device): correctness at 216 and 1728 atoms,
  and a timing harness (`TRNTENSOR_EDGE_BENCH_JSON=…`).
