# Streaming DF integrals over the aux index

`trntensor.stream` holds the packed three-centre integrals `B[μν, P]`
(`(npair, naux)`, `pack_tril` order) as a list of aux-column chunks and runs
the DF builds chunk by chunk. The integrals grow as `nbasis² · naux`: 33 GB in
fp32 at nbasis 1658 / naux 6064 and about 270 GB at nbasis ≈ 3300, against
24 GiB per Trn2 logical core and 96 GB per chip. Every DF quantity used by
an SCF or DF-MP2 is a sum or a concatenation over `P`, so the problem size is
bounded by host memory (or disk), not device memory.

| quantity | over chunks `B_c` |
|---|---|
| `J = B (Bᵀ D)` | sum of `df_coulomb(B_c, D)` |
| `K = α Σ X Xᵀ` | sum of `df_exchange(B_c, C_occ, α)` |
| `B_ia^P` | concatenation of `ao_to_mo_transform(B_c, C_occ, C_vir)` |

## `trntensor.AuxStream(chunks, nbasis, *, devices=None, dtype=float32, resident_bytes=0, prefetch=1)`

- `chunks`: `(npair, nq)` tensors, numpy arrays / memmaps, or zero-argument
  loaders returning one (e.g. `lambda: np.load(path, mmap_mode="c")`).
- `devices`: chunks are dealt round-robin (`neuron:0..3` for one Trn2 chip).
  Each device accumulates partial J and K; partials are summed on the host
  once per call. Inputs reach each core through the host.
- `resident_bytes`: per-device budget for chunks kept on the device between
  calls (first come, first kept). The rest are uploaded on every call by a
  loader thread, `prefetch` per device ahead of the compute, so the upload of
  chunk `k + 1` overlaps the kernels on chunk `k`.
- `stats` (after each pass): resident / streamed chunk counts, bytes
  streamed, main-thread wait and loader time.

Methods: `jk(pw, C_occ, alpha=2.0)` → `(Jp, K)` in one pass (each streamed
chunk uploaded once for both), `coulomb(pw)`, `exchange(C_occ, alpha)`,
`ao_to_mo(C_occ, C_vir, out=None)` (writes into a host tensor or a numpy
memmap).

## Helpers

- `aux_chunk_width(npair, naux, max_bytes=2 GiB, itemsize=4, multiple=128)`:
  the widest 128-multiple chunk under `max_bytes`. The default keeps every
  chunk under the 4 GiB transposed-LHS limit of the device matmul
  (aws-neuron-sdk#1457, #97), and equal widths keep compiled shapes to two.
- `split_aux(B, max_bytes, multiple, dtype)`: contiguous chunk copies of an
  in-memory `B`.
- `pack_density(D)` / `unpack_packed(jp, nbasis)`: the weighted packed
  density `df_coulomb` takes and the inverse for J. On a NeuronCore these
  gathers compile as single ops with `npair` / `nbasis²` indices (over 40 min
  each at nbasis 1658, trnsolver#87): call them on host tensors there.
- `mp2_energy_blocked(B, eps_occ, eps_vir, *, devices, occ_block, max_bytes)`:
  DF-MP2 from a host `B[i, a, P]` larger than one device, as
  `Σ_I e(I, I) + 2 Σ_{I<J} e(I, J)` over occupied blocks; pairs dealt
  round-robin over devices, partial energies summed in float64.

## On-disk layout

Store one C-contiguous `(npair, nq)` `.npy` file per chunk and pass memmap
loaders: each upload is one contiguous read, from the page cache when host
RAM holds the integrals. Column slices of one `(npair, naux)` file are
strided reads.

## Status

CPU-validated (`tests/test_stream.py`). The overlap of uploads with compute
and the multi-core scaling are to be measured on Trn2.
