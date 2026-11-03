"""Density-fitted integrals streamed over the auxiliary index.

The packed three-centre integrals ``B[μν, P]`` (``(npair, naux)``,
:func:`trntensor.pack_tril` order) grow as ``nbasis² · naux``: 33 GB in fp32 at
nbasis 1658 / naux 6064, about 270 GB at nbasis ≈ 3300. One Trn2 logical core
has 24 GiB of HBM and a chip 96 GB, so past a size the integrals cannot stay
resident. Every DF quantity used here is a sum or a concatenation over the
aux index, though:

    J[μν]     = Σ_chunks  B_c (B_cᵀ D)             (sum)
    K[μν]     = Σ_chunks  df_exchange(B_c, C)      (sum)
    B_ia^P    = [ao_to_mo_transform(B_c, C_o, C_v) for each chunk]  (concat)

so :class:`AuxStream` holds the integrals as a list of aux-column chunks
(host tensors, numpy arrays or ``np.load(..., mmap_mode=...)`` memmaps, or
zero-argument loaders that return one), keeps as many as fit a per-device
budget resident, and streams the rest from the host (or disk) on every call.
A background thread uploads chunk ``k + 1`` while the device computes on chunk
``k``.

- **Chunk size.** :func:`aux_chunk_width` keeps every chunk below
  ``max_bytes`` (default 2 GiB), under the 4 GiB transposed-LHS limit of the
  device matmul (aws-neuron-sdk#1457, #97), and a multiple of 128 so the
  ``df_exchange`` kernel tiles it without a short tail; equal widths keep the
  number of compiled shapes at two at most.
- **Several devices.** Chunks are dealt round-robin over ``devices`` (e.g.
  ``neuron:0..3`` for one Trn2 chip, ``neuron:0..63`` for a trn2.48xlarge seen
  by one process); each device accumulates its partial J and K, and the
  partials are summed on the host once per call. The inputs (packed density,
  orbitals) go to each device through the host, because a tensor moved
  between NeuronCores directly stays on its source core. Multi-process /
  multi-host runs can reuse the same split: give each rank its slice of the
  chunk list and all-reduce the partials.
- **Layout on disk.** Store chunks as separate ``(npair, nq)`` C-contiguous
  ``.npy`` files and pass ``lambda: np.load(path, mmap_mode="c")`` loaders:
  each upload is then one contiguous read (served from the page cache when
  host RAM holds it). Column slices of one ``(npair, naux)`` array are strided
  reads.
"""

from __future__ import annotations

import concurrent.futures as _cf
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from .nki import dispatch as _dispatch

# Default upper bound on one chunk's bytes: half the 4 GiB transposed-LHS
# limit (aws-neuron-sdk#1457), so a chunk and its fp32 working copies stay
# well under it.
DEFAULT_CHUNK_BYTES = 1 << 31


def aux_chunk_width(
    npair: int,
    naux: int,
    max_bytes: int = DEFAULT_CHUNK_BYTES,
    itemsize: int = 4,
    multiple: int = 128,
) -> int:
    """Aux columns per chunk: the largest multiple of ``multiple`` whose
    ``(npair, width)`` chunk of ``itemsize``-byte elements is at most
    ``max_bytes``, and at most ``naux``. Raises when even ``multiple``
    columns exceed ``max_bytes``."""
    if npair <= 0 or naux <= 0:
        raise ValueError(f"npair and naux must be positive; got {npair}, {naux}")
    per_col = npair * itemsize
    width = (max_bytes // per_col) // multiple * multiple
    if width == 0:
        if naux * per_col <= max_bytes:
            return naux
        raise ValueError(
            f"a {multiple}-column chunk of {npair} rows is {multiple * per_col} bytes, "
            f"over max_bytes={max_bytes}"
        )
    return min(width, naux)


def split_aux(B, max_bytes: int = DEFAULT_CHUNK_BYTES, multiple: int = 128, dtype=None):
    """Packed ``(npair, naux)`` integrals as a list of contiguous aux chunks.

    ``B`` may be a tensor or a numpy array; each chunk is a contiguous host
    copy of a column block (``dtype`` casts it). The chunk widths come from
    :func:`aux_chunk_width` for the output dtype.
    """
    t = _as_tensor(B)
    dtype = dtype or t.dtype
    npair, naux = t.shape
    width = aux_chunk_width(npair, naux, max_bytes, _itemsize(dtype), multiple)
    return [t[:, q0 : q0 + width].to(dtype).contiguous() for q0 in range(0, naux, width)]


def pack_density(D: torch.Tensor) -> torch.Tensor:
    """Weighted packed density ``w · D[μ, ν]`` (``ν ≤ μ``; ``w`` = 1 on the
    diagonal, 2 off it), the ``(npair,)`` vector :func:`df_coulomb` and
    :meth:`AuxStream.jk` take, computed on ``D``'s device.

    On a NeuronCore the gather compiles as one op with npair indices, which
    at nbasis 1658 (1.4e6 pairs) ran for over 40 min (trnsolver#87): pass a
    host ``D`` there (``D.cpu()``, a few MB) and upload the result.
    """
    n = D.shape[0]
    r = torch.tril_indices(n, n, device=D.device)
    w = torch.where(r[0] == r[1], 1.0, 2.0).to(D.dtype)
    return D.reshape(-1).index_select(0, r[0] * n + r[1]) * w


def unpack_packed(jp: torch.Tensor, nbasis: int) -> torch.Tensor:
    """Symmetric ``(nbasis, nbasis)`` from a packed ``(npair,)`` vector (on
    its device; pass a host vector for large nbasis, see :func:`pack_density`)."""
    from .nki.dispatch import _tril_rows

    idx = _tril_rows(nbasis).reshape(-1).to(jp.device)
    return jp.index_select(0, idx).reshape(nbasis, nbasis)


def _is_memmap(x) -> bool:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return False
    return isinstance(x, np.memmap)


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _as_tensor(x) -> torch.Tensor:
    """A host chunk as a tensor, without a copy where possible."""
    if isinstance(x, torch.Tensor):
        return x
    import numpy as np

    a = np.asarray(x)
    if not a.flags.writeable:
        # torch.from_numpy warns on read-only memory (np.load mmap_mode="r");
        # mmap_mode="c" (copy-on-write) avoids this copy.
        a = np.array(a)
    return torch.from_numpy(a)


@dataclass
class StreamStats:
    """Counters for the last :class:`AuxStream` pass."""

    chunks_resident: int = 0
    chunks_streamed: int = 0
    bytes_streamed: int = 0
    wait_s: float = 0.0  # main thread blocked on a chunk upload
    load_s: float = 0.0  # loader-thread time: host read + cast + upload
    per_device: dict = field(default_factory=dict)


class AuxStream:
    """Packed DF integrals held as aux chunks, resident or streamed to devices.

    Parameters
    ----------
    chunks : sequence
        Aux-column chunks, in aux order: each a ``(npair, nq)`` tensor (host
        or device), numpy array / memmap, or a zero-argument callable
        returning one (loaded when needed, e.g. from disk).
    nbasis : int
        Number of basis functions; ``npair = nbasis(nbasis+1)/2``.
    devices : sequence of devices, optional
        Where to compute; chunks are dealt round-robin. Default: CPU.
    dtype : torch.dtype
        Compute dtype; chunks are cast on the host before upload.
    resident_bytes : float
        Per-device budget for chunks kept on the device between calls. The
        first chunks dealt to a device are kept until the budget is spent;
        the rest are streamed on every call. ``float("inf")`` keeps
        everything (the ``DFJK`` behaviour); ``0`` streams everything.
        A chunk that already lives on its device is always resident.
    prefetch : int
        Uploads in flight ahead of the compute, per device (``0``: load
        synchronously, no overlap).
    """

    def __init__(
        self,
        chunks: Sequence,
        nbasis: int,
        *,
        devices=None,
        dtype: torch.dtype = torch.float32,
        resident_bytes: float = 0.0,
        prefetch: int = 1,
    ):
        if not len(chunks):
            raise ValueError("AuxStream needs at least one chunk")
        self.nbasis = nbasis
        self.npair = nbasis * (nbasis + 1) // 2
        self.devices = [torch.device(d) for d in (devices or ["cpu"])]
        self.device = self.devices[0]
        self.dtype = dtype
        self.prefetch = max(0, int(prefetch))
        self._sources = list(chunks)
        self.widths = []
        for i, c in enumerate(self._sources):
            shape = self._shape_of(c)
            if shape[0] != self.npair:
                raise ValueError(
                    f"chunk {i} has {shape[0]} rows; nbasis={nbasis} needs {self.npair}"
                )
            self.widths.append(shape[1])
        self.offsets = [0]
        for w in self.widths:
            self.offsets.append(self.offsets[-1] + w)
        self.naux = self.offsets[-1]
        self.owner = [self.devices[i % len(self.devices)] for i in range(len(self._sources))]
        self.chunk_bytes = [self.npair * w * _itemsize(dtype) for w in self.widths]
        # Residency: first come, first kept, per device.
        self._resident: dict[int, torch.Tensor] = {}
        used = {d: 0 for d in self.devices}
        for i, src in enumerate(self._sources):
            d = self.owner[i]
            if isinstance(src, torch.Tensor) and src.device == d and src.dtype == dtype:
                self._resident[i] = src
            elif used[d] + self.chunk_bytes[i] <= resident_bytes:
                self._resident[i] = self._load(i)
                used[d] += self.chunk_bytes[i]
        self.resident_bytes_used = used
        self._pool = None
        self.stats = StreamStats()

    # -- chunk access ------------------------------------------------------

    def _shape_of(self, c):
        if callable(c) and not isinstance(c, torch.Tensor):
            # Peek at a lazy chunk once; memmaps make this cheap.
            t = c()
            return tuple(t.shape)
        return tuple(c.shape)

    def _load(self, i: int) -> torch.Tensor:
        """Chunk ``i`` on its device in the compute dtype (host read, cast, upload)."""
        src = self._sources[i]
        if callable(src) and not isinstance(src, torch.Tensor):
            src = src()
        if _is_memmap(src) and self.owner[i].type == "cpu":
            # Host compute: one sequential read into RAM. Left mapped, the
            # kernels' row gathers (df_exchange's tril rows) fault pages in at
            # random, which on a disk-backed store ran > 10x slower than the
            # read itself. A device upload reads the mapping sequentially and
            # needs no copy. On Trn2 an extra copy capped a page-cached 1.6 GB
            # chunk at 2.9 GB/s (one host-memcpy pass on 2 pinned CPUs),
            # against 12-14 GB/s straight from the mapping (trnsci/trnsci#115).
            import numpy as np

            src = np.array(src, copy=True)
        t = _as_tensor(src)
        if t.dtype != self.dtype:
            t = t.to(self.dtype)  # on the host: halves the bytes moved for fp64
        if not t.is_contiguous():
            t = t.contiguous()
        return t.to(self.owner[i])

    def _timed_load(self, i: int):
        t0 = time.perf_counter()
        # No device synchronise here: it would also wait for the compute the
        # main thread has queued, which is what the upload should overlap.
        # A copy from pageable host memory has read its source on return.
        t = self._load(i)
        return t, time.perf_counter() - t0

    def _executor(self):
        if self._pool is None:
            self._pool = _cf.ThreadPoolExecutor(
                max_workers=max(1, len(self.devices)), thread_name_prefix="trntensor-aux"
            )
        return self._pool

    def close(self) -> None:
        """Release resident chunks and the loader threads."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._resident.clear()

    def __del__(self):
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
        except Exception:
            pass

    def chunks(self):
        """Yield ``(i, q0, q1, chunk_on_device)`` for every chunk, in order.

        Resident chunks are yielded as is; streamed ones come from the loader
        threads, ``prefetch`` per device ahead of the consumer, so the upload
        of the next chunk overlaps the work the caller issues on this one.
        Updates :attr:`stats`.
        """
        st = StreamStats(chunks_resident=len(self._resident))
        streamed = [i for i in range(len(self._sources)) if i not in self._resident]
        depth = self.prefetch * len(self.devices)
        futures: dict[int, _cf.Future] = {}
        nxt = 0

        def fill():
            nonlocal nxt
            while depth and nxt < len(streamed) and len(futures) < depth:
                j = streamed[nxt]
                futures[j] = self._executor().submit(self._timed_load, j)
                nxt += 1

        fill()
        try:
            for i in range(len(self._sources)):
                q0, q1 = self.offsets[i], self.offsets[i + 1]
                if i in self._resident:
                    yield i, q0, q1, self._resident[i]
                    continue
                t0 = time.perf_counter()
                if i in futures:
                    chunk, dt = futures.pop(i).result()
                else:
                    chunk, dt = self._timed_load(i)
                    nxt = max(nxt, streamed.index(i) + 1)
                st.wait_s += time.perf_counter() - t0
                st.load_s += dt
                st.chunks_streamed += 1
                st.bytes_streamed += self.chunk_bytes[i]
                fill()
                yield i, q0, q1, chunk
                del chunk
        finally:
            for f in futures.values():
                f.cancel()
            self.stats = st

    # -- distribution helpers ---------------------------------------------

    def scatter(self, x: torch.Tensor) -> dict:
        """``x`` on every device: itself on its own device, through the host elsewhere."""
        h = None
        out = {}
        for d in self.devices:
            if x.device == d:
                out[d] = x
            else:
                if h is None:
                    h = x if x.device.type == "cpu" else x.cpu()
                out[d] = h.to(d)
        return out

    def reduce(self, parts: dict, device=None) -> torch.Tensor:
        """Sum of per-device partials, on ``device`` (default the first device)."""
        device = torch.device(device) if device is not None else self.device
        vals = list(parts.values())
        if len(vals) == 1:
            return vals[0].to(device)
        host = [_dispatch._to_host(v, "cpu") for v in vals]
        return torch.stack(host).sum(0).to(device)

    # -- DF builds ---------------------------------------------------------

    def jk(
        self,
        pw: torch.Tensor | None,
        C_occ: torch.Tensor | None,
        alpha: float = 2.0,
        out_device=None,
    ):
        """Packed J and K in one pass over the chunks (each streamed chunk is
        uploaded once for both).

        ``pw`` is the weighted packed density (:func:`pack_density`), ``C_occ``
        the occupied orbitals; either may be ``None`` to skip that build.
        Returns ``(Jp, K)``: ``Jp`` is ``(npair,)`` (unpack with
        :func:`unpack_packed`), ``K = alpha · Σ X Xᵀ`` (``df_exchange``),
        both on ``out_device`` (default: the first device).
        """
        from .quantum import df_coulomb, df_exchange

        ps = self.scatter(pw.to(self.dtype)) if pw is not None else None
        cs = self.scatter(C_occ.to(self.dtype)) if C_occ is not None else None
        jparts: dict = {}
        kparts: dict = {}
        for i, _q0, _q1, Bc in self.chunks():
            d = self.owner[i]
            if ps is not None:
                jb = df_coulomb(Bc, ps[d])
                jparts[d] = jb if d not in jparts else jparts[d] + jb
            if cs is not None:
                kb = df_exchange(Bc, cs[d], alpha=alpha)
                kparts[d] = kb if d not in kparts else kparts[d] + kb
        Jp = self.reduce(jparts, out_device) if jparts else None
        K = self.reduce(kparts, out_device) if kparts else None
        return Jp, K

    def coulomb(self, pw: torch.Tensor, out_device=None) -> torch.Tensor:
        """Packed ``(npair,)`` J for the weighted packed density ``pw``."""
        return self.jk(pw, None, out_device=out_device)[0]

    def exchange(self, C_occ: torch.Tensor, alpha: float = 1.0, out_device=None) -> torch.Tensor:
        """``K = alpha · Σ_{i,P} X Xᵀ`` (:func:`trntensor.df_exchange`) over all chunks."""
        return self.jk(None, C_occ, alpha=alpha, out_device=out_device)[1]

    def ao_to_mo(
        self,
        C_occ: torch.Tensor,
        C_vir: torch.Tensor,
        out=None,
    ):
        """``B[i, a, P]`` (:func:`trntensor.ao_to_mo_transform`), chunk by chunk,
        written into the host array ``out`` (allocated as a ``(nocc, nvir,
        naux)`` tensor if ``None``; a numpy memmap works for outputs larger
        than host RAM). Returns ``out``.

        Each chunk's slice is copied back as soon as its transform is issued
        on the next device, so at most one result per device is in flight.
        """
        from .quantum import ao_to_mo_transform

        nocc, nvir = C_occ.shape[1], C_vir.shape[1]
        if out is None:
            out = torch.empty(nocc, nvir, self.naux, dtype=self.dtype)
        is_np = not isinstance(out, torch.Tensor)
        co = self.scatter(C_occ.to(self.dtype))
        cv = self.scatter(C_vir.to(self.dtype))
        pending: dict = {}

        def drain(d):
            if d in pending:
                q0, q1, r = pending.pop(d)
                h = _dispatch._to_host(r, "cpu")
                if is_np:
                    out[:, :, q0:q1] = h.numpy()
                else:
                    out[:, :, q0:q1] = h

        for i, q0, q1, Bc in self.chunks():
            d = self.owner[i]
            drain(d)
            pending[d] = (q0, q1, ao_to_mo_transform(Bc, co[d], cv[d]))
        for d in list(pending):
            drain(d)
        return out


# ---------------------------------------------------------------------------
# MP2 over occupied blocks
# ---------------------------------------------------------------------------


def _pair_block_energy(
    BI: torch.Tensor,
    BJ: torch.Tensor,
    eo_I: torch.Tensor,
    eo_J: torch.Tensor,
    ev: torch.Tensor,
    chunk_elems: int = 1 << 24,
) -> torch.Tensor:
    """Σ_{i∈I, j∈J, a, b} T (2T - T_ijba) / Δ with T_ijab = Σ_P BI[i,a,P] BJ[j,b,P]."""
    ni, nvir, _ = BI.shape
    nj = BJ.shape[0]
    evv = ev[:, None] + ev[None, :]
    step = max(1, chunk_elems // max(1, nj * nvir * nvir))
    e = BI.new_zeros(())
    for i0 in range(0, ni, step):
        Bi = BI[i0 : i0 + step]
        T = torch.einsum("iap,jbp->ijab", Bi, BJ)
        denom = (eo_I[i0 : i0 + step, None] + eo_J[None, :])[:, :, None, None] - evv
        e = e + (T * (2 * T - T.transpose(2, 3)) / denom).sum()
    return e


def mp2_energy_blocked(
    B,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
    *,
    devices=None,
    occ_block: int | None = None,
    max_bytes: int = DEFAULT_CHUNK_BYTES,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """DF-MP2 correlation energy from a host ``B[i, a, P]`` too large for one device.

    The same energy as :func:`trntensor.mp2_energy`, summed over blocks of
    occupied orbitals: with ``e(I, J) = Σ_{i∈I, j∈J, ab} T(2T − T_ijba)/Δ``,
    ``E = Σ_I e(I, I) + 2 Σ_{I<J} e(I, J)`` (``e(I, J) = e(J, I)``). Each
    block ``B[I]`` (``occ_block · nvir · naux`` elements, at most
    ``max_bytes``) is uploaded to the device that owns the pair, so device
    memory needs two blocks, not ``B``. Pairs are dealt round-robin over
    ``devices``; the per-device partial energies are summed in float64 on
    the host. ``B`` may be a tensor or a numpy array / memmap.

    When the whole ``B`` fits one device, :func:`trntensor.mp2_energy` on the
    device-resident ``B`` is faster (its row kernel uses the pair symmetry
    inside one launch).
    """
    Bt = _as_tensor(B)
    nocc, nvir, naux = Bt.shape
    dtype = dtype or Bt.dtype
    devs = [torch.device(d) for d in (devices or ["cpu"])]
    if occ_block is None:
        occ_block = max(1, min(nocc, max_bytes // max(1, nvir * naux * _itemsize(dtype))))
    blocks = [(i0, min(nocc, i0 + occ_block)) for i0 in range(0, nocc, occ_block)]
    eo = eps_occ.to(dtype)
    ev = eps_vir.to(dtype)
    evd = {d: ev.to(d) for d in devs}
    eod = {d: eo.to(d) for d in devs}
    cache: dict = {}

    def block_on(b, d):
        key = (b, d)
        if key not in cache:
            i0, i1 = blocks[b]
            cache[key] = Bt[i0:i1].to(dtype).contiguous().to(d)
        return cache[key]

    parts = []
    k = 0
    for bi in range(len(blocks)):
        # Keep only the I block (and the J blocks of this row) per device.
        for key in [key for key in cache if key[0] < bi]:
            del cache[key]
        for bj in range(bi, len(blocks)):
            d = devs[k % len(devs)]
            k += 1
            (i0, i1), (j0, j1) = blocks[bi], blocks[bj]
            BI = block_on(bi, d)
            BJ = BI if bj == bi else block_on(bj, d)
            e = _pair_block_energy(BI, BJ, eod[d][i0:i1], eod[d][j0:j1], evd[d])
            parts.append(e if bj == bi else 2 * e)
            if bj != bi:
                cache.pop((bj, d), None)
    # Read back only after every pair is issued, so the devices run concurrently.
    total = torch.zeros((), dtype=torch.float64)
    for e in parts:
        total += e.double().cpu()
    return total
