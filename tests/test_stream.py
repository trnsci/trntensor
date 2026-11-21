"""CPU tests for trntensor.stream: aux-chunked DF J/K, AO->MO and blocked MP2."""

import numpy as np
import pytest
import torch

import trntensor
from trntensor.quantum import _cpu_mp2_energy


@pytest.fixture(autouse=True)
def _pytorch_backend():
    old = trntensor.get_backend()
    trntensor.set_backend("pytorch")
    yield
    trntensor.set_backend(old)


def _system(n=12, naux=300, nocc=4, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    B3 = torch.randn(n, n, naux, generator=g, dtype=dtype)
    B3 = 0.5 * (B3 + B3.transpose(0, 1))
    C = torch.randn(n, n, generator=g, dtype=dtype)
    return B3, trntensor.pack_tril(B3), C[:, :nocc].contiguous(), C[:, nocc:].contiguous()


def _dense_jk(B3, C_occ):
    D = 2 * C_occ @ C_occ.T
    J = torch.einsum("mnP,lsP,ls->mn", B3, B3, D)
    X = torch.einsum("lvP,li->ivP", B3, C_occ)
    K = 2 * torch.einsum("imP,inP->mn", X, X)
    return D, J, K


def test_aux_chunk_width():
    assert trntensor.aux_chunk_width(1000, 5000, max_bytes=4000 * 256, itemsize=4) == 256
    assert trntensor.aux_chunk_width(1000, 100, max_bytes=1 << 30) == 100
    # Below one 128-column multiple: the whole (small) aux range if it fits.
    assert trntensor.aux_chunk_width(10, 50, max_bytes=10 * 50 * 4) == 50
    with pytest.raises(ValueError):
        trntensor.aux_chunk_width(10**6, 6000, max_bytes=1000)
    # n_ao 1658 / naux 6064 under the default 2 GiB: every chunk < 4 GiB.
    w = trntensor.aux_chunk_width(1658 * 1659 // 2, 6064)
    assert w % 128 == 0 and 1658 * 1659 // 2 * w * 4 <= 1 << 31


def test_split_aux_roundtrip():
    _, Bp, _, _ = _system()
    chunks = trntensor.split_aux(Bp, max_bytes=Bp.shape[0] * 128 * 8)
    assert [c.shape[1] for c in chunks] == [128, 128, 44]
    assert all(c.is_contiguous() for c in chunks)
    torch.testing.assert_close(torch.cat(chunks, 1), Bp, rtol=0, atol=0)


def test_pack_unpack_density():
    _, _, C, _ = _system()
    D = C @ C.T
    pw = trntensor.pack_density(D)
    n = D.shape[0]
    r = torch.tril_indices(n, n)
    w = torch.where(r[0] == r[1], 1.0, 2.0).double()
    torch.testing.assert_close(pw, D[r[0], r[1]] * w)
    torch.testing.assert_close(trntensor.unpack_packed(D[r[0], r[1]], n), D)


@pytest.mark.parametrize("prefetch", [0, 1, 2])
@pytest.mark.parametrize("resident", [0.0, 1e12, "half"])
def test_stream_jk_matches_dense(prefetch, resident):
    B3, Bp, C, _ = _system()
    D, J, K = _dense_jk(B3, C)
    chunks = trntensor.split_aux(Bp, max_bytes=Bp.shape[0] * 128 * 8)
    if resident == "half":
        resident = sum(c.numel() * 8 for c in chunks[:2]) / 1  # two of three chunks
    # Loaders, so nothing counts as already on the (CPU) device.
    loaders = [lambda c=c: c for c in chunks]
    st = trntensor.AuxStream(
        loaders, 12, dtype=torch.float64, resident_bytes=resident, prefetch=prefetch
    )
    Jp, Kc = st.jk(trntensor.pack_density(D), C, alpha=2.0)
    torch.testing.assert_close(trntensor.unpack_packed(Jp, 12), J)
    torch.testing.assert_close(Kc, K)
    assert st.stats.chunks_resident + st.stats.chunks_streamed == 3
    if resident == "half" or resident > 1:
        assert st.stats.chunks_resident == (2 if resident != 1e12 else 3)
    if resident == 0.0:
        assert st.stats.chunks_streamed == 3 and st.stats.chunks_resident == 0
        assert st.stats.bytes_streamed == Bp.numel() * 8
    st.close()


def test_stream_multi_device_round_robin():
    # Several "devices" (all the CPU here): chunks dealt round-robin, partials
    # summed on the host.
    B3, Bp, C, _ = _system(naux=520)
    D, J, K = _dense_jk(B3, C)
    chunks = trntensor.split_aux(Bp, max_bytes=Bp.shape[0] * 128 * 8)
    st = trntensor.AuxStream(chunks, 12, devices=["cpu"] * 3, dtype=torch.float64)
    assert len(st.devices) == 3 and len(chunks) == 5
    Jp, Kc = st.jk(trntensor.pack_density(D), C)
    torch.testing.assert_close(trntensor.unpack_packed(Jp, 12), J)
    torch.testing.assert_close(Kc, K)
    torch.testing.assert_close(st.exchange(C, alpha=2.0), K)
    torch.testing.assert_close(st.coulomb(trntensor.pack_density(D)), Jp)


def test_stream_from_disk_memmaps(tmp_path):
    # Lazy loaders over .npy memmaps: the on-disk chunk-major layout.
    B3, Bp, C, _ = _system()
    D, J, K = _dense_jk(B3, C)
    loaders = []
    for i, c in enumerate(trntensor.split_aux(Bp, max_bytes=Bp.shape[0] * 128 * 8)):
        path = tmp_path / f"B_{i:05d}.npy"
        np.save(path, c.float().numpy())
        loaders.append(lambda p=path: np.load(p, mmap_mode="c"))
    loaders[-1] = lambda p=tmp_path / "B_00002.npy": np.load(p, mmap_mode="r")  # read-only too
    st = trntensor.AuxStream(loaders, 12, dtype=torch.float64)
    assert st.naux == 300 and st.widths == [128, 128, 44]
    Jp, Kc = st.jk(trntensor.pack_density(D), C)
    torch.testing.assert_close(trntensor.unpack_packed(Jp, 12), J, rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(Kc, K, rtol=1e-5, atol=1e-3)


def test_memmap_chunk_copied_for_host_not_for_device(tmp_path, monkeypatch):
    """A memory-mapped chunk is read into RAM for host compute (random row
    gathers), but goes to a device straight from the mapping (no extra copy)."""
    import trntensor.stream as ts

    path = tmp_path / "B_00000.npy"
    np.save(path, np.arange(2 * 128 * 3, dtype=np.float32).reshape(3, 256))  # nbasis 2: 3 rows
    copies = []
    real_array = np.array

    def spy(a, *args, **kw):
        if isinstance(a, np.memmap):
            copies.append(1)
        return real_array(a, *args, **kw)

    monkeypatch.setattr(np, "array", spy)
    host = ts.AuxStream([lambda: np.load(path, mmap_mode="c")], 2, resident_bytes=0.0)
    copies.clear()
    host._load(0)
    assert copies == [1]
    dev = ts.AuxStream([lambda: np.load(path, mmap_mode="c")], 2, devices=["meta"])
    copies.clear()
    t = dev._load(0)
    assert copies == [] and t.device.type == "meta" and tuple(t.shape) == (3, 256)


def test_stream_dtype_cast():
    B3, Bp, C, _ = _system()
    D, J, K = _dense_jk(B3, C)
    st = trntensor.AuxStream(trntensor.split_aux(Bp), 12, dtype=torch.float32)
    Jp, Kc = st.jk(trntensor.pack_density(D), C)
    assert Kc.dtype == torch.float32
    torch.testing.assert_close(Kc.double(), K, rtol=1e-4, atol=1e-3)


def test_stream_rejects_bad_chunks():
    with pytest.raises(ValueError):
        trntensor.AuxStream([], 4)
    with pytest.raises(ValueError):
        trntensor.AuxStream([torch.zeros(9, 3)], 4)  # npair(4) = 10


@pytest.mark.parametrize("out_kind", ["tensor", "memmap"])
def test_stream_ao_to_mo(tmp_path, out_kind):
    B3, Bp, Co, Cv = _system(naux=260)
    ref = torch.einsum("mi,na,mnP->iaP", Co, Cv, B3)
    st = trntensor.AuxStream(
        trntensor.split_aux(Bp, max_bytes=Bp.shape[0] * 128 * 8),
        12,
        devices=["cpu", "cpu"],
        dtype=torch.float64,
    )
    out = None
    if out_kind == "memmap":
        out = np.lib.format.open_memmap(
            tmp_path / "bia.npy", mode="w+", dtype=np.float64, shape=tuple(ref.shape)
        )
    got = st.ao_to_mo(Co, Cv, out=out)
    got = torch.from_numpy(np.asarray(got)) if out_kind == "memmap" else got
    torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("occ_block", [1, 2, 3, 7, None])
def test_mp2_energy_blocked(occ_block):
    g = torch.Generator().manual_seed(3)
    nocc, nvir, naux = 7, 11, 30
    B = 0.1 * torch.randn(nocc, nvir, naux, generator=g, dtype=torch.float64)
    eo = -torch.sort(torch.rand(nocc, generator=g, dtype=torch.float64))[0] - 0.5
    ev = torch.sort(torch.rand(nvir, generator=g, dtype=torch.float64))[0] + 0.1
    ref = _cpu_mp2_energy(B, eo, ev)
    got = trntensor.mp2_energy_blocked(B, eo, ev, occ_block=occ_block, devices=["cpu", "cpu"])
    assert got.dtype == torch.float64
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-14)
    # numpy input, byte-bounded blocks.
    got_np = trntensor.mp2_energy_blocked(B.numpy(), eo, ev, max_bytes=2 * nvir * naux * 8)
    torch.testing.assert_close(got_np, ref, rtol=1e-12, atol=1e-14)


def test_ao_to_mo_then_blocked_mp2_matches_dense_pipeline():
    B3, Bp, Co, Cv = _system(n=10, naux=140, nocc=3)
    g = torch.Generator().manual_seed(5)
    eo = -torch.sort(torch.rand(3, generator=g, dtype=torch.float64))[0] - 1.0
    ev = torch.sort(torch.rand(7, generator=g, dtype=torch.float64))[0] + 0.5
    ref = trntensor.mp2_energy(trntensor.ao_to_mo_transform(B3, Co, Cv), eo, ev)
    st = trntensor.AuxStream(
        trntensor.split_aux(Bp, max_bytes=Bp.shape[0] * 128 * 8), 10, dtype=torch.float64
    )
    Bia = st.ao_to_mo(Co, Cv)
    torch.testing.assert_close(trntensor.mp2_energy_blocked(Bia, eo, ev, occ_block=2), ref.double())
