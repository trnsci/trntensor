"""On-hardware tests for NKI kernels.

Run via ``scripts/run_neuron_tests.sh trn1`` against a provisioned
Trainium instance. All tests are gated on ``@pytest.mark.neuron`` so
they're skipped on CPU-only runners.
"""

import pytest
import torch

pytestmark = pytest.mark.neuron


ATOL = 1e-3
RTOL = 1e-4


@pytest.fixture
def nki_backend(monkeypatch):
    """Force the NKI path for the duration of the test, ignoring the
    FLOP threshold that would otherwise send small shapes to ``torch``.
    The kernels themselves need to be exercised even when the wrapper
    would decline them for performance reasons.
    """
    import trntensor
    from trntensor.nki import dispatch

    monkeypatch.setattr(dispatch, "_MIN_NKI_FLOPS", 0)
    prev = trntensor.get_backend()
    trntensor.set_backend("nki")
    yield
    trntensor.set_backend(prev)


class TestNkiMatmul:
    def test_aligned_shapes(self, nki_backend):
        """M, K, N all multiples of their tile dims, N ≤ TILE_N."""
        import trntensor

        torch.manual_seed(0)
        A = torch.randn(128, 128)
        B = torch.randn(128, 256)
        out = trntensor.einsum("ij,jk->ik", A, B)
        ref = A @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_unpadded_K_and_N(self, nki_backend):
        """Neither K nor N is a tile multiple — exercises the padding path."""
        import trntensor

        torch.manual_seed(1)
        A = torch.randn(100, 70)  # M, K both not tile multiples
        B = torch.randn(70, 200)  # K=70 → pads to 128; N=200 ≤ TILE_N, no N-pad
        out = trntensor.einsum("ij,jk->ik", A, B)
        ref = A @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_transA(self, nki_backend):
        """``ji,jk->ik`` — A transposed by the dispatch before the kernel."""
        import trntensor

        torch.manual_seed(2)
        A = torch.randn(128, 64)  # shape (K, M) from the einsum perspective
        B = torch.randn(128, 256)
        out = trntensor.einsum("ji,jk->ik", A, B)
        ref = A.T @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_both_transposed(self, nki_backend):
        """``ji,kj->ik`` — both operands transposed."""
        import trntensor

        torch.manual_seed(3)
        A = torch.randn(128, 64)
        B = torch.randn(256, 128)
        out = trntensor.einsum("ji,kj->ik", A, B)
        ref = A.T @ B.T
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_large_N_requires_n_tiling(self, nki_backend):
        """N > TILE_N triggers the multi-N-tile code path."""
        import trntensor

        torch.manual_seed(4)
        A = torch.randn(128, 128)
        B = torch.randn(128, 1024)  # N=1024 > TILE_N=512
        out = trntensor.einsum("ij,jk->ik", A, B)
        ref = A @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_df_mp2_pair_pattern(self, nki_backend):
        """``ap,bp->ab`` — DF-MP2 pair contraction. Verifies #12 is covered
        by the existing 2-index matmul kernel via ``plan.transB=True``.
        """
        import trntensor

        torch.manual_seed(5)
        nvir, naux = 19, 72
        Bi = torch.randn(nvir, naux)
        Bj = torch.randn(nvir, naux)
        out = trntensor.einsum("ap,bp->ab", Bi, Bj)
        ref = Bi @ Bj.T
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


class TestNkiBatchedMatmul:
    def test_aligned_batch(self, nki_backend):
        """Batch dim, M, K, N all aligned; exercises the base code path."""
        import trntensor

        torch.manual_seed(10)
        A = torch.randn(4, 128, 128)
        B = torch.randn(4, 128, 256)
        out = trntensor.einsum("bij,bjk->bik", A, B)
        ref = torch.bmm(A, B)
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_unpadded_inner(self, nki_backend):
        """Irregular M, K, N within each batch slice; tests padding path."""
        import trntensor

        torch.manual_seed(11)
        A = torch.randn(3, 100, 70)
        B = torch.randn(3, 70, 200)
        out = trntensor.einsum("bij,bjk->bik", A, B)
        ref = torch.bmm(A, B)
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_large_N(self, nki_backend):
        """Batched case with N > TILE_N; exercises N-tiling."""
        import trntensor

        torch.manual_seed(12)
        A = torch.randn(2, 128, 128)
        B = torch.randn(2, 128, 1024)
        out = trntensor.einsum("bij,bjk->bik", A, B)
        ref = torch.bmm(A, B)
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


class TestMp2Energy:
    """End-to-end DF-MP2 correlation energy via the fully fused kernel.

    Validates that the NKI path (contract + elementwise + reduce in one
    @nki.jit program) matches the CPU reference within tolerance.
    """

    def test_demo_shape(self, nki_backend):
        """5×19×72 — the shape used by examples/df_mp2_einsum.py."""
        import trntensor
        from trntensor.quantum import _cpu_mp2_energy

        torch.manual_seed(42)
        nocc, nvir, naux = 5, 19, 72
        B = torch.randn(nocc, nvir, naux) * 0.1
        eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
        eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1

        e_nki = trntensor.mp2_energy(B, eps_occ, eps_vir)
        e_ref = _cpu_mp2_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(e_nki, e_ref, atol=1e-3, rtol=1e-4)

    def test_medium_shape(self, nki_backend):
        """8×32×128 — larger but still within single-tile constraints."""
        import trntensor
        from trntensor.quantum import _cpu_mp2_energy

        torch.manual_seed(43)
        nocc, nvir, naux = 8, 32, 128
        B = torch.randn(nocc, nvir, naux) * 0.1
        eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
        eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1

        e_nki = trntensor.mp2_energy(B, eps_occ, eps_vir)
        e_ref = _cpu_mp2_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(e_nki, e_ref, atol=1e-3, rtol=1e-4)

    def test_oversize_raises(self, nki_backend):
        """nvir > 128 or naux > 128 should raise until K/M tiling lands."""
        import pytest

        import trntensor

        B = torch.randn(2, 200, 50)
        with pytest.raises(NotImplementedError, match="nvir.*≤.*128"):
            trntensor.mp2_energy(B, torch.randn(2), torch.randn(200))
