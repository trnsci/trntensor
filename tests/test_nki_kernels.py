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
def nki_backend():
    import trntensor
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
        A = torch.randn(100, 70)   # M, K both not tile multiples
        B = torch.randn(70, 200)   # K=70 → pads to 128; N=200 ≤ TILE_N, no N-pad
        out = trntensor.einsum("ij,jk->ik", A, B)
        ref = A @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_transA(self, nki_backend):
        """``ji,jk->ik`` — A transposed by the dispatch before the kernel."""
        import trntensor

        torch.manual_seed(2)
        A = torch.randn(128, 64)   # shape (K, M) from the einsum perspective
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
        B = torch.randn(128, 1024)   # N=1024 > TILE_N=512
        out = trntensor.einsum("ij,jk->ik", A, B)
        ref = A @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
