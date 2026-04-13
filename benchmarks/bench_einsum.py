"""Benchmarks for einsum dispatch and decompositions.

Run with:
    pytest benchmarks/ --benchmark-only

On-hardware (via SSM orchestration):
    scripts/run_benchmarks.sh trn1
"""

import pytest
import torch

import trntensor


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


class TestEinsumBench:

    def test_matmul_512(self, benchmark):
        A = torch.randn(512, 512)
        B = torch.randn(512, 512)
        benchmark(trntensor.einsum, "ij,jk->ik", A, B)

    def test_bmm_batched(self, benchmark):
        A = torch.randn(16, 256, 256)
        B = torch.randn(16, 256, 256)
        benchmark(trntensor.einsum, "bij,bjk->bik", A, B)

    def test_df_mp2_pair(self, benchmark):
        """DF-MP2 pair contraction: T_ab = Σ_P B_ia^P B_jb^P."""
        nvir, naux = 48, 128
        Bi = torch.randn(nvir, naux)
        Bj = torch.randn(nvir, naux)
        benchmark(trntensor.einsum, "ap,bp->ab", Bi, Bj)

    def test_4index_transform(self, benchmark):
        """AO→MO half-transform."""
        nbasis, nocc, naux = 32, 8, 64
        C = torch.randn(nbasis, nocc)
        eri = torch.randn(nbasis, nbasis, naux)
        benchmark(trntensor.einsum, "mi,mnP->inP", C, eri)


class TestDecomposeBench:

    def test_cp_rank8(self, benchmark):
        T = torch.randn(16, 16, 16)
        benchmark(trntensor.cp_decompose, T, rank=8, max_iter=20)

    def test_tucker_low_rank(self, benchmark):
        T = torch.randn(16, 16, 16)
        benchmark(trntensor.tucker_decompose, T, ranks=(4, 4, 4))
