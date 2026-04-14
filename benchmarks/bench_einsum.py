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

    def test_matmul_1024(self, benchmark):
        A = torch.randn(1024, 1024)
        B = torch.randn(1024, 1024)
        benchmark(trntensor.einsum, "ij,jk->ik", A, B)

    def test_matmul_2048(self, benchmark):
        A = torch.randn(2048, 2048)
        B = torch.randn(2048, 2048)
        benchmark(trntensor.einsum, "ij,jk->ik", A, B)

    def test_bmm_batched(self, benchmark):
        A = torch.randn(16, 256, 256)
        B = torch.randn(16, 256, 256)
        benchmark(trntensor.einsum, "bij,bjk->bik", A, B)

    def test_bmm_large(self, benchmark):
        A = torch.randn(32, 1024, 1024)
        B = torch.randn(32, 1024, 1024)
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


class TestMp2Bench:
    """Compare the fused mp2_energy kernel against the Python-loop form."""

    @pytest.fixture
    def mp2_inputs_small(self):
        # Demo size — realistic for tiny molecules.
        nocc, nvir, naux = 5, 19, 72
        B = torch.randn(nocc, nvir, naux) * 0.1
        eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
        eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1
        return B, eps_occ, eps_vir

    @pytest.fixture
    def mp2_inputs_full(self):
        # Saturate the single-tile kernel: nvir=naux=128 at nocc=16 yields
        # 16² = 256 (i,j) pairs, each a 128×128 nc_matmul.
        nocc, nvir, naux = 16, 128, 128
        B = torch.randn(nocc, nvir, naux) * 0.1
        eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
        eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1
        return B, eps_occ, eps_vir

    def test_fused_small(self, benchmark, mp2_inputs_small):
        B, eps_occ, eps_vir = mp2_inputs_small
        benchmark(trntensor.mp2_energy, B, eps_occ, eps_vir)

    def test_loop_small(self, benchmark, mp2_inputs_small):
        B, eps_occ, eps_vir = mp2_inputs_small
        self._run_loop(benchmark, B, eps_occ, eps_vir)

    def test_fused_full(self, benchmark, mp2_inputs_full):
        B, eps_occ, eps_vir = mp2_inputs_full
        benchmark(trntensor.mp2_energy, B, eps_occ, eps_vir)

    def test_loop_full(self, benchmark, mp2_inputs_full):
        B, eps_occ, eps_vir = mp2_inputs_full
        self._run_loop(benchmark, B, eps_occ, eps_vir)

    @staticmethod
    def _run_loop(benchmark, B, eps_occ, eps_vir):
        # The old "trntensor.einsum inside a Python loop" form — what
        # the fused kernel replaces.
        def loop():
            nocc = B.shape[0]
            e = B.new_zeros(())
            for i in range(nocc):
                for j in range(nocc):
                    T = trntensor.einsum("ap,bp->ab", B[i], B[j])
                    denom = eps_occ[i] + eps_occ[j] - eps_vir.unsqueeze(1) - eps_vir.unsqueeze(0)
                    e = e + (T * (2 * T - T.T) / denom).sum()
            return e

        benchmark(loop)


class TestAoToMoTransformBench:
    """Fused ao_to_mo_transform vs. the two-step decomposed einsum form."""

    @pytest.fixture
    def transform_inputs(self):
        nbasis, nocc, nvir, naux = 32, 5, 10, 16
        eri = torch.randn(nbasis, nbasis, naux) * 0.1
        C_occ = torch.randn(nbasis, nocc)
        C_vir = torch.randn(nbasis, nvir)
        return eri, C_occ, C_vir

    def test_fused(self, benchmark, transform_inputs):
        eri, C_occ, C_vir = transform_inputs
        benchmark(trntensor.ao_to_mo_transform, eri, C_occ, C_vir)

    def test_two_step(self, benchmark, transform_inputs):
        eri, C_occ, C_vir = transform_inputs

        def two_step():
            intermediate = torch.einsum("mi,mnP->inP", C_occ, eri)
            return torch.einsum("na,inP->iaP", C_vir, intermediate)

        benchmark(two_step)


class TestResidencyBench:
    """Full DF-MP2 pipeline with and without operand residency."""

    @pytest.fixture
    def pipeline_inputs(self):
        nbasis, nocc, nvir, naux = 32, 5, 10, 16
        eri = torch.randn(nbasis, nbasis, naux) * 0.1
        C_occ = torch.randn(nbasis, nocc)
        C_vir = torch.randn(nbasis, nvir)
        eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
        eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1
        return eri, C_occ, C_vir, eps_occ, eps_vir

    def test_pipeline_cpu(self, benchmark, pipeline_inputs):
        eri, C_occ, C_vir, eps_occ, eps_vir = pipeline_inputs

        def pipeline():
            B = trntensor.ao_to_mo_transform(eri, C_occ, C_vir)
            return trntensor.mp2_energy(B, eps_occ, eps_vir)

        benchmark(pipeline)

    def test_pipeline_xla_resident(self, benchmark, pipeline_inputs):
        eri, C_occ, C_vir, eps_occ, eps_vir = pipeline_inputs
        # Pre-pin — benchmark only the dispatch loop, not the one-shot
        # transfer. This is the apples-to-apples comparison against the
        # CPU-path variant which pays transfer per-dispatch.
        eri_x = trntensor.to_xla(eri)
        C_occ_x = trntensor.to_xla(C_occ)
        C_vir_x = trntensor.to_xla(C_vir)
        eps_occ_x = trntensor.to_xla(eps_occ)
        eps_vir_x = trntensor.to_xla(eps_vir)

        def pipeline():
            B = trntensor.ao_to_mo_transform(eri_x, C_occ_x, C_vir_x)
            E = trntensor.mp2_energy(B, eps_occ_x, eps_vir_x)
            return trntensor.from_xla(E)

        benchmark(pipeline)


class TestDecomposeBench:
    def test_cp_rank8(self, benchmark):
        T = torch.randn(16, 16, 16)
        benchmark(trntensor.cp_decompose, T, rank=8, max_iter=20)

    def test_tucker_low_rank(self, benchmark):
        T = torch.randn(16, 16, 16)
        benchmark(trntensor.tucker_decompose, T, ranks=(4, 4, 4))
