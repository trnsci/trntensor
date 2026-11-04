"""CPU compile check: every NKI kernel lowers to MLIR that passes verification.

``nki.simulate`` does not run the MLIR verifier, so a kernel can pass every
simulator test and still fail to compile for hardware (#41). This module runs
the NKI frontend (trace -> MLIR -> verify) on CPU for every kernel and target.
It needs ``nki`` but no Neuron device and no ``neuronx-cc``; it does not run
the backend compiler, so errors raised after MLIR generation (e.g. the BIR
verifier's DMA-transpose check) are out of scope here.

Shapes are the post-padding shapes the dispatch layer hands each kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.nki_simulator

from trntensor.nki._kernels import HAS_NKI  # noqa: E402

try:
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    source_to_mlir = None


def _r(*shape):
    return np.random.default_rng(0).random(shape, dtype=np.float32)


def _u32(*shape):
    return np.zeros(shape, dtype=np.uint32)


def _blocked(nb, nocc, nvir, naux, rows=1000):
    """Inputs for ``ao_to_mo_trn2_packed_blocked_kernel`` as the dispatcher pads them."""
    from trntensor.nki._kernels import ao_to_mo_blocked_nub, ao_to_mo_blocked_tp

    tp = ao_to_mo_blocked_tp(naux)
    nub = ao_to_mo_blocked_nub(tp, nocc)
    npc = naux // tp
    img = _u32(npc, nb // nub, 128, nb // 128 * nub)
    return (_r(rows * npc, tp), img, _r(nb, nocc), _r(nb, nvir)), {}


# kernel name -> [(case label, () -> (args, kwargs))]
KERNEL_INPUTS = {
    "matmul_kernel": [
        ("128x128x128", lambda: ((_r(128, 128), _r(128, 128)), {})),
        ("256x256x1024", lambda: ((_r(256, 256), _r(256, 1024)), {})),
        ("256x256x1024_sr", lambda: ((_r(256, 256), _r(256, 1024)), {"use_sr": True})),
    ],
    "batched_matmul_kernel": [
        ("b2_128", lambda: ((_r(2, 128, 128), _r(2, 128, 128)), {})),
        ("b2_128_sr", lambda: ((_r(2, 128, 256), _r(2, 256, 128)), {"use_sr": True})),
    ],
    "mp2_energy_kernel": [
        # (B_T (nocc, naux, nvir), eps_occ_b (128, nocc), eps_vir (nvir, 1), eps_vir_b (128, nvir))
        (
            "nocc5_nvir128_naux128",
            lambda: ((_r(5, 128, 128), _r(128, 5), _r(128, 1), _r(128, 128)), {}),
        ),
        (
            "nocc2_nvir256_naux256",
            lambda: ((_r(2, 256, 256), _r(128, 2), _r(256, 1), _r(128, 256)), {}),
        ),
    ],
    "ao_to_mo_transform_kernel": [
        ("nb256_nocc16_nvir512_naux4", lambda: ((_r(256, 256, 4), _r(256, 16), _r(256, 512)), {})),
        (
            "nb512_nocc10_nvir1024_naux2",
            lambda: ((_r(512, 512, 2), _r(512, 10), _r(512, 1024)), {}),
        ),
    ],
    # Native Trn2 kernels (#51): gridless, split by num_programs at lnc=2.
    "mp2_energy_row_kernel": [
        # (B[i].T, window (ND, naux, nvir), eps_i (1,1), eps window (1,ND), eps_vir col/row)
        (
            "nd4_naux256_nvir256",
            lambda: (
                (_r(256, 256), _r(4, 256, 256), _r(1, 1), _r(1, 4), _r(256, 1), _r(1, 256)),
                {},
            ),
        ),
        (
            "nd2_naux384_nvir640",
            lambda: (
                (_r(384, 640), _r(2, 384, 640), _r(1, 1), _r(1, 2), _r(640, 1), _r(1, 640)),
                {},
            ),
        ),
    ],
    "ao_to_mo_trn2_packed_kernel": [
        # (packed (npair·NPC, TP) rows, (NPC, 128, NKB·NB) uint32 gather indices, C_occ, C_vir)
        (
            "nb256_nocc32_nvir256_naux512",
            lambda: (
                (_r(32896 * 2, 256), _u32(2, 128, 512), _r(256, 32), _r(256, 256)),
                {},
            ),
        ),
        (
            "nb384_nocc48_nvir384_naux1152",
            lambda: (
                (_r(73920 * 3, 384), _u32(3, 128, 1152), _r(384, 48), _r(384, 384)),
                {},
            ),
        ),
        (
            "nb256_nocc258_nvir128_naux384_oddchunks",
            lambda: (
                (_r(32896 * 3, 128), _u32(3, 128, 512), _r(256, 258), _r(256, 128)),
                {},
            ),
        ),
        (
            "nb384_nocc200_nvir256_naux512",
            lambda: (
                (_r(73920 * 2, 256), _u32(2, 128, 1152), _r(384, 200), _r(384, 256)),
                {},
            ),
        ),
    ],
    # Blocked packed AO→MO (#112): (rows, TP), (NPC, NB/NUB, 128, NKB·NUB)
    # indices, C_occ, C_vir. Row counts are cut (shapes only).
    "ao_to_mo_trn2_packed_blocked_kernel": [
        ("nb256_nocc32_nvir256_naux512", lambda: _blocked(256, 32, 256, 512)),
        ("nb384_nocc200_nvir256_naux87_occsplit", lambda: _blocked(384, 200, 256, 87)),
        ("nb1280_nocc242_nvir384_naux96_kbtail", lambda: _blocked(1280, 242, 384, 96)),
        ("nb256_nocc129_nvir640_naux768", lambda: _blocked(256, 129, 640, 768)),
        ("nb256_nocc600_nvir128_naux128_passes", lambda: _blocked(256, 600, 128, 128)),
        ("nb256_nocc16_nvir128_naux48", lambda: _blocked(256, 16, 128, 48)),
    ],
    "df_half_packed_kernel": [
        # (packed (npair·NPC, TP) rows, (NPC, NVG, 128, NKB·NVC) uint32 indices, C_occ)
        (
            "nb256_nocc41_naux402",
            lambda: ((_r(32896, 402), _u32(1, 4, 128, 128), _r(256, 41)), {"nub": 2}),
        ),
        (
            "nb256_nocc20_naux390_split_chunks",
            lambda: ((_r(32896 * 2, 195), _u32(2, 2, 128, 256), _r(256, 20)), {"nub": 4}),
        ),
        # (npair·naux / W, W) rows, 128-multiple chunks and a short last one;
        # naux 1206 = 512 + 512 + 182 (npair cut to 1000: shapes only)
        (
            "nb256_nocc42_naux1206_tail",
            lambda: (
                (_r(1000 * 603, 2), _u32(3, 4, 128, 128), _r(256, 42)),
                {"nub": 8, "tp": 512, "tail": 182},
            ),
        ),
        (
            "nb256_nocc20_tail_only",
            lambda: (
                (_r(1000 * 603, 2), _u32(1, 2, 128, 256), _r(256, 20)),
                {"nub": 4, "tp": 512, "tail": 182},
            ),
        ),
    ],
    "df_exchange_syrk_kernel": [
        # (Xt (rows, NVG, nocc, NVC), running partial Ks (2, NB, NB)); partial last tile
        ("rows402_nocc6_nb256", lambda: ((_r(402, 2, 6, 128), _r(2, 256, 256)), {})),
        (
            "rows256_nocc34_nb640_banks2",
            lambda: ((_r(256, 40, 34, 16), _r(2, 640, 640)), {"banks": 2}),
        ),
    ],
    "df_coulomb_d_kernel": [
        # (packed B (npair, naux), weighted packed density (npair, 1))
        ("npair1000_naux450", lambda: ((_r(1000, 450), _r(1000, 1)), {})),
        ("npair600_naux4500", lambda: ((_r(600, 4500), _r(600, 1)), {})),
    ],
    "df_coulomb_j_kernel": [
        # (packed B (npair, naux), d replicated on 128 partitions)
        ("npair1000_naux450", lambda: ((_r(1000, 450), _r(128, 450)), {})),
        ("npair600_naux4500", lambda: ((_r(600, 4500), _r(128, 4500)), {})),
    ],
    "ao_to_mo_trn2_kernel": [
        (
            "nb256_nocc32_nvir256_naux512",
            lambda: ((_r(256, 256, 512), _r(256, 32), _r(256, 256)), {}),
        ),
        (
            "nb128_nocc8_nvir128_naux128",
            lambda: ((_r(128, 128, 128), _r(128, 8), _r(128, 128)), {}),
        ),
        # nocc > 128: occupied blocks in stage 1 (#80).
        (
            "nb256_nocc129_nvir128_naux256",
            lambda: ((_r(256, 256, 256), _r(256, 129), _r(256, 128)), {}),
        ),
        (
            "nb384_nocc257_nvir128_naux256",
            lambda: ((_r(384, 384, 256), _r(384, 257), _r(384, 128)), {}),
        ),
        # Odd aux chunk count (3 x 128): the programs split the occupied index.
        (
            "nb256_nocc200_nvir128_naux384",
            lambda: ((_r(256, 256, 384), _r(256, 200), _r(256, 128)), {}),
        ),
    ],
}


@pytest.fixture(autouse=True)
def _frontend_available():
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    if source_to_mlir is None:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")


# torch-neuronx compiles trn2 kernels at lnc=2 by default; cover both.
@pytest.mark.parametrize("target,lnc", [("trn1", 1), ("trn2", 1), ("trn2", 2)])
@pytest.mark.parametrize(
    "kernel_name,make_inputs",
    [(name, mk) for name, cases in sorted(KERNEL_INPUTS.items()) for _, mk in cases],
    ids=[f"{name}-{label}" for name, cases in sorted(KERNEL_INPUTS.items()) for label, _ in cases],
)
def test_kernel_mlir_verifies(kernel_name, make_inputs, target, lnc):
    from trntensor.nki import _kernels

    args, kwargs = make_inputs()
    result = source_to_mlir(
        getattr(_kernels, kernel_name),
        *args,
        target=target,
        lnc=lnc,
        frontend=FrontendKind.TRACER,
        **kwargs,
    )
    result.verify()
