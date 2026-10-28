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
