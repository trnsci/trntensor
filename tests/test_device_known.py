"""The device-known xfail rules of tests/test_conformance.py (trnsci/trnsci#35).

CPU only: checks that representative case IDs map to their category, that
other cases map to none, and that the device-run summary groups xfail
reasons by category.
"""

from types import SimpleNamespace

import pytest
import torch
from conftest import _expect_category, _expect_summary
from test_conformance import DEVICE_KNOWN, _device_known_category


def _node(name, **params):
    callspec = SimpleNamespace(params=params) if params else None
    return SimpleNamespace(originalname=name, name=name, callspec=callspec)


@pytest.mark.parametrize(
    "name,params,cat",
    [
        ("test_tucker", {"dtype": torch.complex64}, "complex-linalg"),
        ("test_tucker", {"dtype": torch.complex128}, "complex-linalg"),
        ("test_tucker", {"dtype": torch.float32}, None),
        ("test_tucker", {"dtype": torch.float64}, None),
        ("test_tt_svd_error_bound", {"dtype": torch.complex64, "max_rank": 64}, "complex-linalg"),
        ("test_tt_svd_error_bound", {"dtype": torch.float32, "max_rank": 2}, None),
        ("test_cp_decompose_non_fp32", {"dtype": torch.complex128, "warm": True}, "complex-linalg"),
        ("test_cp_decompose_non_fp32", {"dtype": torch.float64, "warm": False}, None),
        ("test_cp_reconstruct_formula", {}, "device-fp64"),
        ("test_einsum_matmul", {"dtype": torch.complex64}, None),
    ],
)
def test_device_known_category(name, params, cat):
    assert _device_known_category(_node(name, **params)) == cat
    if cat is not None:
        assert cat in DEVICE_KNOWN


def test_expect_category_and_summary():
    known = "[device-known:complex-linalg] trnsci/trntensor#130: complex svd"
    assert _expect_category(known) == "complex-linalg"
    assert _expect_category("trnsci/trntensor#1: pending fix") == "other xfail"
    rep = SimpleNamespace
    stats = {
        "passed": [rep(nodeid="a", when="call")],
        "xfailed": [rep(nodeid="b", when="call", wasxfail=known)],
        "xpassed": [rep(nodeid="c", when="call", wasxfail=known)],
        "failed": [rep(nodeid="d", when="call")],
    }
    s = _expect_summary(stats)
    assert s["passed"] == 1
    assert s["xfailed"] == {"complex-linalg": ["b"]}
    assert s["xpassed"] == {"complex-linalg": ["c"]}
    assert s["new_failures"] == ["d"]
