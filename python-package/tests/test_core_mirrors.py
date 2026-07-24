"""Guard: the shared C++ core must stay byte-identical between packages.

The core exists as two copies (R: src/, Python: python-package/cpp/) that are
synced by hand.  This test fails fast when an edit lands on only one side.
It runs from a repo checkout; in an sdist/wheel install the R-side tree is
absent and the test skips.
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]

MIRROR_PAIRS = [
    ("src/huge_core.cpp", "cpp/huge_core.cpp"),
    ("src/huge/huge_core.h", "cpp/include/huge/huge_core.h"),
    ("src/huge/blas_config.h", "cpp/include/huge/blas_config.h"),
]


@pytest.mark.parametrize("r_rel,py_rel", MIRROR_PAIRS)
def test_core_copies_identical(r_rel: str, py_rel: str) -> None:
    r_file = REPO_ROOT / r_rel
    if not r_file.is_file():
        pytest.skip("R-side source tree not present (not a repo checkout)")
    py_file = PKG_ROOT / py_rel
    assert py_file.is_file(), f"missing Python-side core file: {py_rel}"
    assert r_file.read_bytes() == py_file.read_bytes(), (
        f"shared core copies differ: {r_rel} vs python-package/{py_rel} — "
        "copy the edited side over the stale side and rebuild both packages"
    )
