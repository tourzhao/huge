"""Public API smoke tests for the native pyhuge package."""

from __future__ import annotations

import pytest

import pyhuge


def test_test_returns_status_keys():
    status = pyhuge.test()
    assert status["python_import"] is True
    for k in ("numpy", "scipy", "sklearn", "native_extension", "runtime", "rpy2"):
        assert k in status


def test_test_raises_when_runtime_required_and_missing(monkeypatch):
    class _FakeUtil:
        @staticmethod
        def find_spec(name: str):
            if name in {"numpy", "scipy", "sklearn"}:
                return None
            return object()

    monkeypatch.setattr(pyhuge, "_importlib_util", _FakeUtil)
    with pytest.raises(pyhuge.PyHugeError, match="Native runtime for pyhuge is unavailable"):
        pyhuge.test(require_runtime=True)


def test_public_symbols_exported():
    required = {
        "huge",
        "huge_mb",
        "huge_glasso",
        "huge_ct",
        "huge_tiger",
        "huge_select",
        "huge_npn",
        "huge_generator",
        "huge_inference",
        "huge_roc",
        "huge_stockdata",
        "huge_plot",
        "test",
    }
    assert required.issubset(set(pyhuge.__all__))
