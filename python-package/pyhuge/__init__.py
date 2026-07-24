"""pyhuge: native Python package for high-dimensional undirected graph estimation."""

from __future__ import annotations

import importlib.util as _importlib_util

from . import core as _core
from .core import (
    HugeGeneratorResult,
    HugeInferenceResult,
    HugeResult,
    HugeRocResult,
    HugeSelectResult,
    HugeSelectSummary,
    HugeStockDataResult,
    HugeSummary,
    PyHugeError,
    huge,
    huge_ct,
    huge_generator,
    huge_glasso,
    huge_inference,
    huge_mb,
    huge_npn,
    huge_plot,
    huge_plot_graph_matrix,
    huge_plot_network,
    huge_plot_roc,
    huge_plot_sparsity,
    huge_roc,
    huge_select,
    huge_select_summary,
    huge_stockdata,
    huge_summary,
    huge_tiger,
)

__all__ = [
    "PyHugeError",
    "HugeResult",
    "HugeSelectResult",
    "HugeGeneratorResult",
    "HugeInferenceResult",
    "HugeRocResult",
    "HugeStockDataResult",
    "HugeSummary",
    "HugeSelectSummary",
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
    "huge_summary",
    "huge_select_summary",
    "huge_plot",
    "huge_plot_sparsity",
    "huge_plot_roc",
    "huge_plot_graph_matrix",
    "huge_plot_network",
    "test",
]

__version__ = "2.0.0"


def test(require_runtime: bool = False) -> dict[str, bool]:
    """Probe environment readiness for native pyhuge.

    Returned keys include compatibility fields from earlier wrapper versions.
    """

    status = {
        "python_import": True,
        "rpy2": _importlib_util.find_spec("rpy2") is not None,
        "numpy": _importlib_util.find_spec("numpy") is not None,
        "scipy": _importlib_util.find_spec("scipy") is not None,
        # Compatibility field; no longer required for runtime.
        "sklearn": _importlib_util.find_spec("sklearn") is not None,
        "native_extension": getattr(_core, "_CPP", None) is not None,
    }
    status["runtime"] = bool(status["numpy"] and status["scipy"] and status["native_extension"])

    if require_runtime and not status["runtime"]:
        raise PyHugeError(
            "Native runtime for pyhuge is unavailable. "
            "Install dependencies with `pip install \"pyhuge[runtime]\"`."
        )

    return status
