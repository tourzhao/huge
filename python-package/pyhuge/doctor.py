"""Runtime diagnostics for native pyhuge."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any

import importlib.util as _importlib_util
from importlib import metadata as _metadata


def _pkg_version(name: str) -> str | None:
    if _importlib_util.find_spec(name) is None:
        return None
    try:
        return _metadata.version(name)
    except Exception:
        return None


def _native_omp_threads() -> int | None:
    """OpenMP thread count of the native core (1 = serial build).

    Returns None when the native extension is missing or predates the
    diagnostic binding.
    """
    try:
        from . import _native_core

        return int(_native_core.omp_max_threads())
    except Exception:
        return None


def diagnose() -> dict[str, Any]:
    """Collect environment diagnostics."""

    from . import test

    test_status = test(require_runtime=False)
    return {
        "native_core": {
            "omp_max_threads": _native_omp_threads(),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "architecture": platform.machine(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "packages": {
            "pyhuge": _pkg_version("pyhuge"),
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
            "matplotlib": _pkg_version("matplotlib"),
            "networkx": _pkg_version("networkx"),
        },
        "status": test_status,
    }


def main() -> int:
    """CLI entrypoint for ``pyhuge-doctor``."""

    info = diagnose()
    print("pyhuge-doctor")
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0 if info["status"].get("runtime", False) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
