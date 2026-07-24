from __future__ import annotations

import os
import sys

if sys.platform != "darwin" and not sys.platform.startswith("linux"):
    raise RuntimeError(
        "pyhuge native source builds currently support only Linux and macOS."
    )

import numpy
import pybind11
from setuptools import Extension, setup

_HERE = os.path.dirname(os.path.abspath(__file__))


def _get_blas_args():
    """Return (libraries, library_dirs, extra_link_args) for BLAS."""
    if sys.platform == "darwin":
        # macOS: use Accelerate framework (always available)
        return [], [], ["-framework", "Accelerate"]
    else:
        # Linux: link against OpenBLAS (or generic BLAS/CBLAS)
        return ["openblas"], [], []


def _get_openmp_args():
    """Return (extra_compile_args, extra_link_args) enabling OpenMP if found.

    The core parallelizes its per-column solvers with OpenMP. Linux source
    builds pass ``-fopenmp`` by default; macOS probes for Homebrew libomp and
    falls back to serial when it is absent. Set PYHUGE_NO_OPENMP=1 to force a
    serial build on either platform.
    """
    if os.environ.get("PYHUGE_NO_OPENMP") == "1":
        return [], []
    if sys.platform == "darwin":
        for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
            if os.path.exists(os.path.join(prefix, "lib", "libomp.dylib")):
                return (
                    ["-Xpreprocessor", "-fopenmp", "-I" + os.path.join(prefix, "include")],
                    ["-L" + os.path.join(prefix, "lib"), "-lomp"],
                )
        return [], []
    return ["-fopenmp"], ["-fopenmp"]


blas_libs, blas_lib_dirs, blas_link_args = _get_blas_args()
omp_compile_args, omp_link_args = _get_openmp_args()

ext_modules = [
    Extension(
        "pyhuge._native_core",
        [
            os.path.join("cpp", "native_core_bindings.cpp"),
            os.path.join("cpp", "huge_core.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            os.path.join(_HERE, "cpp", "include"),
        ],
        libraries=blas_libs,
        library_dirs=blas_lib_dirs,
        extra_compile_args=["-O3", "-std=c++17"] + omp_compile_args,
        extra_link_args=blas_link_args + omp_link_args,
        language="c++",
    )
]

setup(ext_modules=ext_modules)
