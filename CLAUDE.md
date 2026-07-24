# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**huge** (High-Dimensional Undirected Graph Estimation) provides sparse graphical model estimation and inference in high dimensions, shipped as two packages from one repo:

- **R package `huge`** (CRAN) — lives at the repo root; version in `DESCRIPTION`
- **Python package `pyhuge`** (PyPI) — lives under `python-package/`; native Python, no R runtime; version in `python-package/pyproject.toml`

Both wrap the same C++ core: plain C++17 with direct Fortran-BLAS calls (`dgemv_`, `ddot_`, …) and OpenMP — no Rcpp, pybind11, or Eigen types inside the core (RcppEigen was removed in 1.5).

## The shared C++ core (critical)

The algorithmic core exists as **two byte-identical copies** that must be kept in sync manually:

|              | R package                                    | Python package                              |
|--------------|----------------------------------------------|---------------------------------------------|
| Implementation | `src/huge_core.cpp`                        | `python-package/cpp/huge_core.cpp`          |
| Headers      | `src/huge/huge_core.h`, `src/huge/blas_config.h` | `python-package/cpp/include/huge/` (same files) |

**If you edit the core on one side, copy the change to the other side** and verify with `tools/check_core_mirrors.sh` (also enforced in CI and by `python-package/tests/test_core_mirrors.py`). The core exposes `huge::glasso`, `huge::mb`, `huge::mb_scr`, `huge::tiger`, `huge::ric`, `huge::sfgen`, all operating on raw column-major `double*` buffers.

Thin adapter layers sit around the core:

- R: `src/hugeglasso.cpp`, `src/SPMBgraph.cpp` (mb), `src/SPMBgraphsqrt.cpp` (tiger), `src/RIC.cpp`, `src/SFGen.cpp` — Rcpp wrappers that copy core results directly into R SEXP memory
- Python: `python-package/cpp/native_core_bindings.cpp` — pybind11 module `pyhuge._native_core`

BLAS linkage: R gets `$(BLAS_LIBS)` via `src/Makevars`; Python uses the Accelerate framework on macOS and OpenBLAS on Linux (logic in `python-package/setup.py`).

## Build & Test Commands

### R package (repo root)

```bash
R CMD INSTALL .                                            # build + install (required before tests)
Rscript -e 'testthat::test_dir("tests/testthat")'          # run test suite
Rscript -e 'testthat::test_file("tests/testthat/test-glasso.R")'  # single test file
Rscript tests/test.R                                       # legacy smoke script
R CMD build .                                              # build tarball
R CMD check huge_*.tar.gz                                  # CRAN-style check
```

- `./configure` (from `configure.ac`) detects OpenMP and generates `src/Makevars` from `src/Makevars.in`; `./cleanup` removes generated config files. On macOS, OpenMP needs `brew install llvm` and `~/.R/Makevars` pointing at that clang.
- **Generated files — never hand-edit**:
  - `NAMESPACE`, `man/*.Rd` — edit roxygen comments in `R/*.R`, then `Rscript -e 'roxygen2::roxygenise(".")'`
  - `R/RcppExports.R`, `src/RcppExports.cpp` — after changing `//[[Rcpp::export]]` signatures, run `Rscript -e 'Rcpp::compileAttributes(".")'`
- `.Rbuildignore` excludes `python-package/`, `benchmark/`, `CLAUDE.md`, tarballs, etc. from the CRAN tarball — update it when adding top-level files.
- Vignette is `vignettes/huge.Rnw` (Sweave); CI skips vignette building.
- CRAN release touchpoints: `DESCRIPTION` (Version), `NEWS.md`, `cran-comments.md`.

### Python package (`python-package/`)

```bash
cd python-package
pip install -e ".[dev]"                  # editable install; builds the pybind11 extension
pytest                                   # run tests
pytest tests/test_core_unit.py -k npn    # single file / keyword filter
pip install -e .                         # re-run after any C++ change to rebuild _native_core
python -c "import pyhuge; print(pyhuge.test())"   # smoke check
pyhuge-doctor                            # environment diagnostics
mkdocs build --strict                    # validate docs
bash scripts/build_dist.sh               # wheel + sdist
python scripts/bump_version.py 2.0.0 && bash scripts/release.sh 2.0.0
```

- R-parity tests (`tests/test_native_vs_r_parity.py`) shell out to a local `R` with the `huge` package installed and auto-skip otherwise; plotting tests skip without matplotlib/networkx.
- The native extension is **required** for glasso/mb/tiger (`_require_native_core` raises); only minor helpers fall back to numpy when `_native_core` is missing.

### Standalone core

`CMakeLists.txt` at the root builds the bare core as library `huge_core` with no R or Python — useful to verify the core compiles in isolation.

## Architecture

Layering is the same in both languages: user API → dispatcher → method function → thin C++ wrapper → shared core.

- R: one file per exported function under `R/`; `huge.R` dispatches on `method` to `huge.mb.R` / `huge.glasso.R` / `huge.ct.R` / `huge.tiger.R`; plus `huge.select.R`, `huge.npn.R`, `huge.generator.R`, `huge.inference.R`, `huge.roc.R`, `huge.plot.R`.
- Python: all estimators, selection, transforms, inference, and plotting live in `pyhuge/core.py`, re-exported by `pyhuge/__init__.py`; `pyhuge/parity.py` runs R reference implementations for comparison; `pyhuge/doctor.py` backs the `pyhuge-doctor` CLI.

### Shared pipeline concept

Both packages implement the same modeling pipeline (R name / Python name):

1. **Generate** synthetic data (`huge.generator` / `huge_generator`)
2. **Preprocess** via nonparanormal transform (`huge.npn` / `huge_npn`)
3. **Estimate** graph structure (`huge` with method = `glasso`/`mb`/`tiger`/`ct`)
4. **Select** regularization parameter (`huge.select` / `huge_select` with criterion = `stars`/`ric`/`ebic`)
5. **Infer** edge significance (`huge.inference` / `huge_inference`)
6. **Evaluate** via ROC (`huge.roc` / `huge_roc`)

### CI workflows (`.github/workflows/`)

- `r-cmd-check.yml` — R CMD check on Linux (release + devel) and Windows; vignettes skipped; triggers only on R-side paths
- `python-package-tests.yml` — pytest for pyhuge
- `python-package-docs.yml` — builds and deploys the MkDocs site
- `python-package-release.yml` — publishes pyhuge to PyPI
- `python-wrapper-tests.yml` — tests the older rpy2 wrapper

### Benchmarks

`benchmark/` holds scripts comparing the local build against CRAN (`bench_vs_cran.R`, `bench_report.R`) and pyhuge timing (`bench_pyhuge_*.py`). Not part of either package build.
