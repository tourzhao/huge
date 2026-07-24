# Repository Guidelines

## Project Structure & Module Organization

This repository ships two packages over one C++17 solver core. The root is the R package: public functions live in `R/`, Rcpp adapters and native code in `src/`, `testthat` tests in `tests/testthat/`, and package data and documentation in `data/`, `man/`, and `vignettes/`. The native Python package is under `python-package/`, with API code in `pyhuge/`, pybind11 code in `cpp/`, pytest tests in `tests/`, and supporting `docs/`, `examples/`, and `scripts/`. Benchmarks belong in `benchmark/`; repository checks belong in `tools/`.

## Build, Test, and Development Commands

- `R CMD INSTALL .` builds and installs the R package.
- `Rscript -e 'testthat::test_dir("tests/testthat")'` runs its primary test suite; use `testthat::test_file(...)` for one file.
- `R CMD build .` followed by `R CMD check huge_*.tar.gz` performs a CRAN-style package check.
- `cd python-package && python -m pip install -e '.[dev]'` installs editable Python code and builds the native extension. Re-run it after C++ changes.
- From `python-package/`, run `pytest`; narrow runs with commands such as `pytest tests/test_core_unit.py -k npn`.
- `mkdocs build --strict` validates Python documentation.
- `tools/check_core_mirrors.sh` verifies the shared native sources.

## Coding Style & Naming Conventions

Use four-space indentation in Python and C++; match the surrounding legacy style in R. Python functions use `snake_case`, dataclasses use `PascalCase`, and private helpers start with `_`. C++ uses `snake_case` within namespace `huge`. Preserve R's dotted public API names (for example, `huge.select`) and document exports with roxygen comments. No repository-wide formatter is configured, so keep formatting-only changes focused.

Do not hand-edit `NAMESPACE`, `man/*.Rd`, or Rcpp export files. Regenerate them with `roxygen2::roxygenise(".")` or `Rcpp::compileAttributes(".")` as appropriate.

## Testing Guidelines

Name R tests `test-<feature>.R` and Python tests `test_<feature>.py`. Add regression coverage for every externally visible behavior change; no numeric coverage threshold is configured. R-parity tests require a locally installed `huge` package and otherwise skip. Changes to the core must keep `src/huge_core.cpp` and `src/huge/*.h` byte-identical to their counterparts under `python-package/cpp/`.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative, sentence-case subjects such as `Fix ...`, `Add ...`, or `Remove ...`; optional scopes such as `CI:` are acceptable. For substantial changes, explain the rationale and group details by R, Python, or C++ impact. Pull requests should summarize affected layers, link relevant issues, identify compatibility or generated-file changes, and report the exact tests run. Include screenshots only for plotting or rendered-document changes.
