## Test environments

* macOS Tahoe 26.5, R 4.5.2 (aarch64-apple-darwin20)
* Debian Linux, R 4.6.1 development (aarch64-linux-gnu), OpenMP enabled

## R CMD check results

The final source tarball completed `R CMD check --as-cran` with 0 errors,
0 warnings, and 1 note on Debian Linux. The note reports
`-mbranch-protection=standard`; this flag is supplied by that R installation's
default `CXX17FLAGS`, not by the package.

The local macOS check additionally reports Apple clang's warning for an
unsupported diagnostic pragma in R's own `R_ext/Boolean.h`, an unavailable
`checkbashisms` executable, and inability to verify the current time. None is
emitted by package code.

## Submission

This release fixes the ATLAS additional issue reported for 2.0.0
(<https://www.stats.ox.ac.uk/pub/bdr/Rblas/ATLAS/huge.out>), which showed
4 test failures in `tests/testthat/test-select.R`.

RIC computes rotated inner products with BLAS matrix products. A conforming
BLAS may return any value inside a dot product's roundoff interval, so a
mathematically zero regularization parameter could be returned as a tiny
positive value. Two defects followed from that:

* The tiny value was reported as the selected lambda, so `opt.lambda` was no
  longer exactly zero.
* `huge.select()` gated its zero-lambda safety fallback on a bitwise-zero
  lambda. The tiny value bypassed that fallback, so graphical lasso fitted
  rank-deficient data almost unregularized and raised
  "glasso produced non-finite estimates".

RIC now certifies a rotated inner product to exact zero when its magnitude
lies within that pair's standard dot-product forward-error bound
(`gamma_n * ||u|| * ||v||`, `gamma_n = n*eps/(1 - n*eps)`), and the refit
routing in both packages compares against the same roundoff scale rather than
against zero. The bound is pair-specific and scale-aware, so it does not erase
weak correlations that the working precision can represent; the existing test
for a representable `1.5e-15` correlation still passes.

We verified the fix by linking the core against a deliberately adversarial but
conforming BLAS that returns the largest error the bound permits. That
reproduces the reported failure on 2.0.0 in all 16 probed cases (cyclic shifts
crossed with rescalings) and yields exact zero in all 16 with this release. New
regression tests in `tests/testthat/test-regressions.R` pin both directions:
numerical zeros stay certified, and representable weak correlations survive.

No user-visible API changed.

## Reverse dependencies

We checked all eight identified reverse dependencies: `heterocop`, `NetGreg`,
`netgwas`, `nethet`, `nutriNetwork`, `sparsenetgls`, `SparseTSCGM`, and
`SpiecEasi`.

Seven completed with Status OK. For `nethet`, tests and all vignette R code
completed successfully; its full PDF build was unavailable because `pdflatex`
was absent from that check container, while its remaining warning and notes
are diagnostics from `nethet` itself. No new problems attributable to `huge`
were found.
