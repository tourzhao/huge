## Reason for a second submission two days after 2.0.0

We are aware that 2.0.0 was published on 2026-07-26, so this submission will
raise a "days since last update" NOTE. We are submitting anyway because this
release exists solely to fix the ATLAS additional issue that your check farm
reported for 2.0.0, which we received a deadline of 2026-08-21 for:

<https://www.stats.ox.ac.uk/pub/bdr/Rblas/ATLAS/huge.out>

That check showed 4 failures in `tests/testthat/test-select.R`. This release
contains nothing else: no new features, no API changes, and no unrelated
refactoring. We would rather correct a numerical-portability bug promptly than
leave a package on CRAN that fails on one of the tested BLAS implementations.

## Test environments

* macOS 26.5.2, R 4.5.2 (aarch64-apple-darwin20), Accelerate BLAS
* Ubuntu, R 4.6.1 (2026-06-24) (GitHub Actions)
* Ubuntu, R Under development (unstable) (2026-06-21 r90185) (GitHub Actions)
* Windows, R 4.6.1 (GitHub Actions)

## R CMD check results

0 errors. Examples and tests are OK on every platform above.

`R CMD check --as-cran` on the submitted tarball reports 2 WARNINGs and 1 NOTE
locally on macOS. All three are properties of this machine, not of the package:

* NOTE "Days since last update: 2", explained above.
* WARNING from `checking whether package 'huge' can be installed`: Apple
  clang's warning for an unsupported diagnostic pragma in R's own
  `R_ext/Boolean.h`. It is not emitted by package code.
* WARNING from `checking top-level files`: this machine's Homebrew `autoconf`
  installation cannot exec `aclocal`, so `autoreconf` fails, and the
  `checkbashisms` script is also unavailable. Both are missing local tools.

The Linux and Windows checks listed above are clean.

## What the ATLAS failures were, and what we changed

RIC computes rotated inner products with BLAS matrix products. A conforming
BLAS is free to reassociate, block, vectorize, and use fused multiply-add, so a
mathematically zero inner product may be returned as a tiny nonzero value whose
magnitude differs between implementations. This is what ATLAS does on
rank-deficient input, and two defects followed from it:

* The tiny value was reported as the selected regularization parameter, so
  `opt.lambda` was no longer exactly zero.
* `huge.select()` gated its zero-lambda safety fallback on a bitwise-zero
  lambda. The tiny value bypassed that fallback, so graphical lasso fitted
  rank-deficient data almost unregularized and raised
  "glasso produced non-finite estimates".

`huge:::ric` now certifies a rotated inner product to exact zero when its
magnitude lies within that pair's standard dot-product forward-error bound
(`gamma_n * ||u|| * ||v||` with `gamma_n = n*eps/(1 - n*eps)`, plus one eps for
the rounding introduced by splitting the rotation into two matrix products).
The refit routing compares against that same roundoff scale instead of against
zero. The bound is deliberately not a fixed tolerance: it is pair-specific and
scale-aware, so it does not erase weak correlations that the working precision
can represent. On realistic data the margin is wide; for the graph in
`huge.generator(n = 100, d = 30)` the selected lambda sits about 1e13 times
above the bound.

We did not simply restore the scalar accumulation loop used before 2.0.0. A
compiler may contract `s += u[i]*v[i]` into a fused multiply-add, and FMA
accumulation is exactly what leaves such a residual; on the data in the failing
test, sequential summation gives 0 while FMA accumulation gives 5.63e-18. The
older code was therefore not BLAS-independent either, and restoring it would
have hidden the defect rather than fixed it.

Because ATLAS is not available to us locally, we verified the fix by linking
the C++ core against a deliberately adversarial but *conforming* BLAS: it
computes each dot product exactly by compensated summation, then perturbs the
result by the largest amount the forward-error bound permits. This exercises the
space of legal BLAS implementations rather than guessing ATLAS's blocking
strategy. It reproduces the reported failure on 2.0.0 in all 16 probed cases
(cyclic shifts crossed with rescalings) and yields exact zero in all 16 with
this release, including under 4-thread OpenMP. The expected values were
confirmed independently with exact rational arithmetic.

New regression tests in `tests/testthat/test-regressions.R` pin both
directions: numerical zeros stay certified, and representable weak correlations
survive. For the second direction the test input needs care, because RIC
minimizes over rotations: if any rotation yields an exact zero, the selected
lambda is zero for any tolerance, so such input cannot detect an
over-aggressive bound. The test therefore uses input where every rotation is
nonzero and the smallest sits at roughly 2.4 and 24 times the bound. We
confirmed by mutation testing that it discriminates: inflating the bound by
1e4 erases both signals and fails the test.

No user-visible API changed. Timing is unchanged; the added work is O(nd)
against RIC's existing O(t*n*d^2).

## Reverse dependencies

The change affects only which regularization parameter RIC selects on input
where that parameter is mathematically zero, so we did not expect reverse
dependency breakage.

We re-checked six of the eight reverse dependencies against this release on
macOS: `heterocop`, `NetGreg`, `sparsenetgls` (Status OK), `SparseTSCGM`
(1 WARNING), and `netgwas`, `nutriNetwork` (ERRORs). All four non-OK results
are unrelated to `huge`:

* `netgwas` and `nutriNetwork` fail in their own core-count logic. Both compute
  `ncores <- detectCores() - 1` and then test `if(!ncores)`
  (e.g. `nutriNetwork/R/selectnet.R:86-88`). `parallel::detectCores()` returns
  `NA` on our machine, so that condition raises "missing value where
  TRUE/FALSE needed" before any `huge` code is reached. We confirmed this is
  pre-existing by re-running both against `huge` 2.0.0 in an otherwise
  identical library: they produce exactly the same ERRORs.
* `SparseTSCGM`'s WARNING is Apple clang's warning for an unsupported
  diagnostic pragma in R's own `R_ext/Boolean.h`, i.e. the same toolchain
  diagnostic noted above.

`nethet` and `SpiecEasi` depend on Bioconductor packages that we could not
install in this environment, so we did not re-check them for 2.0.1. Both
completed successfully against 2.0.0 (`nethet`'s tests and vignette R code
passed then; only its PDF build was unavailable for lack of `pdflatex`), and
neither exercises RIC's zero-lambda boundary.

No new problems attributable to `huge` were found.
