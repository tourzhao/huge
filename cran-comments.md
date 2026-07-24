## Test environments
* macOS Tahoe 26.5, R 4.5.2 (aarch64-apple-darwin20)
* GitHub Actions: ubuntu-latest R release, ubuntu-latest R devel, windows-latest R release

## R CMD check results
* All three CI platforms: Status OK (0 errors, 0 warnings, 0 notes)
* Local `R CMD check --as-cran`: 2 local-only WARNINGs and 1 local-only
  NOTE — R's own header (`R_ext/Boolean.h`) triggering
  `-Wunknown-warning-option` under Apple clang 21, the missing
  `checkbashisms` script, and `unable to verify current time`. None come
  from package code; none appear on CRAN/CI servers.

## Submission
This is a major update from version 1.6 to 2.0.0. Changes include:

* Fixed tiger method returning asymmetric precision matrices
* Fixed RIC selection scale-dependence (rescaling input data silently
  changed the selected graph)
* Fixed huge.npn(npn.func = "skeptic") erroring on input with row names
* Added huge.select(num.cores = k): parallel StARS subsampling via the
  base parallel package (new Imports: parallel)
* Performance: BLAS matrix products in RIC (~10-20x), strong screening rule
  with KKT certification in mb (~1.2x), incremental residual updates in
  tiger (~1.4x), active-set covariance updates in glasso (~1.15x),
  BLAS crossprod correlations and other R-layer vectorizations (ct ~2x,
  skeptic ~5x, roc ~3x, generator ~1.6x)
* Reduced memory use by keeping sparse matrices sparse in the R layer
* Fixed graphics parameter leaks in all plot functions (par restored on exit)
* Documentation corrections (dimensions, defaults, typos)
