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

## Reverse dependencies

We checked all eight identified reverse dependencies: `heterocop`, `NetGreg`,
`netgwas`, `nethet`, `nutriNetwork`, `sparsenetgls`, `SparseTSCGM`, and
`SpiecEasi`.

Seven completed with Status OK. For `nethet`, tests and all vignette R code
completed successfully; its full PDF build was unavailable because `pdflatex`
was absent from that check container, while its remaining warning and notes
are diagnostics from `nethet` itself. No new problems attributable to `huge`
were found.

## Submission

This is a major update to version 2.0.0 focused on correctness, numerical
validation, and bounded parallel execution.

* TIGER now receives the original matrix and resolved input type in C++.
  Covariance validation, correlation construction, automatic lambda selection,
  and fitting therefore use the same native correlation matrix.
* Glasso symmetrizes and verifies precision estimates, checks positive
  definiteness, and certifies covariance/precision consistency. Indefinite
  pairwise covariance estimates are accepted only when regularization produces
  a certified result.
* Auto-detected glasso covariance input retains the historical default-lambda
  scale for compatibility. Explicit covariance input uses the corrected
  off-diagonal scale; explicit lambda paths are unchanged.
* Parallel StARS starts at most two workers. Package-owned OpenMP regions use
  one thread in each child; the default remains serial.
* Input and lambda-path validation, solver diagnostics, documentation, and
  memory handling were also improved.
