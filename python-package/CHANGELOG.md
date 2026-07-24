# Changelog

## 2.0.0

- Release gates now install and run the generated sdist outside the checkout,
  exercise native MB/TIGER and compressed-support bindings on every published
  CPython ABI (3.9 through 3.14), and test the standalone CMake install on
  Linux and macOS. Unsupported Windows source builds now fail with a clear
  platform message instead of receiving Linux compiler flags. Public wrapper
  signatures and exact argument forwarding are covered by contracts.
- MB and TIGER now build public sparse graph paths directly from compressed
  native support. They no longer allocate a dense `nlambda * d * d`
  coefficient cube that is discarded immediately; private native calls keep
  dense output by default for compatibility. RIC also caps its OpenMP workers
  at the number of rotations so idle workers do not allocate `d * d` scratch
  matrices.
- TIGER now constructs its correlation matrix and default lambda path in the
  shared C++ core, then solves the correlation-domain square-root Lasso with
  full KKT certification. This removes Python/R preprocessing drift and fixes
  active-set solutions that could exit before checking inactive coordinates.
  Both raw observations and covariance/correlation input now use this native
  path.
  Generated paths now stop at the longest certified prefix with a warning;
  uncertifiable explicit lambda values raise an error.
- TIGER's native automatic lambda grid now forms underflowing endpoints in log
  space and saturates only below the smallest positive double. Subnormal
  correlations or ratios no longer yield zero/`NaN` paths or false solver
  certification failures; lambda selection remains in the shared C++ core.
- Explicit TIGER lambda paths now accept ties but reject increases in both
  public frontends and native C++ entry points. This makes automatic paths from
  `lambda_min_ratio=1` or subnormal saturation directly replayable in Python.
- Explicit path validation now follows the same method-level contract in R and
  Python. MB and glasso require positive, finite, non-increasing values and
  allow ties. CT accepts finite non-negative thresholds, including zero, in
  supplied order; only StARS selection requires a non-increasing CT path.
- Explicit lambda inputs now accept only a numeric scalar (including a NumPy
  0-D value) or a non-empty one-dimensional sequence. Multidimensional inputs
  fail closed instead of being flattened in R's column-major order or Python's
  row-major order; valid paths keep their method-specific rules.
- Direct native MB and glasso calls now reject empty, non-finite, non-positive,
  or increasing lambda paths before allocation and solving. R native wrappers
  validate matrix dimensions and declared path lengths before entering C++;
  Python's native CT threshold entry rejects non-square matrices and invalid
  thresholds without imposing an order on valid thresholds.
- Fixed tiger returning asymmetric `icov` matrices (shared-core symmetrization bug).
- Fixed the shared glasso core returning numerically asymmetric precision and
  graph matrices at finite solver tolerance. Precision components are now
  symmetrically projected before output and warm starts; path, edge count,
  sparsity, and log-likelihood all come from that same matrix in R and Python.
  The projection now averages half-scaled entries so large finite precision
  values do not overflow during symmetrization. Log-determinant evaluation no
  longer treats a fixed `1e-15` pivot as singular, so valid rescaled problems
  retain finite likelihoods. Truly unrepresentable precision, internal
  covariance, or log-likelihood results now fail before they can become a warm
  start or escape either language binding.
- Fixed raw one-variable inputs being collapsed to a scalar correlation by
  NumPy. CT, MB, glasso, and TIGER now return well-formed `1 x 1` paths for
  valid single-variable observations. Raw inputs with only one observation or
  a constant column are rejected before an undefined sample correlation can
  reach a solver.
- Aligned Python covariance auto-detection with R's machine-precision
  symmetry tolerance. Nearly symmetric square raw observations are no longer
  misrouted as covariance matrices; TIGER passes the corrected raw/covariance
  flag to C++, where correlation construction and automatic lambda selection
  remain.
- Fixed covariance-to-correlation conversion applying an absolute `1e-12`
  variance floor. CT now gives the same graph after uniformly rescaling a
  covariance matrix. Covariance inputs with a non-positive diagonal or a
  pairwise Cauchy--Schwarz violation are now rejected explicitly; only tiny
  correlation roundoff is clipped back to `[-1, 1]`. Symmetrization now adds
  half-scaled matrices, avoiding intermediate overflow for large finite input.
- Covariance input must now be positive semidefinite, not merely satisfy every
  pairwise Cauchy--Schwarz bound. A scale-aware roundoff tolerance preserves
  singular sample covariance matrices; TIGER enforces the same rule in the
  shared C++ core before native lambda selection and fitting.
- TIGER covariance symmetrization now avoids adding two large finite entries
  before scaling. Valid covariance matrices near the floating-point maximum
  no longer overflow during native correlation construction.
- TIGER covariance normalization now applies the larger inverse standard
  deviation first, preserving representable weak correlations across extreme
  variance ratios during native automatic lambda selection.
- Covariance projection now leaves entries that are already exactly symmetric
  untouched, so the smallest positive subnormal diagonal is not rounded to
  zero before validation.
- Direct native TIGER covariance input now rejects material asymmetry using
  the same scale-aware tolerance as the public frontend, while retaining the
  existing projection of floating-point roundoff.
- Covariance symmetry tolerance in Python, R, and native TIGER is now measured
  on the implied-correlation scale. Raw/covariance routing stays stable across
  representable finite uniform rescalings, so tiny square raw samples are no
  longer misclassified. TIGER correlation construction and automatic lambda
  selection remain in the shared C++ core.
- TIGER raw-data correlation construction now uses power-of-two column scaling,
  reference centering, and compensated sums in C++. Extreme finite scales and
  representable differences near the floating-point maximum therefore retain
  the correct native automatic lambda path.
- Python correlation thresholding, MB, and graphical lasso now standardize raw
  columns with exact binary-power scaling and reference centering. Extreme
  finite scales and adjacent values near the floating-point maximum retain the
  same correlations, automatic lambda paths, and graph estimates.
- Fixed MB and glasso default lambda paths replacing every weak nonzero
  off-diagonal value below `1e-3` with `1e-3`. The fallback is now used only
  when no off-diagonal signal exists, matching R; TIGER lambda selection
  remains in the shared C++ core.
- Fixed MB and glasso rejecting the documented `lambda_min_ratio=1` boundary.
  Automatic grids now contain the requested number of exact tied values, and
  explicit non-increasing paths may also contain ties so StARS can refit them;
  increasing paths remain invalid.
- Fixed `huge_select` RIC criterion scale-dependence: data is now standardized
  before the rotation criterion, and every positive finite sample standard
  deviation is preserved instead of applying an absolute `1e-12` floor. Thus,
  even very small rescalings no longer change selection. Its R-compatible
  empty-graph boundary now handles single-variable and exactly uncorrelated
  inputs without refitting at an invalid zero lambda.
- Fixed `huge_generator` to match the R package's model: `sigma` is now
  standardized to a correlation matrix (cov2cor) with `omega` its exact
  inverse. Previously `sigma` was the raw precision-matrix inverse with
  non-unit diagonal — a different generative model from R's, so generated
  partial correlations were weaker (this also biased any benchmark built on
  `huge_generator` data). SPD inverses now use Cholesky.
- Fixed `huge_npn` shrinkage/truncation to match R's `huge.npn` exactly
  (now agrees to ~1e-15): the previous implementation skipped the final
  per-column sd normalization, added extra clipping to the shrunken ECDF,
  and used rank/(n+1) instead of rank/n in the truncation branch.
- Fixed `huge_npn(..., npn_func="skeptic")` for one- and two-variable input.
  It now forms the rank correlation by a standardized crossproduct, preserves
  the valid nonconstant `1 x 1` boundary, and rejects fewer than two
  observations or undefined correlations caused by a constant column.
- Speedups in the shared C++ core: BLAS matrix products in RIC (~10x at d=2000),
  sequential strong screening rule with KKT certification in mb (~1.2x; estimates
  now satisfy exact KKT), contiguous memory access in the glasso log-likelihood
  (~1.3x at d=2000), and active-set W-update in glasso (~1.15x at d=1000).
- Solvers now emit a `RuntimeWarning` when an iteration budget is exhausted
  instead of silently returning partially converged estimates.
- Added `huge_select(..., n_jobs=k)`: stars subsamplings fit in a thread pool
  (~3x with 4 threads at d=1000); workers are capped at the number of
  subsamples and results are identical to serial. The native solver bindings
  now release the GIL during C++ execution. Parallel calls warn that each fit
  may also start OpenMP or BLAS threads.
- Reduced native path memory: glasso adapters derive graph support from the
  returned precision matrices instead of retaining a duplicate dense double
  path, and CT converts threshold matrices one lambda at a time.
- TIGER with StARS now fails fast until subsample fits expose a common
  certified lambda-path prefix. TIGER with RIC remains supported.
- `huge_select(..., criterion=None)` now matches the R defaults (RIC for
  MB/TIGER, StARS for CT, and EBIC for graphical lasso) and validates only
  arguments used by the active criterion.
- `huge_generator().sigmahat` now matches R's empirical correlation matrix
  instead of returning a sample covariance matrix; `n=1` is rejected because
  correlation is undefined.
- `huge_roc()` now rejects one-class truth matrices instead of fabricating a
  denominator (which could return false-positive rates above 1).
- Python distributions now include the GPL-2.0-only license text, and builds
  fail if NumPy or pybind11 is unavailable instead of silently producing a
  wheel without the required native extension.
- PyPI publishing now waits for the full R/Python parity suite and a clean
  wheel smoke test. CI also exercises the declared NumPy 1.23/SciPy 1.9
  dependency floor.
- Path sparsity and EBIC edge counts now operate directly on sparse upper
  triangles instead of allocating a dense matrix for every lambda value.
- StARS now consumes subsample paths as they finish and accumulates sparse
  coordinates directly, avoiding storage of every replication and dense
  conversion of each graph. Frequency counts use the smallest safe unsigned
  integer type for `rep_num` (normally one byte instead of eight) and convert
  one lambda layer at a time for the unchanged variability calculation.
  CT and MB store only the condensed upper triangle; glasso keeps both
  directions for compatibility with previously created or caller-constructed
  paths. StARS rejects fewer than two variables, where variability is
  undefined. Subsample sizes now use the requested ratio's true floor and
  fail clearly when that selects fewer than two observations, instead of
  silently increasing the subsample to two.
- Numeric scalar validation no longer truncates fractional counts, accepts
  booleans/strings, or leaks raw conversion exceptions. Explicit lambda paths
  ignore inactive grid arguments, CT accepts zero thresholds, and generator
  default probabilities are capped at 1 in small dimensions.
- ROC, inference, and plotting now reject non-finite secondary matrix inputs
  and normalize non-numeric conversion failures to `PyHugeError`.
- Inference now rejects fewer than two observations, constant columns,
  non-positive precision diagonals, and numerically non-finite edge p-values.
  Gaussian inference supports the valid one-variable boundary with the same
  finite result as R; Nonparanormal inference requires at least two variables.
- Gaussian inference now ignores the nonparanormal-only `method` argument;
  plotting validates output filename and directory arguments only when EPS
  output is requested.
- The extension now builds with OpenMP when available (libgomp on Linux,
  Homebrew libomp on macOS; set `PYHUGE_NO_OPENMP=1` to opt out), enabling
  the shared core's per-column parallelism that the R package already had.
- Added `tests/test_core_mirrors.py` and CI checks guarding that the C++ core
  stays byte-identical with the R package copy.
- Aligned with R package huge 2.0.0.


## 0.8.0

- Bumped package version to 0.8.0, aligned with R package huge 1.5.

## 0.3.3

- Bumped package version to 0.3.3.
- Removed hard dependency and fallback path to `scikit-learn` for `mb`, `tiger`, and `glasso`.
- Enforced native C++ core (`pyhuge._native_core`) for `mb`, `tiger`, and `glasso`.
- Updated docs/tests to reflect native-core runtime expectations.


## 0.3.0

- Introduced native Python implementation (`pyhuge` 0.3 line).
- Added optional C++ acceleration module (`pyhuge._native_core`).
- Added packaged dataset loader: `huge_stockdata`.
- Added runtime diagnostics: `test()` compatibility status + `pyhuge-doctor` CLI.
- Added expanded plotting support including network graph view.
- Added docs/man skeleton parity with previous package structure.
- Added release/build helper scripts for wheel/sdist workflows.
