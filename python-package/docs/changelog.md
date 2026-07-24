# Changelog

Version history for pyhuge is maintained in:

- [`python-package/CHANGELOG.md`](https://github.com/Gatech-Flash/huge/blob/master/python-package/CHANGELOG.md)

Current release line:

- `2.0.0`: correctness fixes (native correlation-domain TIGER with KKT
  certification, tiger `icov` symmetry, RIC scale-invariance,
  `huge_generator` correlation-matrix model matching R), shared-core speedups,
  compressed MB/TIGER support paths, bounded RIC scratch workers, parallel
  stars via `n_jobs`, OpenMP builds, installed-sdist validation, and CPython
  3.9--3.14 native binding gates. TIGER+StARS now fails fast until subsample
  fits expose a common certified lambda-path prefix.
- `0.8.x`: shared C++ core with the R package; BLAS acceleration.
- `0.3.3`: native-core alignment release (removed sklearn runtime dependency for `mb`/`tiger`/`glasso`).
