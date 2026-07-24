# Performance Notes

`pyhuge` runtime is dominated by the C++ solver cost; the Python layer adds
little overhead.

## Practical guidance

- Use `ct` for fast threshold-style path baselines.
- Use `mb` or `glasso` when selection quality matters more than raw speed.
- `stars` is slower than `ric`/`ebic` because it resamples repeatedly.
  `n_jobs > 1` runs subsample fits concurrently, but each native fit may
  already use OpenMP and threaded BLAS. Benchmark on the target runtime;
  `n_jobs=1` is usually the safest choice for an OpenMP-enabled build and
  avoids nested oversubscription.
- `ric` selection is fast (BLAS matrix products in the core, ~10x faster than
  0.8.x at d=2000).
- Reuse transformed data from `huge_npn(...)` when running multiple methods.

## Multicore builds

The core parallelizes per-column solvers with OpenMP when the extension is
built with it (automatic when a toolkit is present; on macOS install
Homebrew's `libomp` first — see [Installation](installation.md)). A serial
build is typically 4-6x slower for `mb`/`tiger` on multicore machines.

## The native core

`mb`, `glasso`, and `tiger` require the native extension
(`pyhuge._native_core`); it is not optional for the estimators. Check:

```python
import pyhuge
print(pyhuge.test()["native_extension"])
```

## Benchmark pattern

```python
import time
import numpy as np
from pyhuge import huge

x = np.random.default_rng(0).normal(size=(300, 100))
t0 = time.perf_counter()
fit = huge(x, method="mb", nlambda=10, verbose=False)
print("sec", time.perf_counter() - t0, "path", len(fit.path))
```

## Native vs R parity report

A dedicated script produces reproducible parity metrics against local R `huge`
(when available):

```bash
cd python-package
python scripts/r_parity_report.py --out parity_report.json
```

Current behavior:

- `ct + stars` parity is evaluated by default.
- `glasso + ebic` parity is evaluated via the native C++ backend.

Use the JSON output to track drift after solver or selection changes.
