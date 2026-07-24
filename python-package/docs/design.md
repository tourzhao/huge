# Design and Roadmap

## Current architecture

- Native Python core (`pyhuge/core.py`)
- NumPy/SciPy data model
- Native C++ kernels for MB/glasso/tiger (`pyhuge._native_core`)

## Data flow

1. User inputs NumPy/SciPy matrices
2. Python validation and preprocessing
3. Method-specific native solver
4. Results converted to typed dataclasses
5. Plot and summary helpers consume same dataclasses

## Compatibility goals

- Keep public API names aligned with earlier `pyhuge` wrapper and R `huge`
- Keep result fields stable where possible (`lambda_path`, `path`, `opt_lambda`, etc.)
- Preserve one-page-per-function manual docs under `docs/man/`

## Known approximation boundaries

- Native and R implementations may still differ numerically on some datasets,
  though core algorithm families are aligned.
- Selection/inference implementations target practical parity, not strict
  numerical identity with R code paths.

## Roadmap

1. Strengthen numerical parity tests against R outputs (function-by-function)
2. Add optional compiled kernels for more hotspots
3. Extend solver backends for tighter TIGER parity
4. Keep Python and R user docs aligned on semantics and examples
