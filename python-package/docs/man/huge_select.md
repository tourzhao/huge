# `huge_select`

## Usage

```python
huge_select(
    est,
    criterion=None,
    ebic_gamma=0.5,
    stars_thresh=0.1,
    stars_subsample_ratio=None,
    rep_num=20,
    n_jobs=1,
    verbose=True,
    backend="native",
) -> HugeSelectResult
```

## Description

Native model selection for an estimated graph path.

## Key arguments

- `est`: `HugeResult`
- `criterion`: `"ric"`, `"stars"`, or `"ebic"`. With `None`, the default is
  RIC for MB/TIGER, StARS for CT, and EBIC for graphical lasso. TIGER supports
  `"ric"` but rejects `"stars"` until subsample fits expose a common certified prefix.
  StARS requires `est.lambda_path` to be non-increasing; tied values are allowed
- `ebic_gamma`: finite numeric EBIC tuning parameter
- `stars_thresh`: threshold in `(0, 1]`
- `stars_subsample_ratio`: optional subsample ratio
- `rep_num`: repetition count for stochastic criteria
- `n_jobs`: thread count for fitting stars subsamplings in parallel
  (results identical to serial; mirrors R's `num.cores`). Each native fit may
  also start OpenMP or BLAS threads, so use `n_jobs=1` when a bounded thread
  budget matters
- `backend`: currently only `"native"`

## Returns

`HugeSelectResult` with `opt_lambda`, `opt_sparsity`, `refit`, and optional
fields (`opt_icov`, `opt_cov`, `variability`, `ebic_score`).

## Notes

- `criterion="ebic"` requires a glasso fit.
- `criterion="stars"` is unavailable for a TIGER fit; use `"ric"`.
- StARS rejects increasing or unordered lambda paths before subsampling.
- Criterion-specific arguments are ignored when their criterion is inactive.
- `opt_index` is 1-based for compatibility with prior wrapper behavior.
