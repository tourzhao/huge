# `huge`

## Usage

```python
huge(
    x,
    lambda_=None,
    nlambda=None,
    lambda_min_ratio=None,
    method="mb",
    scr=None,
    scr_num=None,
    cov_output=False,
    sym="or",
    verbose=True,
    backend="native",
    *,
    input_type="auto",
) -> HugeResult
```

## Description

Native graph path estimation entry.

## Key arguments

- `x`: 2D numeric matrix (`n_samples x n_features`) or a square covariance
  input with positive diagonal and valid pairwise correlations. CT and TIGER
  require positive semidefiniteness. Glasso may accept an indefinite pairwise
  estimate only when regularization produces a certified positive-definite
  result
- `input_type`: `"auto"` keeps symmetry-based detection; `"data"` forces an
  observation matrix; `"covariance"` requires a square covariance/correlation
  matrix. Set `"data"` for square symmetric observations. For glasso with no
  explicit lambda, `"auto"` matches R's historical diagonal-sensitive default
  scale, while `"covariance"` uses only off-diagonal entries
- `method`: one of `"mb"`, `"glasso"`, `"ct"`, `"tiger"`
- `lambda_`: optional numeric scalar, NumPy 0-D value, or non-empty
  one-dimensional sequence; multidimensional inputs are rejected. MB, glasso,
  and TIGER require finite, positive, non-increasing values and permit ties.
  CT requires finite, non-negative values, permits zero, and applies thresholds
  in supplied order
- `nlambda`: positive-integer path length when `lambda_` is not provided
- `lambda_min_ratio`: numeric minimum lambda ratio in `(0, 1]`; like
  `nlambda`, it is ignored when an explicit `lambda_` is supplied
- `scr`: enables lossy correlation-neighborhood screening for MB or safe
  coordinate screening for glasso
- `scr_num`: MB-only number of candidate neighbors retained per node; it must
  satisfy `1 <= scr_num < d` and requires `scr=True`. With `scr=True` and no
  value, MB uses `n - 1` when `n < d`; when `n >= d`, screening is skipped
- `cov_output`: only valid for `method="glasso"`
- `sym`: only applicable to `mb` and `tiger`
- `backend`: currently only `"native"`

Python MB currently requires `input_type="data"`; CT, glasso, and TIGER also
accept covariance/correlation input.

## Returns

`HugeResult` with `method`, `lambda_path`, `sparsity`, `path`, `data`, and
optional `icov/cov/df/loglik` depending on method.
