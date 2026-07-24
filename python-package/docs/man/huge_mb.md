# `huge_mb`

## Usage

```python
huge_mb(
    x,
    lambda_=None,
    nlambda=None,
    lambda_min_ratio=None,
    scr=None,
    scr_num=None,
    sym="or",
    verbose=True,
    backend="native",
    *,
    input_type="auto",
) -> HugeResult
```

## Description

Convenience wrapper for `huge(..., method="mb")`.
Set `input_type="data"` to force square symmetric observations to remain raw
data. Python MB does not accept covariance input.

An explicit `lambda_` must be a numeric scalar, NumPy 0-D value, or non-empty
one-dimensional sequence; multidimensional inputs are rejected. Values must be
finite, strictly positive, and non-increasing; tied values are allowed.

With `scr=True`, `scr_num` is the number of candidate neighbors retained per
node and must satisfy `1 <= scr_num < d`. If omitted, it defaults to `n - 1`
when `n < d`; screening is skipped when `n >= d`.

## Returns

`HugeResult`.
