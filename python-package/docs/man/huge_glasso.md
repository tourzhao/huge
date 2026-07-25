# `huge_glasso`

## Usage

```python
huge_glasso(
    x,
    lambda_=None,
    nlambda=None,
    lambda_min_ratio=None,
    scr=None,
    cov_output=False,
    verbose=True,
    backend="native",
    *,
    input_type="auto",
) -> HugeResult
```

## Description

Convenience wrapper for `huge(..., method="glasso")`.
Use `input_type="data"` for square symmetric observations or
`input_type="covariance"` to require covariance/correlation input.
With no explicit `lambda_`, auto-detected covariance input matches R's
historical diagonal-sensitive lambda scale; explicit covariance routing uses
only off-diagonal entries. An indefinite pairwise covariance estimate is
accepted only when regularization produces a finite positive-definite
precision estimate and a certified covariance/precision pair.

An explicit `lambda_` must be a numeric scalar, NumPy 0-D value, or non-empty
one-dimensional sequence; multidimensional inputs are rejected. Values must be
finite, strictly positive, and non-increasing; tied values are allowed.

## Notes

- `cov_output=True` is supported for glasso only.
