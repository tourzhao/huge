# `huge_ct`

## Usage

```python
huge_ct(
    x,
    lambda_=None,
    nlambda=None,
    lambda_min_ratio=None,
    verbose=True,
    backend="native",
    *,
    input_type="auto",
) -> HugeResult
```

## Description

Convenience wrapper for `huge(..., method="ct")`.
Use `input_type="data"` for square symmetric observations or
`input_type="covariance"` to require covariance/correlation input.

An explicit `lambda_` must be a numeric scalar, NumPy 0-D value, or non-empty
one-dimensional sequence; multidimensional inputs are rejected. Values must be
finite and non-negative. Zero and tied values are allowed, and thresholds are
applied in supplied order. StARS selection separately requires a
non-increasing path.
