# `huge_tiger`

## Usage

```python
huge_tiger(
    x,
    lambda_=None,
    nlambda=None,
    lambda_min_ratio=None,
    sym="or",
    verbose=True,
    backend="native",
    *,
    input_type="auto",
) -> HugeResult
```

## Description

Convenience wrapper for `huge(..., method="tiger")`. The native C++ core
centers and normalizes raw observations, builds the correlation matrix,
generates the default lambda path when needed, and solves the
correlation-domain square-root Lasso. An explicit `lambda_` must be a numeric
scalar, NumPy 0-D value, or non-empty one-dimensional sequence;
multidimensional inputs are rejected. Values are preserved and must be finite,
strictly positive, and non-increasing; ties are allowed.
Generated paths contain only the longest prefix that passes the native KKT
and numerical-degeneracy checks; truncation emits a `RuntimeWarning`. A
user-supplied lambda that cannot be certified raises `PyHugeError`.

`input_type="auto"` preserves symmetry-based detection. Use `"data"` for
square symmetric observations and `"covariance"` to require covariance or
correlation input. After routing, covariance validation, correlation
construction, automatic lambda selection, and fitting all occur in C++.
