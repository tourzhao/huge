# `huge_inference`

## Usage

```python
huge_inference(
    data,
    t,
    adj,
    alpha=0.05,
    type_="Gaussian",
    method="score",
) -> HugeInferenceResult
```

## Description

Native edge-wise inference helper using a partial-correlation z-test
approximation.

## Key arguments

- `data`: sample matrix (`n x d`) with `n >= 2` and no constant columns;
  Nonparanormal inference also requires `d >= 2`
- `t`: finite precision-like matrix (`d x d`) with a positive diagonal
- `adj`: reference adjacency (`d x d`)
- `type_`: `"Gaussian"` or `"Nonparanormal"`
- `method`: `"score"` or `"wald"` for Nonparanormal inference; ignored for
  Gaussian inference

## Returns

`HugeInferenceResult` with transformed `data`, p-value matrix `p`, and `error`.
Numerically degenerate inputs that produce non-finite edge p-values raise
`PyHugeError` instead of returning an apparently successful result.
