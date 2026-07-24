# `huge_generator`

## Usage

```python
huge_generator(
    n=200,
    d=50,
    graph="random",
    v=None,
    u=None,
    g=None,
    prob=None,
    vis=False,
    verbose=True,
    random_state=None,
) -> HugeGeneratorResult
```

## Description

Native synthetic data generator for huge-style graph models.

## Key arguments

- `graph`: `"random"`, `"hub"`, `"cluster"`, `"band"`, `"scale-free"`
- `g`: group/bandwidth style parameter depending on graph type
- `prob`: finite edge probability for random/cluster settings; automatic
  probabilities are capped at 1 for small dimensions

## Returns

`HugeGeneratorResult` with `data`, `sigma`, `omega`, `sigmahat`, `theta`,
`sparsity`, and `graph_type`. `sigmahat` is the empirical correlation matrix,
matching the R package. At least two observations are required.
