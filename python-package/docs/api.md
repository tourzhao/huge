# API Reference

## `pyhuge.huge`

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

Native entry for graph-path estimation.
`input_type="auto"` preserves symmetry-based detection. Use `"data"` for a
square symmetric observation matrix, or `"covariance"` to require a square
covariance/correlation matrix.

## `pyhuge.test`

```python
test(require_runtime=False) -> dict
```

Environment probe for native runtime.

Returned keys include:

- `python_import`
- `numpy`
- `scipy`
- `sklearn` (compatibility field; not required for runtime)
- `native_extension`
- `runtime`
- `rpy2` (compatibility field)

## Wrapper shortcuts

```python
huge_mb(x, lambda_=None, nlambda=None, lambda_min_ratio=None,
        scr=None, scr_num=None, sym="or", verbose=True, backend="native",
        *, input_type="auto")
huge_glasso(x, lambda_=None, nlambda=None, lambda_min_ratio=None,
            scr=None, cov_output=False, verbose=True, backend="native",
            *, input_type="auto")
huge_ct(x, lambda_=None, nlambda=None, lambda_min_ratio=None,
        verbose=True, backend="native", *, input_type="auto")
huge_tiger(x, lambda_=None, nlambda=None, lambda_min_ratio=None,
           sym="or", verbose=True, backend="native", *,
           input_type="auto")
```

These call `huge(...)` with the method fixed; arguments match the
corresponding subset of `huge()`.

## `pyhuge.huge_select`

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

Model selection on `HugeResult`. With `criterion="stars"`, `n_jobs > 1` fits
the subsamplings in a thread pool (the native solvers release the GIL);
results are identical to the serial path. Mirrors `num.cores` in the R
package. Each fit may also start OpenMP or BLAS threads; use `n_jobs=1` when
a bounded thread budget matters. TIGER currently supports RIC selection, not StARS: subsample TIGER
fits can certify different path prefixes, and no common-prefix protocol is
exposed yet. When `criterion=None`, defaults match R: RIC for MB/TIGER,
StARS for CT, and EBIC for graphical lasso. Parameters are validated only
when their criterion uses them. StARS requires a non-increasing
`est.lambda_path`; tied values are allowed.

## `pyhuge.huge_npn`

```python
huge_npn(x, npn_func="shrinkage", verbose=True) -> numpy.ndarray
```

Native nonparanormal transformation.

## `pyhuge.huge_generator`

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

Native synthetic data generator.

## `pyhuge.huge_inference`

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

Native edge-wise inference approximation. Data must have at least two rows and
no constant columns; Nonparanormal inference also requires at least two
variables. The precision-like matrix must have a finite, positive diagonal.

## `pyhuge.huge_roc`

```python
huge_roc(path, theta, verbose=True, plot=False) -> HugeRocResult
```

Native ROC metrics over graph path. The truth matrix must contain at least
one edge and one absent off-diagonal edge; otherwise ROC/AUC is undefined.

## `pyhuge.huge_stockdata`

```python
huge_stockdata() -> HugeStockDataResult
```

Loads packaged stock dataset (`1258 x 452` matrix + `452 x 3` info table).

## Summaries

```python
huge_summary(fit: HugeResult) -> HugeSummary
huge_select_summary(sel: HugeSelectResult) -> HugeSelectSummary
```

## Plot helpers

```python
huge_plot_sparsity(fit, ax=None, show_points=True)
huge_plot_roc(roc, ax=None)
huge_plot_graph_matrix(fit, index=-1, ax=None)
huge_plot_network(fit, index=-1, ax=None, layout="spring",
                  with_labels=False, node_size=120.0,
                  node_color="#c44e52", edge_color="#4d4d4d",
                  min_abs_weight=0.0)
huge_plot(g, epsflag=False, graph_name="default", cur_num=1, location=None)
```

## Dataclasses

- `HugeResult`
- `HugeSelectResult`
- `HugeGeneratorResult`
- `HugeInferenceResult`
- `HugeRocResult`
- `HugeStockDataResult`
- `HugeSummary`
- `HugeSelectSummary`

## Exception

- `PyHugeError`: raised for validation failures or missing native dependencies.
