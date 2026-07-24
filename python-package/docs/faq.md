# FAQ

## Is `pyhuge` pure Python?

Yes. `pyhuge` is native Python and does not require `rpy2`. The `mb`/`glasso`/`tiger` estimators require the bundled C++ extension (built automatically on install).

## `runtime=False` in `pyhuge.test()` means what?

Usually at least one core dependency is missing:

- `numpy`
- `scipy`
- `pyhuge._native_core` (native extension)

Install with:

```bash
pip install "pyhuge[runtime]"
```

## Which method should I start with?

Use `method="mb"` first. Then compare with `method="glasso"`.

## Difference between `fit.path` and `sel.refit`?

- `fit.path`: full path of estimated graphs
- `sel.refit`: single selected graph under criterion (`ric`, `stars`, `ebic`)

## Which selection criterion should I use?

- `ric`: fast and simple
- `stars`: stability-focused, slower; currently for MB, CT, and glasso
- `ebic`: common for glasso

Use `ric` for TIGER. StARS is rejected for TIGER until all subsample fits can
report a common certified lambda-path prefix.

## Why is plotting failing?

Install visualization deps:

```bash
pip install "pyhuge[viz]"
```

In headless environments:

```bash
export MPLBACKEND=Agg
```

## Can input be covariance/correlation matrix?

Yes. For `ct`, `glasso`, and `tiger`, `huge(...)` accepts square
covariance/correlation input. For `mb`, use a raw data matrix (`n x d`).
The compatible default `input_type="auto"` detects covariance input by
symmetry. If observations happen to form a square symmetric matrix, pass
`input_type="data"` explicitly; use `"covariance"` to require covariance
routing and validation.

## Is there a built-in dataset?

Yes:

```python
from pyhuge import huge_stockdata
stock = huge_stockdata()
print(stock.data.shape, stock.info.shape)
```

## Where are full function docs?

- API overview: [api.md](api.md)
- One-page function manual: [man/index.md](man/index.md)
