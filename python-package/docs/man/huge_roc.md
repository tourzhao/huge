# `huge_roc`

## Usage

```python
huge_roc(path, theta, verbose=True, plot=False) -> HugeRocResult
```

## Description

Native ROC computation across graph path estimates.

## Key arguments

- `path`: sequence of adjacency matrices
- `theta`: ground-truth adjacency matrix containing at least one edge and one
  absent off-diagonal edge
- `plot`: if `True`, draw ROC curve via matplotlib helper

## Returns

`HugeRocResult` with:

- `f1`: F1-score array
- `tp`: true-positive-rate array
- `fp`: false-positive-rate array
- `auc`: area under ROC curve

ROC/AUC is undefined for a one-class truth matrix, which is rejected with a
clear error.
