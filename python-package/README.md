# pyhuge Python Package

`pyhuge` is the native Python package for high-dimensional undirected graph
estimation and inference, sharing the same C++ core as the R `huge` package.

## Table of contents

- [Background](#background)
- [Directory structure](#directory-structure)
- [What this package provides](#what-this-package-provides)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Developer workflow](#developer-workflow)
- [Citation](#citation)

## Background

`pyhuge` runs natively in Python with a shared C++ backend and does not
require an R runtime.

## Directory structure

- `pyhuge/`: package source code
- `pyhuge/data/`: packaged datasets (`stockdata.npz`)
- `cpp/`: optional pybind11 acceleration kernels
- `tests/`: unit/e2e/parity tests
- `examples/`: runnable scripts
- `docs/`: MkDocs documentation pages
- `scripts/`: release and docs helper scripts

## What this package provides

- Core estimators: `huge`, `huge_mb`, `huge_glasso`, `huge_ct`, `huge_tiger`
- Model selection: `huge_select` (`ric`, `stars`, `ebic`)
- Data transforms and utilities: `huge_npn`, `huge_generator`, `huge_roc`, `huge_inference`
- Plotting helpers: `huge_plot_sparsity`, `huge_plot_roc`, `huge_plot_graph_matrix`, `huge_plot_network`, `huge_plot`
- Dataset helper: `huge_stockdata`
- Diagnostics: `pyhuge.test()`, `pyhuge-doctor`

## Requirements

- Python `>=3.9`
- Runtime packages: `numpy`, `scipy`
- Native extension: `pyhuge._native_core` is required for `mb`, `tiger`, `glasso`
- Native source builds: Linux or macOS; Windows is not currently supported

Optional:

- plotting: `matplotlib`, `networkx`
- docs: `mkdocs`, `mkdocs-material`

## Installation

Published Linux x86_64 wheels bundle OpenBLAS and the OpenMP runtime:

```bash
pip install "pyhuge[runtime]"
```

Linux source builds require a C++17 compiler and OpenBLAS development files
(`libopenblas-dev` on Ubuntu/Debian or `openblas-devel` on Fedora/RHEL).

From source:

```bash
cd python-package
pip install -e ".[runtime]"
```

Optional extras:

```bash
pip install -e ".[viz]"
pip install -e ".[test]"
pip install -e ".[docs]"
pip install -e ".[dev]"
```

Runtime check:

```bash
python -c "import pyhuge; print(pyhuge.test())"
pyhuge-doctor
```

## Usage

```python
import numpy as np
from pyhuge import huge, huge_select

rng = np.random.default_rng(1)
x = rng.normal(size=(120, 30))

fit = huge(x, method="mb", nlambda=8, verbose=False)
sel = huge_select(fit, criterion="ric", verbose=False)

print(fit.method, len(fit.path), sel.opt_lambda, sel.opt_sparsity)
```

`input_type="auto"` preserves covariance detection by symmetry. For an
observation matrix that is itself square and symmetric, set
`input_type="data"` explicitly; use `"covariance"` to require covariance or
correlation input.

Network visualization:

```python
import matplotlib.pyplot as plt
from pyhuge import huge_plot_network

fig, ax = plt.subplots(figsize=(5, 5))
huge_plot_network(fit, index=-1, ax=ax, layout="spring")
plt.show()
```

## Documentation

- Docs source: `python-package/docs`
- Function manual pages: `python-package/docs/man`

Build locally:

```bash
cd python-package
mkdocs build --strict
```

## Developer workflow

```bash
cd python-package
pytest
bash scripts/build_dist.sh
python scripts/bump_version.py 2.0.0
bash scripts/release.sh 2.0.0
```

## Citation

If you use `huge`/`pyhuge` in research, cite:

```bibtex
@article{zhao2012huge,
  title   = {The huge Package for High-dimensional Undirected Graph Estimation in R},
  author  = {Zhao, Tuo and Liu, Han and Roeder, Kathryn and Lafferty, John and Wasserman, Larry},
  journal = {Journal of Machine Learning Research},
  volume  = {13},
  pages   = {1059--1062},
  year    = {2012}
}
```
