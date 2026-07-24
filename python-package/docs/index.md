# pyhuge Documentation

`pyhuge` is a native Python package for high-dimensional undirected graph
estimation and inference workflows inspired by `huge`.

If you are new, follow this order:

1. [Quick Start](quickstart.md)
2. [Installation](installation.md)
3. [Beginner Workflow](beginner-workflow.md)
4. [Troubleshooting](troubleshooting.md)

## Package capabilities

- Estimators: `huge`, `huge_mb`, `huge_glasso`, `huge_ct`, `huge_tiger`
- Selection and preprocessing: `huge_select`, `huge_npn`
- Simulation and inference: `huge_generator`, `huge_roc`, `huge_inference`
- Plot helpers: `huge_plot_sparsity`, `huge_plot_roc`,
  `huge_plot_graph_matrix`, `huge_plot_network`, `huge_plot`
- Dataset helper: `huge_stockdata`
- Environment probe: `pyhuge.test(require_runtime=False)`, `pyhuge-doctor`

## Runtime model

`pyhuge` is native Python:

- Core logic in Python + NumPy/SciPy
- Native C++ kernels (`pyhuge._native_core`) for `mb`/`glasso`/`tiger`
- No R runtime dependency

## Documentation map

- Concept pages: [Quick Start](quickstart.md), [FAQ](faq.md)
- Practical pages: [Quick Start](quickstart.md), [Tutorials](tutorials.md), [Troubleshooting](troubleshooting.md)
- Reference pages: [API Reference](api.md), [Function Manual](man/index.md), [Design](design.md), [Performance](performance.md)
