<h1 align="center">Huge</h1>

[![](https://cranlogs.r-pkg.org/badges/huge)](https://cran.r-project.org/package=huge)[![](https://cranlogs.r-pkg.org/badges/grand-total/huge)](https://cran.r-project.org/package=huge)

<h4 align="center">General Package for High-Dimensional Undirected Graph Estimation and Inference (R + Python)</h4>

___Huge___ (High-Dimensional Undirected Graph Estimation) is a general project for sparse graphical model estimation and inference in high dimensions. The core algorithms are implemented in plain C++17 with direct BLAS calls and OpenMP parallelism; thin Rcpp and pybind11 adapters expose the same core to R and Python.

This repository provides two package variants:

- R package: `huge` (native R interface, available on CRAN)
- Python package: `pyhuge` (native Python interface with shared C++ core)

Both variants share the same C++ core and target the same modeling pipeline, including graph estimation, model selection, and inferential analysis.

## Package Variants

- **R version (`huge`)**: see sections below for prerequisites and installation.
- **Python version (`pyhuge`)**: see [Python Package (`pyhuge`)](#python-package-pyhuge).

## R Package (`huge`)

### Prerequisites

Huge uses OpenMP to parallelize the core solvers. The package builds and runs without OpenMP (the `configure` script detects availability), but enabling it is recommended for large problems.

For Windows and Linux users, recent GCC supports OpenMP out of the box.

For macOS users, Apple's default toolchain does not ship OpenMP. Install LLVM from Homebrew:

```
brew install llvm
```

Then point R at that compiler by appending the following to `~/.R/Makevars` (Homebrew installs to `/opt/homebrew/opt/llvm` on Apple Silicon and `/usr/local/opt/llvm` on Intel):

```
LLVM_LOC = /opt/homebrew/opt/llvm
CC  = $(LLVM_LOC)/bin/clang
CXX = $(LLVM_LOC)/bin/clang++
CXX17 = $(LLVM_LOC)/bin/clang++
SHLIB_OPENMP_CFLAGS = -fopenmp
SHLIB_OPENMP_CXXFLAGS = -fopenmp
CPPFLAGS += -I$(LLVM_LOC)/include
LDFLAGS += -L$(LLVM_LOC)/lib
```

### Installing from GitHub

First, you need to install the devtools package. You can do this from CRAN. Invoke R and then type

```
install.packages(devtools)
```

Then load the devtools package and install huge

```
library(devtools)
install_github("Gatech-Flash/huge")
library(huge)
```

*Windows User:*  If you encounter a Rtools version issue: 1. make sure you install the latest [Rtools](https://cran.r-project.org/bin/windows/Rtools/); 2. try the following code
```R
assignInNamespace("version_info", c(devtools:::version_info, list("3.5" = list(version_min = "3.3.0", version_max = "99.99.99", path = "bin"))), "devtools")
```

### Install from CRAN

Ideally you can just install and enable huge using with the help of CRAN on an R console.

```
install.packages("huge")
library(huge)
```

## Python Package (`pyhuge`)

This repository includes a native Python package under `python-package/`.
It shares the same C++ core as the R package for portable high performance.

### Python package location

- `python-package/README.md`
- `python-package/docs/`
- `python-package/examples/`

### Python installation

```bash
git clone https://github.com/Gatech-Flash/huge.git
cd huge/python-package
pip install -e .
python -c "import pyhuge; print(pyhuge.test())"
```

Optional extras:

```bash
pip install -e ".[viz]"      # matplotlib + networkx
pip install -e ".[dev]"      # tests + docs + release tooling
```

## Standalone C++ Core

The bare C++17 core requires CMake 3.18 or newer and a 32-bit-integer (LP64)
BLAS implementation such as OpenBLAS or Apple Accelerate:

```bash
cmake -S . -B build -DHUGE_OPENMP=ON
cmake --build build
cmake --install build --prefix /path/to/prefix
```

`HUGE_OPENMP=ON` falls back to a serial build when OpenMP is unavailable.
Run `tools/check_cmake_install.sh` to build, install, and consume both the
static and shared libraries.

### Python documentation website and CI

- Docs site: <https://tourzhao.github.io/huge/>
- Python tests workflow: `.github/workflows/python-wrapper-tests.yml`
- Python docs workflow: `.github/workflows/python-package-docs.yml`
- Python release workflow: `.github/workflows/python-package-release.yml`

### Python API coverage (summary)

- Estimation: `huge`, `huge_mb`, `huge_glasso`, `huge_ct`, `huge_tiger`
- Selection/preprocessing: `huge_select`, `huge_npn`
- Simulation/inference/ROC: `huge_generator`, `huge_inference`, `huge_roc`
- Utility/plots: `huge_summary`, `huge_select_summary`, `huge_plot_*`, `huge_plot_network`

## Examples

```R
#generate data  
L = huge.generator(n = 50, d = 12, graph = "hub", g = 4)

#graph path estimation using glasso  
est = huge(L$data, method = "glasso")
plot(est)

#inference of Gaussian graphical model at 0.05 significance level  
T = est$icov[[10]]  
inf = huge.inference(L$data, T, L$theta)
print(inf$error) # print out type-I error
```

## Experiments
For detailed implementation of the experiments, please refer to `benchmark/benchmark.R`

### Graph Estimation

We compared our package on hub graph with (n=200,d=200) with other packages, namely, QUIC and clime.
Huge significantly outperforms clime, QUIC and original huge in timing performance. We also calculated the likelihood for estimation.

<center>
<table>
  <thead>
    <tr>
      <th></th>
      <th>CPU Times(s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><center>Huge glasso</center></td>
      <td><center>1.12</center></td>
    </tr>
    <tr>
      <td><center>Huge tiger</center></td>
      <td><center>1.88</center></td>
    <tr>
      <td><center>Huge (CRAN 1.2.7)</center></td>
      <td><center>1.80</center></td>
    </tr>
	<tr>
	  <td><center>QUIC</center></td>
      <td><center>7.50</center></td>
	</tr>
	<tr>
	  <td><center>Clime</center></td>
      <td><center>416.77</center></td>
	</tr>
  </tbody>
</table>
</center>

<center>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Object value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><center>Huge glasso</center></td>
      <td><center>-125.96</center></td>
    </tr>
    <tr>
      <td><center>Huge tiger</center></td>
      <td><center>-125.47</center></td>
	<tr>
	  <td><center>QUIC</center></td>
      <td><center>-90.58</center></td>
	</tr>
	<tr>
	  <td><center>Clime</center></td>
      <td><center>-136.96</center></td>
	</tr>
  </tbody>
</table>
</center>

### Graph Inference
When using the Gaussian graphical model, huge controls the type I error well.

<center>
<table>
    <tr>
	  <td></td>
	  <td colspan = "2"><center>band</center></td>
	  <td colspan = "2"><center>hub</center></td>
	  <td colspan = "2"><center>scale-free</center></td>
	<tr>
      <td><center>significance level</center></td>
      <td><center>0.05</center></td>
      <td><center>0.10</center></td>
	  <td><center>0.05</center></td>
      <td><center>0.10</center></td>
      <td><center>0.05</center></td>
      <td><center>0.10</center></td>
    </tr>
    <tr>
      <td><center>type I error</center></td>
	  <td><center>0.0175</center></td>
	  <td><center>0.0391</center></td>
      <td><center>0.0347</center></td>
	  <td><center>0.0669</center></td>
	  <td><center>0.0485</center></td>
      <td><center>0.0854</center></td>
	</tr>
  </tbody>
</table>
</center>

## References
[1] [T. Zhao and H. Liu, The huge Package for High-dimensional Undirected Graph Estimation in R, 2012](https://cran.r-project.org/web/packages/huge/vignettes/vignette.pdf)  
[2] [Xingguo Li, Jason Ge, Haoming Jiang, Mingyi Hong, Mengdi Wang, and Tuo Zhao, Boosting Pathwise Coordinate Optimization: Sequential Screening and Proximal Subsampled Newton Subroutine, 2016](https://www2.isye.gatech.edu/~tzhao80/)  
[3] [Quanquan Gu, Yuan Cao, et al. Local and Global Inference for High Dimensional Nonparanormal Graphical Models](https://arxiv.org/abs/1502.02347)  
[4] [Conﬁdence intervals for high-dimensional inverse covariance estimation](https://projecteuclid.org/download/pdfview_1/euclid.ejs/1433195859)  
[5] D. Witten and J. Friedman, New insights and faster computations for the graphical lasso,2011  
[6] N. Meinshausen and P. Buhlmann, High-dimensional Graphs and Variable Selection with the Lasso, 2006
