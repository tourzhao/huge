# Installation

## Recommended

```bash
pip install "pyhuge[runtime]"
```

Published Linux x86_64 wheels are self-contained: OpenBLAS and the OpenMP
runtime are bundled, so no system BLAS package is needed.

Source builds require a C++17 compiler and OpenBLAS development files on
Linux. Install them first, for example:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential libopenblas-dev

# Fedora/RHEL
sudo dnf install gcc-c++ openblas-devel
```

Then install from a checkout:

```bash
git clone https://github.com/Gatech-Flash/huge.git
cd huge/python-package
pip install -e ".[runtime]"
```

Native source builds currently support Linux and macOS. Windows source builds
are not yet supported; use a Linux environment or container.

## Optional extras

```bash
pip install -e ".[viz]"   # matplotlib + networkx
pip install -e ".[test]"  # pytest
pip install -e ".[docs]"  # mkdocs
pip install -e ".[dev]"   # common contributor setup
```

## Verify install

```bash
python -c "import pyhuge; print(pyhuge.test())"
pyhuge-doctor
```

`runtime=True` means core dependencies are available.

## OpenMP (multicore solvers)

The C++ core parallelizes its per-column solvers (mb, tiger, and parts of
glasso) with OpenMP. Linux source builds request OpenMP by default, while
macOS source builds enable it when Homebrew `libomp` is found. If the Linux
toolchain lacks OpenMP, set `PYHUGE_NO_OPENMP=1` to build a serial extension.

- **Linux wheel**: no additional runtime package is needed.
- **Linux source build**: use a compiler with OpenMP support; the package also
  needs the OpenBLAS development package shown above.
- **macOS**: install Homebrew's libomp *before* building:

  ```bash
  brew install libomp
  pip install --force-reinstall --no-binary :all: pyhuge
  # or for a source checkout: pip install -e . (after brew install libomp)
  ```

- Opt out with `PYHUGE_NO_OPENMP=1 pip install ...`.

Check whether your build is parallel with `pyhuge-doctor`: the
`native_core.omp_max_threads` field reports the thread count (1 means a
serial build).

## Apple Silicon / architecture notes

`pyhuge` does not depend on R architecture, but Python/native wheels must
match your interpreter architecture. Check:

```bash
python -c 'import platform,sys; print(platform.machine()); print(sys.executable)'
```

## PEP 668 environments (externally-managed)

If `pip install` shows `externally-managed-environment`, use a virtual env:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "pyhuge[runtime]"
```
