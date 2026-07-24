# Contributing to pyhuge

## Local setup

```bash
cd python-package
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Run docs locally

```bash
mkdocs serve
```

## Validate docs build

```bash
mkdocs build --strict
```

## Build wheel/sdist

```bash
bash scripts/build_dist.sh
```

## Bump version and prepare release

```bash
python scripts/bump_version.py 2.0.0
bash scripts/release.sh 2.0.0
```

## Code principles

- Keep public API aligned with huge-style semantics.
- Preserve backward-compatible dataclass fields when possible.
- Prefer sparse graph outputs (`scipy.sparse`) for path objects.
- Add tests for every externally visible behavior change.
