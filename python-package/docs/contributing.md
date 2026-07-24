# Contributing

## Local setup

```bash
cd python-package
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Run parity checks against R huge (optional)

Requires local R with package `huge` installed.

```bash
cd python-package
python scripts/r_parity_report.py --out parity_report.json
```

## Build docs

```bash
mkdocs build --strict
```

## Build release artifacts

```bash
bash scripts/build_dist.sh
```

## Bump version

```bash
python scripts/bump_version.py 2.0.0
bash scripts/release.sh 2.0.0
```

## Code principles

- Keep public API aligned with huge-style semantics.
- Keep dataclass fields backward-stable where possible.
- Add tests for every new public behavior.
- Document behavior changes in `CHANGELOG.md`.
