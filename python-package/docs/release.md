# Release Process

## 1. Bump version

From `python-package`:

```bash
python scripts/bump_version.py 2.0.0
```

This updates:

- `pyproject.toml`
- `pyhuge/__init__.py`
- `CHANGELOG.md` (adds heading if missing)

## 2. Build and validate wheel/sdist

```bash
bash scripts/build_dist.sh
```

This runs:

- `python -m build`
- `python -m twine check dist/*`

## 3. Prepare git tag

```bash
git add pyproject.toml pyhuge/__init__.py CHANGELOG.md
git commit -m "pyhuge: release 2.0.0"
git tag pyhuge-v2.0.0
git push origin <branch> --tags
```

## 4. Publish via CI

Publishing workflow tag pattern:

- `pyhuge-v*`

Recommended dedicated workflow file for this package directory:

- `.github/workflows/python-package-release.yml`

## 5. Docs website

Docs source:

- `python-package/mkdocs.yml`

Recommended dedicated workflow file:

- `.github/workflows/python-package-docs.yml`
