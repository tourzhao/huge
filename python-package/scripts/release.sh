#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <new-version>"
  echo "Example: $0 2.0.0"
  exit 2
fi

VERSION="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/bump_version.py "$VERSION"
bash scripts/build_dist.sh

echo
echo "[next] Review and publish:"
echo "  git add pyproject.toml pyhuge/__init__.py CHANGELOG.md"
echo "  git commit -m 'pyhuge: release $VERSION'"
echo "  git tag pyhuge-v$VERSION"
echo "  git push origin <branch> --tags"
echo
echo "Artifacts are ready under: $ROOT/dist"
