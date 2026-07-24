#!/bin/sh
# Verify that the shared C++ core copies are byte-identical between the
# R package (src/) and the Python package (python-package/cpp/).
# Modeled on picasso's cmake/CheckMirrors.cmake.
#
# Usage: tools/check_core_mirrors.sh   (from the repo root, or pass root as $1)
# Exit 0 when all pairs match; exit 1 with a list of divergent pairs otherwise.

root="${1:-.}"
fail=0

check_pair() {
  a="$root/$1"
  b="$root/$2"
  if [ ! -f "$a" ] || [ ! -f "$b" ]; then
    echo "MISSING mirror pair: $1 <-> $2"
    fail=1
  elif ! cmp -s "$a" "$b"; then
    echo "DIFFERENT mirror pair: $1 <-> $2"
    fail=1
  fi
}

check_pair "src/huge_core.cpp"       "python-package/cpp/huge_core.cpp"
check_pair "src/huge/huge_core.h"    "python-package/cpp/include/huge/huge_core.h"
check_pair "src/huge/blas_config.h"  "python-package/cpp/include/huge/blas_config.h"

if [ "$fail" -ne 0 ]; then
  echo "Core mirror check FAILED. Copy the edited side over the stale side"
  echo "and rebuild both packages (see CLAUDE.md, 'The shared C++ core')."
  exit 1
fi
echo "Core mirror check passed (3 pairs)."
