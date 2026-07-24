#!/bin/sh
# Negative probes for tools/check_core_mirrors.sh — verify the guard itself
# fails when it should (a guard that cannot fail protects nothing).
# Modeled on picasso's CheckMirrorNegativeProbes.cmake.
#
# Usage: tools/check_core_mirrors_probes.sh   (from the repo root)

root="${1:-.}"
checker="$root/tools/check_core_mirrors.sh"
work="$(mktemp -d "${TMPDIR:-/tmp}/mirror-probe-XXXXXX")"
trap 'rm -rf "$work"' EXIT

make_fixture() {
  rm -rf "$work/fix"
  mkdir -p "$work/fix/src/huge" \
           "$work/fix/python-package/cpp/include/huge" \
           "$work/fix/tools"
  cp "$root/src/huge_core.cpp"      "$work/fix/src/"
  cp "$root/src/huge/huge_core.h"   "$work/fix/src/huge/"
  cp "$root/src/huge/blas_config.h" "$work/fix/src/huge/"
  cp "$root/src/huge_core.cpp"      "$work/fix/python-package/cpp/"
  cp "$root/src/huge/huge_core.h"   "$work/fix/python-package/cpp/include/huge/"
  cp "$root/src/huge/blas_config.h" "$work/fix/python-package/cpp/include/huge/"
}

fail=0
expect() {
  # expect <PASS|FAIL> <label>
  if sh "$checker" "$work/fix" >/dev/null 2>&1; then got=PASS; else got=FAIL; fi
  if [ "$got" != "$1" ]; then
    echo "PROBE FAILED: $2 — expected checker to $1 but it did $got"
    fail=1
  else
    echo "probe ok: $2 ($1)"
  fi
}

# 1. Pristine fixture must pass
make_fixture
expect PASS "pristine mirror"

# 2. Single divergent byte must fail
make_fixture
printf '\n// drift\n' >> "$work/fix/python-package/cpp/huge_core.cpp"
expect FAIL "divergent core copy"

# 3. Missing python-side file must fail
make_fixture
rm "$work/fix/python-package/cpp/include/huge/blas_config.h"
expect FAIL "missing mirror file"

# 4. Missing R-side file must fail
make_fixture
rm "$work/fix/src/huge/huge_core.h"
expect FAIL "missing source file"

if [ "$fail" -ne 0 ]; then
  echo "Mirror negative probes FAILED — the checker is not trustworthy."
  exit 1
fi
echo "All mirror negative probes passed (4 scenarios)."
