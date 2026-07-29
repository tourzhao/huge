#!/usr/bin/env python3
"""Bump pyhuge version in project metadata files."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "pyhuge" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"

# Release contracts (tests/test_packaging_contract.py, tests/test_cmake_contract.py)
# require the R package, the Python package, the standalone CMake project, and
# the documented release commands to agree. Keep every one of those in sync
# here; a missed file fails the contract tests rather than shipping a mismatch.
# The R DESCRIPTION and configure(.ac) are updated by the R-side release step.
SHARED_VERSION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("CMakeLists.txt", r"(^project\(huge VERSION )(\d+\.\d+\.\d+)"),
    (
        "tools/cmake-consumer/CMakeLists.txt",
        r"(^find_package\(huge )(\d+\.\d+\.\d+)( CONFIG REQUIRED\))",
    ),
    ("python-package/README.md", r"((?:bump_version\.py|release\.sh) )(\d+\.\d+\.\d+)"),
    ("python-package/CONTRIBUTING.md", r"((?:bump_version\.py|release\.sh) )(\d+\.\d+\.\d+)"),
    ("python-package/docs/contributing.md", r"((?:bump_version\.py|release\.sh) )(\d+\.\d+\.\d+)"),
    ("python-package/docs/release.md", r"((?:bump_version\.py|release\.sh) )(\d+\.\d+\.\d+)"),
    ("python-package/docs/release.md", r"(pyhuge: release )(\d+\.\d+\.\d+)"),
    ("python-package/docs/release.md", r"(pyhuge-v)(\d+\.\d+\.\d+)"),
    ("python-package/scripts/release.sh", r"(Example: \$0 )(\d+\.\d+\.\d+)"),
    (
        "python-package/tests/test_packaging_contract.py",
        r"(^    expected = \")(\d+\.\d+\.\d+)(\")",
    ),
)


def _validate_version(version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Version must follow MAJOR.MINOR.PATCH (e.g. 0.3.1).")


def _replace_regex(path: pathlib.Path, pattern: str, repl: str) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n == 0:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def _ensure_changelog_heading(version: str) -> None:
    text = CHANGELOG.read_text(encoding="utf-8")
    heading = f"## {version}"
    if heading in text:
        return
    lines = text.splitlines()
    insert_at = 1 if lines and lines[0].startswith("# ") else 0
    block = ["", heading, "", "- TBD", ""]
    lines[insert_at:insert_at] = block
    CHANGELOG.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bump pyhuge version.")
    parser.add_argument("version", help="New version (MAJOR.MINOR.PATCH)")
    args = parser.parse_args()

    try:
        _validate_version(args.version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ok_pyproject = _replace_regex(
        PYPROJECT,
        r'(^version\s*=\s*")([^"]+)(")',
        rf'\g<1>{args.version}\g<3>',
    )
    ok_init = _replace_regex(
        INIT_PY,
        r'(^__version__\s*=\s*")([^"]+)(")',
        rf'\g<1>{args.version}\g<3>',
    )
    if not ok_pyproject or not ok_init:
        print("error: failed to update version fields.", file=sys.stderr)
        return 1

    updated = [PYPROJECT, INIT_PY]
    missed = []
    for relative_path, pattern in SHARED_VERSION_PATTERNS:
        path = REPOSITORY_ROOT / relative_path
        if not path.exists():
            missed.append(relative_path)
            continue
        # Group 1 is the literal prefix, group 2 the version; an optional
        # group 3 is a literal suffix that must be preserved.
        replacement = rf"\g<1>{args.version}"
        if re.compile(pattern).groups >= 3:
            replacement += r"\g<3>"
        if _replace_regex(path, pattern, replacement):
            if path not in updated:
                updated.append(path)
        else:
            missed.append(relative_path)

    if missed:
        print(
            "error: no version match in: " + ", ".join(sorted(set(missed))),
            file=sys.stderr,
        )
        return 1

    _ensure_changelog_heading(args.version)
    updated.append(CHANGELOG)
    print(f"Updated version to {args.version}")
    for path in updated:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
