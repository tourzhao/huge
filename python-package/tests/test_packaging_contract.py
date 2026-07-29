"""Release-artifact contracts that should fail before upload."""

from __future__ import annotations

import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
EXPECTED_LINUX_BUILDS = (
    "cp39-manylinux_x86_64",
    "cp310-manylinux_x86_64",
    "cp311-manylinux_x86_64",
    "cp312-manylinux_x86_64",
    "cp313-manylinux_x86_64",
    "cp314-manylinux_x86_64",
)


def _python_project_version() -> str:
    source = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version = "([0-9]+\.[0-9]+\.[0-9]+)"$', source)
    assert match is not None
    return match.group(1)


def _release_workflow() -> str:
    return (
        REPOSITORY_ROOT / ".github/workflows/python-package-release.yml"
    ).read_text(encoding="utf-8")


def _release_job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n.*?(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )
    assert match is not None, f"missing release job: {name}"
    return match.group(0)


def test_native_extension_build_cannot_silently_downgrade():
    setup_source = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")

    assert "except ImportError" not in setup_source
    assert "ext_modules = []" not in setup_source
    assert "Extension(" in setup_source
    assert '"pyhuge._native_core"' in setup_source


def test_source_build_fails_clearly_on_unsupported_platforms():
    setup_source = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'sys.platform != "darwin"' in setup_source
    assert 'sys.platform.startswith("linux")' in setup_source
    assert (
        "native source builds currently support only Linux and macOS"
        in setup_source
    )
    assert '"Operating System :: MacOS :: MacOS X"' in pyproject
    assert '"Operating System :: POSIX :: Linux"' in pyproject


def test_release_documentation_uses_concrete_consistent_versions():
    project_version = _python_project_version()
    for relative_path in (
        "README.md",
        "CONTRIBUTING.md",
        "docs/contributing.md",
        "docs/release.md",
    ):
        source = (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")
        versions = re.findall(
            r"(?:bump_version\.py|release\.sh) ([0-9]+\.[0-9]+\.[0-9]+)",
            source,
        )
        assert versions, relative_path
        assert set(versions) == {project_version}, (relative_path, versions)
        assert "0.8.x" not in source

    release_script = (PACKAGE_ROOT / "scripts" / "release.sh").read_text(
        encoding="utf-8"
    )
    assert f'echo "Example: $0 {project_version}"' in release_script


def test_r_python_and_standalone_release_versions_are_aligned():
    expected = "2.0.1"
    escaped = re.escape(expected)
    files_and_patterns = (
        (PACKAGE_ROOT / "pyproject.toml", rf'(?m)^version = "{escaped}"$'),
        (
            PACKAGE_ROOT / "pyhuge" / "__init__.py",
            rf'(?m)^__version__ = "{escaped}"$',
        ),
        (PACKAGE_ROOT / "CHANGELOG.md", rf"(?m)^## {escaped}$"),
        (REPOSITORY_ROOT / "DESCRIPTION", rf"(?m)^Version: {escaped}$"),
        (
            REPOSITORY_ROOT / "configure.ac",
            rf"AC_INIT\(\[huge\],{escaped}\)",
        ),
        (
            REPOSITORY_ROOT / "configure",
            rf"PACKAGE_VERSION='{escaped}'",
        ),
        (
            REPOSITORY_ROOT / "CMakeLists.txt",
            rf"project\(huge VERSION {escaped} LANGUAGES CXX\)",
        ),
        (REPOSITORY_ROOT / "NEWS.md", rf"(?m)^# huge {escaped}$"),
        (
            REPOSITORY_ROOT / "tools" / "cmake-consumer" / "CMakeLists.txt",
            rf"find_package\(huge {escaped} CONFIG REQUIRED\)",
        ),
    )

    assert _python_project_version() == expected
    for path, pattern in files_and_patterns:
        source = path.read_text(encoding="utf-8")
        assert re.search(pattern, source), path


def test_gpl_license_is_part_of_python_distributions():
    license_text = (PACKAGE_ROOT / "LICENSE").read_text(encoding="utf-8")
    manifest = (PACKAGE_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 2, June 1991" in license_text
    assert "include LICENSE" in manifest
    assert 'license-files = ["LICENSE"]' in pyproject


def test_release_workflow_is_gated_by_full_tests_and_wheel_smoke():
    workflow = _release_workflow()
    verify_job = _release_job(workflow, "verify")
    sdist_job = _release_job(workflow, "sdist")
    wheels_job = _release_job(workflow, "wheels")
    smoke_job = _release_job(workflow, "wheel-smoke")
    publish_job = _release_job(workflow, "publish")

    assert "Run full Python suite against the current R package" in verify_job
    assert "needs: verify" in sdist_job
    assert "Install and smoke-test source distribution" in sdist_job
    assert "needs: verify" in wheels_job
    assert "needs: wheels" in smoke_job
    assert "Smoke-test CPython 3.11 wheel without system OpenBLAS" in smoke_job
    assert "needs: [sdist, wheel-smoke]" in publish_job
    assert "startsWith(github.ref, 'refs/tags/pyhuge-v')" in publish_job


def test_release_builds_and_tests_the_pinned_manylinux_wheel_matrix():
    workflow = _release_workflow()
    wheels_job = _release_job(workflow, "wheels")
    sdist_job = _release_job(workflow, "sdist")
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"cibuildwheel==4.1.0"' in wheels_job
    assert '"cibuildwheel==4.1.0"' in pyproject
    assert "python -m cibuildwheel --output-dir wheelhouse ." in wheels_job
    assert "python -m build" not in wheels_job
    assert "libopenblas-dev" not in wheels_job

    assert "python -m build --sdist --outdir dist" in sdist_job
    assert "cibuildwheel" not in sdist_job
    assert "--wheel" not in sdist_job
    assert "pyhuge-sdist" in sdist_job

    for identifier in EXPECTED_LINUX_BUILDS:
        assert identifier in pyproject
    assert 'archs = ["x86_64"]' in pyproject
    assert 'manylinux-x86_64-image = "manylinux_2_28"' in pyproject
    assert 'before-all = "dnf install -y openblas-devel"' in pyproject
    assert (
        'repair-wheel-command = "auditwheel repair -w {dest_dir} {wheel}"'
        in pyproject
    )
    assert '"tests/test_native_smoke.py"' in pyproject
    assert '"tests/test_native_symbols.py"' in pyproject
    assert (
        'test-command = "pytest -q tests/test_native_smoke.py '
        'tests/test_native_symbols.py"'
    ) in pyproject


def test_sdist_is_installed_and_smoked_outside_the_checkout():
    sdist_job = _release_job(_release_workflow(), "sdist")

    build = sdist_job.index("Build source distribution")
    smoke = sdist_job.index("Install and smoke-test source distribution")
    upload = sdist_job.index("Upload source distribution")
    assert build < smoke < upload

    assert "python -m venv" in sdist_job
    assert '--no-cache-dir "${sdists[0]}"' in sdist_job
    assert 'cd "${RUNNER_TEMP}"' in sdist_job
    assert '"${SDIST_VENV}/bin/python" -I' in sdist_job
    assert "import pyhuge._native_core as native" in sdist_job
    assert "pyhuge.test(require_runtime=True)" in sdist_job
    assert "pyhuge.huge_mb" in sdist_job
    assert "pyhuge.huge_tiger" in sdist_job
    assert "pip install -e" not in sdist_job


def test_release_rejects_generic_linux_wheels_and_checks_every_artifact():
    wheels_job = _release_job(_release_workflow(), "wheels")

    expected_literal = (
        'expected = {"cp39", "cp310", "cp311", "cp312", "cp313", "cp314"}'
    )
    assert expected_literal in wheels_job
    assert 'assert not name.endswith("-linux_x86_64.whl")' in wheels_job
    assert 'assert "manylinux_2_28_x86_64" in name' in wheels_job
    assert "for wheel in wheels:" in wheels_job
    assert "assert match.group(1) == match.group(2)" in wheels_job
    assert 'item.startswith("pyhuge/_native_core")' in wheels_job
    assert '"pyhuge/data/stockdata.npz" in names' in wheels_job
    assert '".dist-info/licenses/LICENSE"' in wheels_job
    assert 'item.startswith("pyhuge.libs/libopenblas")' in wheels_job


def test_release_smoke_uses_clean_cpython_311_container():
    smoke_job = _release_job(_release_workflow(), "wheel-smoke")

    assert "name: pyhuge-wheels" in smoke_job
    assert "docker run --rm --platform linux/amd64" in smoke_job
    assert "python:3.11-slim-bookworm" in smoke_job
    assert ":/dist:ro" in smoke_job
    assert 'libopenblas*.so*' in smoke_job
    assert "pyhuge-*-cp311-cp311-manylinux*.whl" in smoke_job
    assert "--only-binary=:all:" in smoke_job
    assert 'dependencies="$(ldd "${extension}")"' in smoke_job
    assert 'grep -q "not found"' in smoke_job
    assert 'case "$(readlink -f "${openblas}")"' in smoke_job
    assert "*/site-packages/pyhuge.libs/*" in smoke_job
    assert "import numpy as np, pyhuge, pyhuge._native_core" in smoke_job


def test_publish_merges_only_validated_wheels_and_sdist():
    publish_job = _release_job(_release_workflow(), "publish")

    assert "needs: [sdist, wheel-smoke]" in publish_job
    assert "name: pyhuge-sdist" in publish_job
    assert "name: pyhuge-wheels" in publish_job
    assert publish_job.count("path: dist") == 2
    assert "packages-dir: dist" in publish_job


def test_ci_exercises_declared_minimum_numpy_version():
    workflow = (
        REPOSITORY_ROOT / ".github/workflows/python-package-tests.yml"
    ).read_text(encoding="utf-8")

    assert "\n  minimum-dependencies:\n" in workflow
    assert "numpy==1.23.5" in workflow
    assert "scipy==1.9.3" in workflow


def test_unit_ci_covers_every_published_cpython_and_sparse_bindings():
    workflow = (
        REPOSITORY_ROOT / ".github/workflows/python-package-tests.yml"
    ).read_text(encoding="utf-8")

    for version in ("3.9", "3.10", "3.11", "3.12", "3.13", "3.14"):
        assert f'"{version}"' in workflow
    assert "tests/test_native_symbols.py" in workflow
