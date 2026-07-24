"""Contracts for the standalone C++ core build and install interface."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_cmake_exports_real_headers_and_transitive_dependencies():
    source = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    config = (
        REPOSITORY_ROOT / "cmake/hugeConfig.cmake.in"
    ).read_text(encoding="utf-8")

    assert "cmake_minimum_required(VERSION 3.18)" in source
    assert "set(BLA_SIZEOF_INTEGER 4)" in source
    assert "find_package(BLAS REQUIRED)" in source
    assert "target_link_libraries(huge_core PUBLIC BLAS::BLAS)" in source
    assert "${CMAKE_CURRENT_SOURCE_DIR}/src" in source
    assert "DIRECTORY src/huge" in source
    assert "RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}" in source
    assert "if(HUGE_OPENMP AND OpenMP_CXX_FOUND)" in source

    assert "find_dependency(BLAS)" in config
    assert "find_dependency(OpenMP COMPONENTS CXX)" in config
    assert "if(CMAKE_VERSION VERSION_LESS 3.18)" in config
    assert "huge_NOT_FOUND_MESSAGE" in config
    assert 'include("${CMAKE_CURRENT_LIST_DIR}/hugeTargets.cmake")' in config


def test_ci_runs_the_installed_consumer_smoke():
    workflow = (
        REPOSITORY_ROOT / ".github/workflows/r-cmd-check.yml"
    ).read_text(encoding="utf-8")
    smoke = (
        REPOSITORY_ROOT / "tools/check_cmake_install.sh"
    ).read_text(encoding="utf-8")
    consumer = (
        REPOSITORY_ROOT / "tools/cmake-consumer/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "\n  core-cmake:\n" in workflow
    assert "os: ubuntu-latest" in workflow
    assert "os: macos-latest" in workflow
    assert "expect_openmp: 'ON'" in workflow
    assert "blas_vendor: 'Apple'" in workflow
    assert "tools/check_cmake_install.sh" in workflow
    assert "for shared in OFF ON" in smoke
    assert "cmake --install" in smoke
    assert '-DEXPECT_HUGE_OPENMP="$expect_openmp"' in smoke
    assert "find_package(huge 2.0.0 CONFIG REQUIRED)" in consumer
    assert "huge::huge_core" in consumer
