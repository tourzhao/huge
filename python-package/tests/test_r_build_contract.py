"""Static contracts for the R package's native build configuration."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_r_makevars_explicitly_select_cxx17():
    for relative_path in ("src/Makevars.in", "src/Makevars.win"):
        source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        assert "CXX_STD = CXX17" in source

    description = (REPOSITORY_ROOT / "DESCRIPTION").read_text(encoding="utf-8")
    assert "SystemRequirements: C++17" in description


def test_macos_openmp_probe_compiles_and_links_cxx():
    source = (REPOSITORY_ROOT / "configure.ac").read_text(encoding="utf-8")

    assert "AC_PATH_PROG([RBIN], [R])" in source
    assert "AC_LANG_PUSH([C++])" in source
    assert "AC_LANG_POP([C++])" in source
    assert 'PKG_CXXFLAGS="${OPENMP_CXXFLAGS}"' in source
    assert 'PKG_LIBS="${OPENMP_CXXFLAGS}"' in source
    assert "CMD SHLIB conftest.cpp" in source
    assert "huge_openmp_probe" in source
    assert '#error "OpenMP compilation is not enabled"' in source
    assert "#pragma omp parallel for schedule(dynamic)" in source
    assert "omp_get_max_threads()" in source
    assert "integer(2)" in source
    assert "OPENMP_CFLAGS" not in source


def test_generated_configure_matches_openmp_source_contract():
    generated = (REPOSITORY_ROOT / "configure").read_text(encoding="utf-8")

    assert "ac_cv_path_RBIN" in generated
    assert 'PKG_CXXFLAGS="${OPENMP_CXXFLAGS}"' in generated
    assert 'PKG_LIBS="${OPENMP_CXXFLAGS}"' in generated
    assert "CMD SHLIB conftest.cpp" in generated
    assert "huge_openmp_probe" in generated
    assert '#error "OpenMP compilation is not enabled"' in generated
    assert "omp_get_max_threads()" in generated
    assert "integer(2)" in generated


def test_readme_macos_openmp_makevars_are_complete_and_additive():
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "SHLIB_OPENMP_CFLAGS = -fopenmp" in readme
    assert "SHLIB_OPENMP_CXXFLAGS = -fopenmp" in readme
    assert "CPPFLAGS += -I$(LLVM_LOC)/include" in readme
    assert "LDFLAGS += -L$(LLVM_LOC)/lib" in readme
