#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
smoke_root=$(mktemp -d "${TMPDIR:-/tmp}/huge-cmake-smoke.XXXXXX")
trap 'rm -rf -- "$smoke_root"' EXIT HUP INT TERM

openmp=${HUGE_CMAKE_SMOKE_OPENMP:-ON}
expect_openmp=${HUGE_CMAKE_SMOKE_EXPECT_OPENMP:-OFF}
blas_vendor=${HUGE_CMAKE_SMOKE_BLA_VENDOR:-}

for shared in OFF ON; do
    if [ "$shared" = "ON" ]; then
        library_kind=shared
    else
        library_kind=static
    fi

    core_build="$smoke_root/core-$library_kind"
    install_root="$smoke_root/install-$library_kind"
    consumer_build="$smoke_root/consumer-$library_kind"

    set -- -S "$repository_root" -B "$core_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHUGE_BUILD_SHARED="$shared" \
        -DHUGE_OPENMP="$openmp"
    if [ -n "$blas_vendor" ]; then
        set -- "$@" "-DBLA_VENDOR=$blas_vendor"
    fi
    cmake "$@"
    cmake --build "$core_build" --config Release
    cmake --install "$core_build" --config Release --prefix "$install_root"

    cmake -S "$repository_root/tools/cmake-consumer" -B "$consumer_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$install_root" \
        -DEXPECT_HUGE_OPENMP="$expect_openmp"
    cmake --build "$consumer_build" --config Release
    "$consumer_build/huge_core_consumer"
done
