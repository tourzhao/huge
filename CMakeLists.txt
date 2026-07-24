cmake_minimum_required(VERSION 3.18)
project(huge VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ---------- Options ----------------------------------------------------------
option(HUGE_BUILD_SHARED "Build shared library instead of static" OFF)
option(HUGE_OPENMP       "Enable OpenMP parallelism"             ON)

# ---------- BLAS -------------------------------------------------------------
# The core calls the Fortran BLAS interface with 32-bit integer arguments.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
    set(BLA_SIZEOF_INTEGER 4)
endif()
find_package(BLAS REQUIRED)

# ---------- OpenMP -----------------------------------------------------------
set(HUGE_WITH_OPENMP OFF)
if(HUGE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        set(HUGE_WITH_OPENMP ON)
    else()
        message(STATUS "OpenMP CXX was not found; building serial huge_core")
    endif()
endif()

# ---------- Core library -----------------------------------------------------
set(HUGE_LIB_TYPE STATIC)
if(HUGE_BUILD_SHARED)
    set(HUGE_LIB_TYPE SHARED)
endif()

add_library(huge_core ${HUGE_LIB_TYPE} src/huge_core.cpp)
target_include_directories(huge_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(huge_core PUBLIC cxx_std_17)
target_compile_options(huge_core PRIVATE
    "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>"
    "$<$<CXX_COMPILER_ID:MSVC>:/W4>"
)
target_link_libraries(huge_core PUBLIC BLAS::BLAS)

if(HUGE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(huge_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------- Install rules ----------------------------------------------------
install(TARGETS huge_core
    EXPORT hugeTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY src/huge
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
install(EXPORT hugeTargets
    FILE hugeTargets.cmake
    NAMESPACE huge::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/huge
)

configure_package_config_file(
    cmake/hugeConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/hugeConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/huge
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hugeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/hugeConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/hugeConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/huge
)
