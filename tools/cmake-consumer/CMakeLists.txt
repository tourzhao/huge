cmake_minimum_required(VERSION 3.18)
project(huge_consumer LANGUAGES CXX)

find_package(huge 2.0.0 CONFIG REQUIRED)

option(EXPECT_HUGE_OPENMP "Require the installed target to export OpenMP" OFF)
get_target_property(huge_link_libraries huge::huge_core INTERFACE_LINK_LIBRARIES)
if(EXPECT_HUGE_OPENMP)
    if(NOT "${huge_link_libraries}" MATCHES "(^|;)OpenMP::OpenMP_CXX($|;)")
        message(FATAL_ERROR "huge::huge_core did not export OpenMP as expected")
    endif()
elseif("${huge_link_libraries}" MATCHES "(^|;)OpenMP::OpenMP_CXX($|;)")
    message(FATAL_ERROR "huge::huge_core unexpectedly exported OpenMP")
endif()

add_executable(huge_core_consumer main.cpp)
target_link_libraries(huge_core_consumer PRIVATE huge::huge_core)
