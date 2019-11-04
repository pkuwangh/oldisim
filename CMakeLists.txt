cmake_minimum_required(VERSION 3.13)
project(oldisim CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})



find_library(JEMALLOC_LIB NAMES jemalloc)
find_path(JEMALLOC_INCLUDE_DIR NAMES jemalloc/jemalloc.h)
if (JEMALLOC_INCLUDE_DIR AND JEMALLOC_LIB)
    include_directories(${JEMALLOC_INCLUDE_DIR})

    set (CMAKE_REQUIRED_INCLUDES ${JEMALLOC_INCLUDE_DIR})
    include(CheckCXXSourceCompiles)
    CHECK_CXX_SOURCE_COMPILES("
#include <jemalloc/jemalloc.h>
#define JEMALLOC_VERSION_NUMERIC ((JEMALLOC_VERSION_MAJOR << 24) | (JEMALLOC_VERSION_MINOR << 16) | (JEMALLOC_VERSION_BUGFIX << 8) | JEMALLOC_VERSION_NDEV)
#if JEMALLOC_VERSION_NUMERIC < 0x03050100
# error jemalloc version >= 3.5.1 required
#endif
int main(void) { return 0; }" JEMALLOC_VERSION_MINIMUM)
    set (CMAKE_REQUIRED_INCLUDES)

    if(JEMALLOC_VERSION_MINIMUM)
        message(STATUS "Found jemalloc: ${JEMALLOC_LIB}")
        set(JEMALLOC_ENABLED 1)
    else()
        message(STATUS "Found jemalloc, but it was too old")
    endif()
endif()

if (JEMALLOC_ENABLED)
  add_definitions(-DUSE_JEMALLOC=1)
else()
  add_definitions(-DNO_JEMALLOC=1)
endif()

find_package(Threads REQUIRED)
find_package(LibEvent REQUIRED)


add_subdirectory(third_party)
add_subdirectory(oldisim)
add_subdirectory(workloads)

