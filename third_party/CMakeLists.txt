cmake_minimum_required(VERSION 3.12)
project(oldisimtp LANGUAGES C CXX)

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/cereal/CMakeLists.txt" OR
   NOT EXISTS "${PROJECT_SOURCE_DIR}/gapbs/src/pr.cc")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()


find_package(OpenMP)
set(GAPBS_SRCS
    gapbs/src/pr.cc
    gapbs/src/util.h
)
add_library(gapbs ${GAPBS_SRCS})
target_include_directories(gapbs INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(gapbs PRIVATE -Wall)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gapbs PRIVATE OpenMP::OpenMP_CXX)
endif()
