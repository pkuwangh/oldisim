cmake_minimum_required(VERSION 3.12)
project(oldisimtp LANGUAGES C CXX)

add_library(boost_1_53 INTERFACE)
target_include_directories(
    boost_1_53 INTERFACE boost_1_53_0/)

add_library(cereal INTERFACE)
target_include_directories(
    cereal INTERFACE cereal/include/
)

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
