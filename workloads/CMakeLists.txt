cmake_minimum_required(VERSION 3.13)
project(oldisim_workloads)

find_package(LibEvent REQUIRED)

find_package(PythonInterp REQUIRED)
set(ICACHEBUSTER_NUM_SPLITS 24)
set(ICACHEBUSTER_NUM_METHODS 100000)

find_program(GENGETOPT_EXECUTABLE gengetopt)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/search/LeafNodeCmdline.h
           ${CMAKE_CURRENT_BINARY_DIR}/search/LeafNodeCmdline.cc
    COMMAND ${GENGETOPT_EXECUTABLE}
        -i ${CMAKE_CURRENT_SOURCE_DIR}/search/LeafNodeCmdline.ggo
        -F LeafNodeCmdline
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/search
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/search/LeafNodeCmdline.ggo
)
add_custom_target(
    leafnode_gengetopt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/search/LeafNodeCmdline.h
            ${CMAKE_CURRENT_BINARY_DIR}/search/LeafNodeCmdline.cc
)
add_library(leafnodecmdline
    ${CMAKE_CURRENT_BINARY_DIR}/search/LeafNodeCmdline.h
    ${CMAKE_CURRENT_BINARY_DIR}/search/LeafNodeCmdline.cc)

add_dependencies(leafnodecmdline leafnode_gengetopt)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/search/ICacheBuster.cc
           ${CMAKE_CURRENT_BINARY_DIR}/search/ICacheBuster.h
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/search/gen_icache_buster.py
        --num_methods "${ICACHEBUSTER_NUM_METHODS}"
        --num_splits "${ICACHEBUSTER_NUM_SPLITS}"
        --output_dir "${CMAKE_CURRENT_BINARY_DIR}/search"
)
add_custom_target(
    genicache ALL
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/search/ICacheBuster.cc
        ${CMAKE_CURRENT_BINARY_DIR}/search/ICacheBuster.h)

file(GLOB icache_srclist "${CMAKE_CURRENT_BINARY_DIR}/search/ICacheBuster*")
add_library(icachebuster ${icache_srclist})

target_include_directories(icachebuster PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/search/)
target_compile_features(icachebuster PRIVATE cxx_std_11)
add_dependencies(icachebuster genicache)


add_executable(LeafNode
               search/LeafNode.cc
               search/PointerChase.cc)

target_compile_features(LeafNode PRIVATE cxx_std_11)
target_include_directories(LeafNode
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/search/ ${LIBEVENT_INCLUDE_DIR})
target_link_libraries(LeafNode
                      PRIVATE oldisimlib icachebuster leafnodecmdline
                      PUBLIC Threads::Threads ${LIBEVENT_LIB})
