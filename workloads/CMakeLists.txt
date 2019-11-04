cmake_minimum_required(VERSION 3.13)
project(oldisim_workloads)


find_package(PythonInterp REQUIRED)

# ICacheBuster parameters
set(ICACHEBUSTER_NUM_SPLITS 24)
set(ICACHEBUSTER_NUM_METHODS 100000)

find_program(GENGETOPT_EXECUTABLE gengetopt)

# Generate getopts for LeafNode
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


# Generate ICacheBuster methods
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/search)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/search/gen_icache_buster.py
        --num_methods "${ICACHEBUSTER_NUM_METHODS}"
        --num_splits "${ICACHEBUSTER_NUM_SPLITS}"
        --output_dir ${CMAKE_CURRENT_BINARY_DIR}/search
)
add_custom_target(
    genicache ALL
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/search/ICacheBuster.cc
        ${CMAKE_CURRENT_BINARY_DIR}/search/ICacheBuster.h)

# Define ICacheBuster library target
file(GLOB icache_srclist "${CMAKE_CURRENT_BINARY_DIR}/search/ICacheBuster*")
add_library(icachebuster ${icache_srclist})

target_include_directories(icachebuster PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/search/)
target_compile_features(icachebuster PRIVATE cxx_std_11)
add_dependencies(icachebuster genicache)


# Build LeafNode binary
add_executable(LeafNode
               search/LeafNode.cc
               search/PointerChase.cc
               search/HistogramRandomSampler.cc)

target_compile_features(LeafNode PRIVATE cxx_std_11)
target_include_directories(LeafNode
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/search/ ${LIBEVENT_INCLUDE_DIR})
target_link_libraries(LeafNode
                      PRIVATE oldisimlib icachebuster leafnodecmdline
                      PUBLIC Threads::Threads ${LIBEVENT_LIB})


# Generate getopts for ParentNode
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/search/ParentNodeCmdline.h
           ${CMAKE_CURRENT_BINARY_DIR}/search/ParentNodeCmdline.cc
    COMMAND ${GENGETOPT_EXECUTABLE}
        -i ${CMAKE_CURRENT_SOURCE_DIR}/search/ParentNodeCmdline.ggo
        -F ParentNodeCmdline
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/search
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/search/ParentNodeCmdline.ggo
)
add_custom_target(
    parentnode_gengetopt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/search/ParentNodeCmdline.h
            ${CMAKE_CURRENT_BINARY_DIR}/search/ParentNodeCmdline.cc
)
add_library(parentnodecmdline
    ${CMAKE_CURRENT_BINARY_DIR}/search/ParentNodeCmdline.h
    ${CMAKE_CURRENT_BINARY_DIR}/search/ParentNodeCmdline.cc)

add_dependencies(parentnodecmdline parentnode_gengetopt)


# Build ParentNode binary
add_executable(ParentNode
               search/ParentNode.cc)
target_compile_features(ParentNode PRIVATE cxx_std_11)
target_include_directories(
    ParentNode
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/search/ ${LIBEVENT_INCLUDE_DIR})
target_link_libraries(
    ParentNode
    PRIVATE oldisimlib parentnodecmdline
    PUBLIC Threads::Threads ${LIBEVENT_LIB})



# Generate getops for LoadBalancerNode
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/search/LoadBalancerNodeCmdline.h
           ${CMAKE_CURRENT_BINARY_DIR}/search/LoadBalancerNodeCmdline.cc
    COMMAND ${GENGETOPT_EXECUTABLE}
        -i ${CMAKE_CURRENT_SOURCE_DIR}/search/LoadBalancerNodeCmdline.ggo
        -F LoadBalancerNodeCmdline
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/search
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/search/LoadBalancerNodeCmdline.ggo
)
add_custom_target(
    LoadBalancernode_gengetopt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/search/LoadBalancerNodeCmdline.h
            ${CMAKE_CURRENT_BINARY_DIR}/search/LoadBalancerNodeCmdline.cc
)
add_library(LoadBalancernodecmdline
    ${CMAKE_CURRENT_BINARY_DIR}/search/LoadBalancerNodeCmdline.h
    ${CMAKE_CURRENT_BINARY_DIR}/search/LoadBalancerNodeCmdline.cc)

add_dependencies(LoadBalancernodecmdline LoadBalancernode_gengetopt)


# Build LoadBalancerNode binary
add_executable(LoadBalancerNode
               search/LoadBalancerNode.cc)
target_compile_features(LoadBalancerNode PRIVATE cxx_std_11)
target_include_directories(
    LoadBalancerNode
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/search/ ${LIBEVENT_INCLUDE_DIR})
target_link_libraries(
    LoadBalancerNode
    PRIVATE oldisimlib LoadBalancernodecmdline
    PUBLIC Threads::Threads ${LIBEVENT_LIB})


# Generate getops for DriverNode
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/search/DriverNodeCmdline.h
           ${CMAKE_CURRENT_BINARY_DIR}/search/DriverNodeCmdline.cc
    COMMAND ${GENGETOPT_EXECUTABLE}
        -i ${CMAKE_CURRENT_SOURCE_DIR}/search/DriverNodeCmdline.ggo
        -F DriverNodeCmdline
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/search
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/search/DriverNodeCmdline.ggo
)
add_custom_target(
    Drivernode_gengetopt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/search/DriverNodeCmdline.h
            ${CMAKE_CURRENT_BINARY_DIR}/search/DriverNodeCmdline.cc
)
add_library(Drivernodecmdline
    ${CMAKE_CURRENT_BINARY_DIR}/search/DriverNodeCmdline.h
    ${CMAKE_CURRENT_BINARY_DIR}/search/DriverNodeCmdline.cc)

add_dependencies(Drivernodecmdline Drivernode_gengetopt)


# Build DriverNode binary
add_executable(DriverNode
               search/DriverNode.cc)
target_compile_features(DriverNode PRIVATE cxx_std_11)
target_include_directories(
    DriverNode
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/search/ ${LIBEVENT_INCLUDE_DIR})
target_link_libraries(
    DriverNode
    PRIVATE oldisimlib Drivernodecmdline
    PUBLIC Threads::Threads ${LIBEVENT_LIB})


# Build LeafKernel binary
add_executable(LeafKernel
               search/LeafKernel.cc
               search/PointerChase.cc)
target_compile_features(LeafKernel PRIVATE cxx_std_11)
target_include_directories(LeafKernel
                            PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/search/)
target_link_libraries(LeafKernel
                      PRIVATE oldisimlib icachebuster
                      )
