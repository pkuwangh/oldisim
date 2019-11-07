cmake_minimum_required(VERSION 3.12)
project(oldisim_ranking)


find_package(ZLIB REQUIRED)
find_package(Thrift REQUIRED)
find_program(GENGETOPT_EXECUTABLE gengetopt REQUIRED)

THRIFT_GENERATE_CPP(RANKING_THRIFT_SRCS RANKING_THRIFT_HDRS RANKING_THRIFT_TGTS
    SOURCE_ROOT gen-cpp/
    BINARY_ROOT gen-cpp/
    THRIFT_FILES ranking.thrift)

# Build Ranking Dwarfs library
add_library(rankingDwarfs
    dwarfs/pagerank.cpp
    dwarfs/pagerank.h)
target_include_directories(rankingDwarfs PUBLIC dwarfs/)
target_compile_features(rankingDwarfs PRIVATE cxx_std_11)
target_link_libraries(rankingDwarfs PUBLIC gapbs)

# Generate getopts for LeafNodeRank
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LeafNodeRankCmdline.h
           ${CMAKE_CURRENT_BINARY_DIR}/LeafNodeRankCmdline.cc
    COMMAND ${GENGETOPT_EXECUTABLE}
        -i ${CMAKE_CURRENT_SOURCE_DIR}/LeafNodeRankCmdline.ggo
        -F LeafNodeRankCmdline
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/LeafNodeRankCmdline.ggo
)
add_custom_target(
    leafNodeRank_gengetopt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/LeafNodeRankCmdline.h
            ${CMAKE_CURRENT_BINARY_DIR}/LeafNodeRankCmdline.cc
)
add_library(leafNodeRankcmdline
    ${CMAKE_CURRENT_BINARY_DIR}/LeafNodeRankCmdline.h
    ${CMAKE_CURRENT_BINARY_DIR}/LeafNodeRankCmdline.cc)

add_dependencies(leafNodeRankcmdline leafNodeRank_gengetopt)

# Build LeafNodeRank binary
add_executable(LeafNodeRank
    LeafNodeRank.cc
    ${RANKING_THRIFT_SRCS} ${RANKING_THRIFT_HDRS})
target_compile_features(LeafNodeRank PRIVATE cxx_std_11)
target_include_directories(LeafNodeRank
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${LIBEVENT_INCLUDE_DIR} ${JEMALLOC_INCLUDE_DIR} ${THRIFT_INCLUDE_DIR})

target_link_libraries(LeafNodeRank
    PRIVATE oldisimlib leafNodeRankcmdline rankingDwarfs icachebuster
    PUBLIC  Threads::Threads ZLIB::ZLIB ${LIBEVENT_LIB} ${JEMALLOC_LIB} ${THRIFT_SHARED_LIBRARY})
target_compile_options(LeafNodeRank PUBLIC -fno-omit-frame-pointer)

# Generate getopts for ParentNodeRank
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ParentNodeRankCmdline.h
           ${CMAKE_CURRENT_BINARY_DIR}/ParentNodeRankCmdline.cc
    COMMAND ${GENGETOPT_EXECUTABLE}
        -i ${CMAKE_CURRENT_SOURCE_DIR}/ParentNodeRankCmdline.ggo
        -F ParentNodeRankCmdline
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ParentNodeRankCmdline.ggo
)
add_custom_target(
    parentNodeRank_gengetopt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ParentNodeRankCmdline.h
            ${CMAKE_CURRENT_BINARY_DIR}/ParentNodeRankCmdline.cc
)
add_library(parentNodeRankcmdline
    ${CMAKE_CURRENT_BINARY_DIR}/ParentNodeRankCmdline.h
    ${CMAKE_CURRENT_BINARY_DIR}/ParentNodeRankCmdline.cc)

add_dependencies(parentNodeRankcmdline parentNodeRank_gengetopt)


# Build ParentNodeRank binary
add_executable(ParentNodeRank
               ParentNodeRank.cc
               ${RANKING_THRIFT_SRCS} ${RANKING_THRIFT_HDRS})
target_compile_features(ParentNodeRank PRIVATE cxx_std_11)
target_include_directories(
    ParentNodeRank
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/
           ${LIBEVENT_INCLUDE_DIR}
           ${JEMALLOC_INCLUDE_DIR}
           ${THRIFT_INCLUDE_DIR})
target_link_libraries(
    ParentNodeRank
    PRIVATE oldisimlib parentNodeRankcmdline
    PUBLIC Threads::Threads ${LIBEVENT_LIB} ${JEMALLOC_LIB} ${THRIFT_SHARED_LIBRARY})


# Generate getops for DriverNodeRank
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/DriverNodeRankCmdline.h
           ${CMAKE_CURRENT_BINARY_DIR}/DriverNodeRankCmdline.cc
    COMMAND ${GENGETOPT_EXECUTABLE}
        -i ${CMAKE_CURRENT_SOURCE_DIR}/DriverNodeRankCmdline.ggo
        -F DriverNodeRankCmdline
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/DriverNodeRankCmdline.ggo
)
add_custom_target(
    DriverNodeRank_gengetopt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/DriverNodeRankCmdline.h
            ${CMAKE_CURRENT_BINARY_DIR}/DriverNodeRankCmdline.cc
)
add_library(DriverNodeRankcmdline
    ${CMAKE_CURRENT_BINARY_DIR}/DriverNodeRankCmdline.h
    ${CMAKE_CURRENT_BINARY_DIR}/DriverNodeRankCmdline.cc)

add_dependencies(DriverNodeRankcmdline DriverNodeRank_gengetopt)


# Build DriverNodeRank binary
add_executable(DriverNodeRank
               DriverNodeRank.cc)
target_compile_features(DriverNodeRank PRIVATE cxx_std_11)
target_include_directories(
    DriverNodeRank
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/
           ${LIBEVENT_INCLUDE_DIR}
           ${JEMALLOC_INCLUDE_DIR})

target_link_libraries(
    DriverNodeRank
    PRIVATE oldisimlib DriverNodeRankcmdline
    PUBLIC Threads::Threads ${LIBEVENT_LIB} ${JEMALLOC_LIB})
