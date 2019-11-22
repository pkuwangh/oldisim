cmake_minimum_required(VERSION 3.12)
project(oldisim_ranking)

include(build-folly)

find_package(DoubleConversion REQUIRED)
find_package(Glog REQUIRED)
find_package(BZip2 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Zstd REQUIRED)
find_package(LZ4 REQUIRED)
find_package(Snappy REQUIRED)
find_package(LibLZMA REQUIRED)
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
    TimekeeperPool.cpp
    ${RANKING_THRIFT_SRCS} ${RANKING_THRIFT_HDRS})
target_include_directories(LeafNodeRank
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${LIBEVENT_INCLUDE_DIR}
        ${JEMALLOC_INCLUDE_DIR}
        ${THRIFT_INCLUDE_DIR}
        ${BZIP2_INCLUDE_DIR}
        ${ZSTD_INCLUDE_DIR}
        ${LZ4_INCLUDE_DIR}
        ${SNAPPY_INCLUDE_DIR}
        ${LIBLZMA_INCLUDE_DIRS}
        ${FOLLY_INCLUDE_DIR})
target_link_libraries(LeafNodeRank
    PRIVATE
        OLDISim::OLDISim
        leafNodeRankcmdline
        rankingDwarfs
        icachebuster
        PointerChaser
        ${FOLLY_LIBRARIES}
        ${CMAKE_DL_LIBS}
        glog::glog
        ${DOUBLE_CONVERSION_LIBRARY}
    PUBLIC
        Threads::Threads
        ZLIB::ZLIB
        ${BZIP2_LIBRARIES}
        ${LZ4_LIBRARY}
        ${ZSTD_LIBRARY}
        ${SNAPPY_LIBRARY}
        ${LIBEVENT_LIB}
        ${JEMALLOC_LIB}
        ${LIBLZMA_LIBRARIES}
        ${THRIFT_SHARED_LIBRARY})
target_compile_options(LeafNodeRank PUBLIC -fno-omit-frame-pointer)
add_dependencies(LeafNodeRank folly ${RANKING_THRIFT_TGTS})


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
target_include_directories(
    ParentNodeRank
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/
           ${LIBEVENT_INCLUDE_DIR}
           ${JEMALLOC_INCLUDE_DIR}
           ${BZIP2_INCLUDE_DIR}
           ${ZSTD_INCLUDE_DIR}
           ${LZ4_INCLUDE_DIR}
           ${SNAPPY_INCLUDE_DIR}
           ${LIBLZMA_INCLUDE_DIRS}
           ${THRIFT_INCLUDE_DIR}
           ${FOLLY_INCLUDE_DIR})
target_link_libraries(
    ParentNodeRank
    PRIVATE
        OLDISim::OLDISim
        parentNodeRankcmdline
        ${FOLLY_LIBRARIES}
        ${CMAKE_DL_LIBS}
        glog::glog
        ${DOUBLE_CONVERSION_LIBRARY}
    PUBLIC
        Threads::Threads
        ZLIB::ZLIB
        ${BZIP2_LIBRARIES}
        ${LZ4_LIBRARY}
        ${ZSTD_LIBRARY}
        ${SNAPPY_LIBRARY}
        ${LIBLZMA_LIBRARIES}
        ${LIBEVENT_LIB}
        ${JEMALLOC_LIB}
        ${THRIFT_SHARED_LIBRARY})
add_dependencies(ParentNodeRank ${RANKING_THRIFT_TGTS})

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
target_include_directories(
    DriverNodeRank
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/
           ${LIBEVENT_INCLUDE_DIR}
           ${JEMALLOC_INCLUDE_DIR})

target_link_libraries(
    DriverNodeRank
    PRIVATE OLDISim::OLDISim DriverNodeRankcmdline
    PUBLIC Threads::Threads ${LIBEVENT_LIB} ${JEMALLOC_LIB})
