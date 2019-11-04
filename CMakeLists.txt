cmake_minimum_required(VERSION 3.13)
project(oldisim CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

find_package(Threads REQUIRED)
find_package(LibEvent REQUIRED)


add_subdirectory(third_party)
add_subdirectory(oldisim)
add_subdirectory(workloads)

