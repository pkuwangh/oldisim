cmake_minimum_required(VERSION 3.13)
project(oldisimtp)

add_library(boost_1_53 INTERFACE)
target_include_directories(
    boost_1_53 INTERFACE boost_1_53_0/)

add_library(cereal INTERFACE)
target_include_directories(
    cereal INTERFACE cereal/include/
)
