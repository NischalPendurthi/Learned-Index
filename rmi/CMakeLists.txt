cmake_minimum_required(VERSION 3.16)
project(rmi CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(rmi_core STATIC src/rmi.cpp src/dataset.cpp)
target_include_directories(rmi_core PUBLIC include src)

# -ffp-contract=off is load-bearing: it stops the compiler from contracting
# `slope * key + intercept` into an FMA at some call sites and not others.
# Training and lookup must compute bit-identical predictions, or a stored error
# bound can be off by one and a key goes missing.
target_compile_options(rmi_core PUBLIC
  $<$<CONFIG:Release>:-O3 -march=native>
  -ffp-contract=off -Wall -Wextra)

add_executable(rmi_main src/main.cpp)
target_link_libraries(rmi_main PRIVATE rmi_core)

add_executable(rmi_bench bench/benchmark.cpp)
target_link_libraries(rmi_bench PRIVATE rmi_core)
