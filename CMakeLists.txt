cmake_minimum_required(VERSION 3.24)

project(blackwell_moe_kernel_scheduler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(blackwell_moe_core
  src/workload/workload.cpp
  src/scheduler/scheduler.cpp)

target_include_directories(blackwell_moe_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_options(blackwell_moe_core PRIVATE
  $<$<CXX_COMPILER_ID:Clang,GNU>:-Wall;-Wextra;-Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>)

add_executable(moe_workload_bench benchmarks/benchmark_main.cpp)
target_link_libraries(moe_workload_bench PRIVATE blackwell_moe_core)

option(BLACKWELL_MOE_BUILD_TESTS "Build unit tests" ON)

if(BLACKWELL_MOE_BUILD_TESTS)
  enable_testing()

  add_executable(test_workload tests/test_workload.cpp)
  target_link_libraries(test_workload PRIVATE blackwell_moe_core)
  add_test(NAME workload COMMAND test_workload)

  add_executable(test_scheduler tests/test_scheduler.cpp)
  target_link_libraries(test_scheduler PRIVATE blackwell_moe_core)
  add_test(NAME scheduler COMMAND test_scheduler)
endif()
