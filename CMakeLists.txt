cmake_minimum_required(VERSION 3.24)

project(blackwell_moe_kernel_scheduler VERSION 0.1.0 LANGUAGES CXX)

option(BLACKWELL_MOE_ENABLE_CUDA "Build CUDA kernel entry points" OFF)
option(BLACKWELL_MOE_BUILD_TESTS "Build unit tests" ON)

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

if(BLACKWELL_MOE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 12.8 REQUIRED)
  include(cmake/CudaArch.cmake)
  include(cmake/Dependencies.cmake)

  add_library(blackwell_moe_cuda src/kernels/cutlass_baseline.cu)
  target_include_directories(blackwell_moe_cuda PUBLIC include)
  target_link_libraries(blackwell_moe_cuda
    PUBLIC blackwell_moe_core
    PRIVATE blackwell_moe_cutlass_headers CUDA::cudart)
  set_target_properties(blackwell_moe_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")
endif()

if(BLACKWELL_MOE_BUILD_TESTS)
  enable_testing()

  add_executable(test_workload tests/test_workload.cpp)
  target_link_libraries(test_workload PRIVATE blackwell_moe_core)
  add_test(NAME workload COMMAND test_workload)

  add_executable(test_scheduler tests/test_scheduler.cpp)
  target_link_libraries(test_scheduler PRIVATE blackwell_moe_core)
  add_test(NAME scheduler COMMAND test_scheduler)

  if(BLACKWELL_MOE_ENABLE_CUDA)
    add_executable(test_cutlass_correctness tests/test_cutlass_correctness.cu)
    target_link_libraries(test_cutlass_correctness PRIVATE
      blackwell_moe_cuda blackwell_moe_cutlass_headers CUDA::cudart)
    set_target_properties(test_cutlass_correctness PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")
    add_test(NAME cutlass_correctness COMMAND test_cutlass_correctness)
  endif()
endif()
