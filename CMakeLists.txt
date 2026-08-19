cmake_minimum_required(VERSION 3.24)

project(blackwell_moe_kernel_scheduler VERSION 0.1.0 LANGUAGES CXX)

option(BLACKWELL_MOE_ENABLE_CUDA "Build CUDA kernel entry points" OFF)
option(BLACKWELL_MOE_BUILD_CUTLASS_BASELINE
       "Build the optional external CUTLASS grouped-GEMM baseline" OFF)
option(BLACKWELL_MOE_ENABLE_SM100_NATIVE
       "Build the legacy CuTe SM100a reference kernels" OFF)
option(BLACKWELL_MOE_BUILD_TESTS "Build unit tests" ON)

if(BLACKWELL_MOE_ENABLE_SM100_NATIVE AND NOT BLACKWELL_MOE_ENABLE_CUDA)
  message(FATAL_ERROR
    "BLACKWELL_MOE_ENABLE_SM100_NATIVE requires BLACKWELL_MOE_ENABLE_CUDA=ON")
endif()

if(BLACKWELL_MOE_BUILD_CUTLASS_BASELINE AND NOT BLACKWELL_MOE_ENABLE_CUDA)
  message(FATAL_ERROR
    "BLACKWELL_MOE_BUILD_CUTLASS_BASELINE requires BLACKWELL_MOE_ENABLE_CUDA=ON")
endif()

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

  add_library(blackwell_moe_cuda
    src/kernels/scheduler_probe.cu)
  target_include_directories(blackwell_moe_cuda PUBLIC include)
  target_link_libraries(blackwell_moe_cuda
    PUBLIC blackwell_moe_core
    PRIVATE CUDA::cudart)
  set_target_properties(blackwell_moe_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")

  add_executable(moe_scheduler_gpu_bench
    benchmarks/scheduler_gpu_benchmark.cu)
  target_link_libraries(moe_scheduler_gpu_bench PRIVATE
    blackwell_moe_cuda CUDA::cudart)
  set_target_properties(moe_scheduler_gpu_bench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")

  if(BLACKWELL_MOE_ENABLE_SM100_NATIVE)
    include(cmake/Dependencies.cmake)
    # tcgen05 and TMEM are architecture-accelerated features. A plain sm_100
    # target is insufficient: this translation unit must be compiled for
    # compute_100a/sm_100a. CUDA_ARCHITECTURES is disabled only on this target
    # so the explicit native flag does not produce a duplicate sm_100 image.
    add_library(blackwell_moe_sm100
      src/kernels/sm100_dense_gemm.cu
      src/kernels/sm100_2sm_gemm.cu
      src/kernels/direct_cute_gemm.cu)
    target_include_directories(blackwell_moe_sm100 PUBLIC include)
    target_link_libraries(blackwell_moe_sm100
      PUBLIC blackwell_moe_cuda
      PRIVATE blackwell_moe_cutlass_headers CUDA::cudart)
    target_compile_options(blackwell_moe_sm100 PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_100a,code=sm_100a>)
    set_target_properties(blackwell_moe_sm100 PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES OFF)

    add_executable(sm100_dense_bench benchmarks/sm100_dense_benchmark.cu)
    target_link_libraries(sm100_dense_bench PRIVATE
      blackwell_moe_sm100 blackwell_moe_cutlass_headers CUDA::cudart)
    set_target_properties(sm100_dense_bench PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")

    add_executable(sm100_2sm_bench benchmarks/sm100_2sm_benchmark.cu)
    target_link_libraries(sm100_2sm_bench PRIVATE
      blackwell_moe_sm100 blackwell_moe_cutlass_headers CUDA::cudart)
    set_target_properties(sm100_2sm_bench PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")

    add_executable(direct_cute_bench benchmarks/direct_cute_benchmark.cu)
    target_link_libraries(direct_cute_bench PRIVATE
      blackwell_moe_sm100 blackwell_moe_cutlass_headers CUDA::cudart)
    set_target_properties(direct_cute_bench PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")
  endif()

  if(BLACKWELL_MOE_BUILD_CUTLASS_BASELINE)
    if(NOT TARGET blackwell_moe_cutlass_headers)
      include(cmake/Dependencies.cmake)
    endif()

    add_library(blackwell_moe_cutlass_baseline
      baselines/cutlass/cutlass_baseline.cu)
    target_include_directories(blackwell_moe_cutlass_baseline PUBLIC include)
    target_link_libraries(blackwell_moe_cutlass_baseline
      PUBLIC blackwell_moe_core
      PRIVATE blackwell_moe_cutlass_headers CUDA::cudart)
    set_target_properties(blackwell_moe_cutlass_baseline PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")

    add_executable(moe_cutlass_baseline_bench
      baselines/cutlass/benchmark.cu)
    target_link_libraries(moe_cutlass_baseline_bench PRIVATE
      blackwell_moe_cutlass_baseline blackwell_moe_cutlass_headers CUDA::cudart)
    set_target_properties(moe_cutlass_baseline_bench PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")
  endif()
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
    add_executable(test_scheduler_probe tests/test_scheduler_probe.cu)
    target_link_libraries(test_scheduler_probe PRIVATE
      blackwell_moe_cuda CUDA::cudart)
    set_target_properties(test_scheduler_probe PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")
    add_test(NAME scheduler_probe COMMAND test_scheduler_probe)

    if(BLACKWELL_MOE_ENABLE_SM100_NATIVE)
      add_executable(test_sm100_dense_correctness
        tests/test_sm100_dense_correctness.cu)
      target_link_libraries(test_sm100_dense_correctness PRIVATE
        blackwell_moe_sm100 blackwell_moe_cutlass_headers CUDA::cudart)
      set_target_properties(test_sm100_dense_correctness PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")
      add_test(NAME sm100_dense_correctness
               COMMAND test_sm100_dense_correctness)

      add_executable(test_direct_cute_correctness
        tests/test_direct_cute_correctness.cu)
      target_link_libraries(test_direct_cute_correctness PRIVATE
        blackwell_moe_sm100 blackwell_moe_cutlass_headers CUDA::cudart)
      set_target_properties(test_direct_cute_correctness PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")
      add_test(NAME direct_cute_correctness
               COMMAND test_direct_cute_correctness)

      add_executable(test_sm100_2sm_correctness
        tests/test_sm100_2sm_correctness.cu)
      target_link_libraries(test_sm100_2sm_correctness PRIVATE
        blackwell_moe_sm100 blackwell_moe_cutlass_headers CUDA::cudart)
      set_target_properties(test_sm100_2sm_correctness PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")
      add_test(NAME sm100_2sm_correctness
               COMMAND test_sm100_2sm_correctness)
    endif()

    if(BLACKWELL_MOE_BUILD_CUTLASS_BASELINE)
      add_executable(test_cutlass_baseline_correctness
        baselines/cutlass/test_correctness.cu)
      target_link_libraries(test_cutlass_baseline_correctness PRIVATE
        blackwell_moe_cutlass_baseline blackwell_moe_cutlass_headers CUDA::cudart)
      set_target_properties(test_cutlass_baseline_correctness PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BLACKWELL_MOE_CUDA_ARCHITECTURES}")
      add_test(NAME cutlass_baseline_correctness
               COMMAND test_cutlass_baseline_correctness)
    endif()
  endif()
endif()
