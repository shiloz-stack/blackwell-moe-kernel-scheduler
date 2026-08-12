include(FetchContent)

# Keep the baseline reproducible. This tag resolves to NVIDIA CUTLASS release
# commit e6233cb (v4.6.0). We consume headers only instead of configuring the
# full CUTLASS project and its large kernel/test matrix.
set(BLACKWELL_MOE_CUTLASS_TAG "v4.6.0" CACHE STRING
    "Pinned NVIDIA CUTLASS release")

FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG ${BLACKWELL_MOE_CUTLASS_TAG}
  GIT_SHALLOW TRUE)

FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  FetchContent_Populate(cutlass)
endif()

add_library(blackwell_moe_cutlass_headers INTERFACE)
target_include_directories(blackwell_moe_cutlass_headers INTERFACE
  ${cutlass_SOURCE_DIR}/include)
target_compile_definitions(blackwell_moe_cutlass_headers INTERFACE
  BLACKWELL_MOE_CUTLASS_REVISION="${BLACKWELL_MOE_CUTLASS_TAG}")

