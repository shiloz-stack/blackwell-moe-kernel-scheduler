#pragma once

#include <cstddef>
#include <cstdint>

#include "blackwell_moe/kernels/grouped_gemm.cuh"

namespace blackwell_moe {

// Direct CuTe SM100 bring-up contract:
//   D_f32[M, N] = A_bf16[M, K] * transpose(B_bf16_storage[N, K])
//
// The first version intentionally requires complete tiles. Keeping boundary
// predication out of the kernel makes TMA, tcgen05, TMEM, and barrier failures
// independently diagnosable during the first B200 validation.
struct DirectCuteGemmArguments {
  const void* a = nullptr;
  const void* b = nullptr;
  void* d = nullptr;
  std::uint32_t m = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  void* stream = nullptr;
};

struct DirectCuteGemmStats {
  std::uint32_t grid_m = 0;
  std::uint32_t grid_n = 0;
  std::uint32_t tile_m = 128;
  std::uint32_t tile_n = 256;
  std::uint32_t tile_k = 64;
  std::size_t dynamic_smem_bytes = 0;
};

KernelStatus launch_direct_cute_gemm(
    const DirectCuteGemmArguments& args,
    DirectCuteGemmStats* stats = nullptr);

const char* direct_cute_kernel_name();

}  // namespace blackwell_moe
