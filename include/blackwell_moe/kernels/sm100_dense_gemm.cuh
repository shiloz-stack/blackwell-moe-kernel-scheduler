#pragma once

#include <cstddef>
#include <cstdint>

#include "blackwell_moe/kernels/grouped_gemm.cuh"

namespace blackwell_moe {

// One dense BF16 GEMM used to validate the native Blackwell math pipeline
// before that pipeline is embedded into an irregular grouped-MoE scheduler.
//
// A is row-major [M, K]. B is stored row-major [N, K] and interpreted as the
// column-major logical matrix [K, N]. D is row-major [M, N].
struct Sm100DenseGemmArguments {
  const void* a = nullptr;
  const void* b = nullptr;
  void* d = nullptr;
  std::uint32_t m = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  void* stream = nullptr;
};

struct Sm100DenseGemmPlanStats {
  std::size_t workspace_bytes = 0;
  std::uint32_t sm_count = 0;
  std::uint32_t persistent_ctas = 0;
  std::uint32_t tile_m = 128;
  std::uint32_t tile_n = 128;
  std::uint32_t tile_k = 64;
};

// Reusable launch plan for an SM100a-native BF16 GEMM. The selected CUTLASS
// collective uses a one-SM warp-specialized TMA mainloop, a one-SM TMA
// epilogue, and CUTLASS's persistent CLC tile scheduler. CUTLASS implements the
// tcgen05/TMEM/CLC plumbing; this wrapper owns problem validation, strides,
// workspace, and repeatable launch.
class Sm100DenseGemmPlan {
 public:
  Sm100DenseGemmPlan();
  ~Sm100DenseGemmPlan();

  Sm100DenseGemmPlan(const Sm100DenseGemmPlan&) = delete;
  Sm100DenseGemmPlan& operator=(const Sm100DenseGemmPlan&) = delete;
  Sm100DenseGemmPlan(Sm100DenseGemmPlan&& other) noexcept;
  Sm100DenseGemmPlan& operator=(Sm100DenseGemmPlan&& other) noexcept;

  KernelStatus initialize(const Sm100DenseGemmArguments& args);
  KernelStatus run(void* stream = nullptr);
  Sm100DenseGemmPlanStats stats() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

KernelStatus launch_sm100_dense_gemm(const Sm100DenseGemmArguments& args);
const char* sm100_dense_kernel_name();

}  // namespace blackwell_moe
