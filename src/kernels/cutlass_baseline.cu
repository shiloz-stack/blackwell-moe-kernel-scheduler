#include "blackwell_moe/kernels/grouped_gemm.cuh"

namespace blackwell_moe {

KernelStatus launch_grouped_gemm_baseline(const GroupedGemmArguments& args) {
  if (args.a == nullptr || args.b == nullptr || args.d == nullptr ||
      args.tokens_per_expert == nullptr || args.experts == 0 || args.n == 0 ||
      args.k == 0) {
    return KernelStatus::kInvalidArgument;
  }
  // Phase 2 pins CUTLASS and implements the first trusted grouped-GEMM here.
  return KernelStatus::kNotImplemented;
}

}  // namespace blackwell_moe

