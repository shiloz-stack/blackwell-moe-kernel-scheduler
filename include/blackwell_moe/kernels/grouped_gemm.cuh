#pragma once

#include <cstdint>

#include "blackwell_moe/scheduler/scheduler.cuh"

namespace blackwell_moe {

enum class DataType { kBf16, kFp8E4M3 };

struct GroupedGemmArguments {
  const void* const* a = nullptr;
  const void* const* b = nullptr;
  void* const* d = nullptr;
  const std::uint32_t* tokens_per_expert = nullptr;
  std::uint32_t experts = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  DataType data_type = DataType::kBf16;
  SchedulerKind scheduler = SchedulerKind::kExpertOrder;
  void* stream = nullptr;
};

enum class KernelStatus {
  kSuccess,
  kInvalidArgument,
  kNotImplemented,
  kRuntimeError,
};

// Phase 1 provides a deliberately explicit boundary. Phase 2 will replace the
// not-implemented CUDA translation unit with a pinned CUTLASS grouped-GEMM.
KernelStatus launch_grouped_gemm_baseline(const GroupedGemmArguments& args);

}  // namespace blackwell_moe

