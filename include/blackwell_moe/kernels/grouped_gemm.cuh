#pragma once

#include <cstddef>
#include <cstdint>

#include "blackwell_moe/scheduler/scheduler.cuh"

namespace blackwell_moe {

enum class DataType { kBf16, kFp8E4M3 };

struct GroupedGemmArguments {
  // Host arrays containing device pointers. A is row-major [M_i, K], B is
  // row-major [N, K] (logically transposed by the GEMM), and D is row-major
  // [M_i, N]. Arrays and tokens_per_expert each contain `experts` entries.
  const void* const* a = nullptr;
  const void* const* b = nullptr;
  void* const* d = nullptr;
  const std::uint32_t* tokens_per_expert = nullptr;
  std::uint32_t experts = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  DataType data_type = DataType::kBf16;
  SchedulerKind scheduler = SchedulerKind::kExpertOrder;
  // Optional default CUDA stream, represented opaquely to keep this public
  // header usable by a non-CUDA C++ translation unit.
  void* stream = nullptr;
};

enum class KernelStatus {
  kSuccess,
  kInvalidArgument,
  kUnsupportedDataType,
  kUnsupportedScheduler,
  kUnsupportedDevice,
  kCudaError,
  kCutlassError,
};

const char* kernel_status_string(KernelStatus status);

struct GroupedGemmPlanStats {
  std::uint32_t active_experts = 0;
  std::uint32_t threadblock_count = 0;
  std::size_t workspace_bytes = 0;
};

// Owns device-side grouped-GEMM metadata and CUTLASS workspace. initialize()
// is intentionally separate from run() so benchmark timing excludes metadata
// allocation and host-to-device setup.
class Bf16GroupedGemmPlan {
 public:
  Bf16GroupedGemmPlan();
  ~Bf16GroupedGemmPlan();

  Bf16GroupedGemmPlan(const Bf16GroupedGemmPlan&) = delete;
  Bf16GroupedGemmPlan& operator=(const Bf16GroupedGemmPlan&) = delete;
  Bf16GroupedGemmPlan(Bf16GroupedGemmPlan&& other) noexcept;
  Bf16GroupedGemmPlan& operator=(Bf16GroupedGemmPlan&& other) noexcept;

  KernelStatus initialize(const GroupedGemmArguments& args);
  KernelStatus run(void* stream = nullptr);
  GroupedGemmPlanStats stats() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

// Convenience one-shot path. Performance measurements should reuse a plan.
KernelStatus launch_grouped_gemm_baseline(const GroupedGemmArguments& args);

}  // namespace blackwell_moe
