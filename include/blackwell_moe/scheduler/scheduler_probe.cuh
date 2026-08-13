#pragma once

#include <cstdint>

#include "blackwell_moe/kernels/grouped_gemm.cuh"
#include "blackwell_moe/work_tile.cuh"

namespace blackwell_moe {

enum class DeviceSchedulerKind : std::uint32_t {
  kStaticPersistent = 0,
  kDynamicQueue = 1,
};

// Device-resident observability buffers make the scheduling policy testable
// independently from GEMM math.  Every tile must be visited exactly once;
// per-CTA counters expose the realized (not simulated) work distribution.
struct SchedulerProbeArguments {
  const ExpertTile* tiles = nullptr;
  std::uint32_t tile_count = 0;
  std::uint32_t cta_count = 0;
  std::uint32_t claim_size = 1;
  std::uint32_t work_scale = 1;
  std::uint32_t* next_tile = nullptr;
  std::uint32_t* tile_visits = nullptr;
  std::uint64_t* tile_checksums = nullptr;
  std::uint32_t* cta_tile_counts = nullptr;
  std::uint64_t* cta_valid_elements = nullptr;
  void* stream = nullptr;
};

KernelStatus launch_scheduler_probe(
    DeviceSchedulerKind scheduler,
    const SchedulerProbeArguments& args);

const char* device_scheduler_name(DeviceSchedulerKind scheduler);

}  // namespace blackwell_moe
