#pragma once

#include <cstdint>

namespace blackwell_moe {

// Device-side scheduler state. Each CTA claims a contiguous chunk with one
// atomicAdd. The CUDA kernel implementation will consume this interface in
// Phase 4; keeping it here fixes the ABI without claiming a completed kernel.
struct DynamicQueueState {
  std::uint32_t* next_tile = nullptr;
  std::uint32_t tile_count = 0;
  std::uint32_t claim_size = 1;
};

}  // namespace blackwell_moe

