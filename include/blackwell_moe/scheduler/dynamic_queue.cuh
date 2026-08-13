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

#if defined(__CUDACC__)
// Returns the first tile in a claimed half-open interval.  The caller clamps
// begin + claim_size to tile_count.  Keeping this primitive in the scheduler
// ABI lets the GEMM kernel and the standalone scheduler benchmark share the
// exact same queue semantics.
__device__ __forceinline__ std::uint32_t claim_dynamic_tiles(
    const DynamicQueueState& queue) {
  return atomicAdd(queue.next_tile, queue.claim_size);
}
#endif

}  // namespace blackwell_moe
