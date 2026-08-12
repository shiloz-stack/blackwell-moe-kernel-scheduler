#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define BLACKWELL_MOE_HOST_DEVICE __host__ __device__
#else
#define BLACKWELL_MOE_HOST_DEVICE
#endif

namespace blackwell_moe {

struct ExpertTile {
  std::uint32_t expert_id;
  std::uint32_t tile_m;
  std::uint32_t tile_n;
};

BLACKWELL_MOE_HOST_DEVICE inline bool operator==(
    const ExpertTile& lhs, const ExpertTile& rhs) {
  return lhs.expert_id == rhs.expert_id && lhs.tile_m == rhs.tile_m &&
         lhs.tile_n == rhs.tile_n;
}

}  // namespace blackwell_moe

#undef BLACKWELL_MOE_HOST_DEVICE

