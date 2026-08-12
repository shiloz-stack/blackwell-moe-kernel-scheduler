#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "blackwell_moe/scheduler/scheduler.cuh"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::size_t assigned_tiles(const blackwell_moe::CtaAssignment& assignment) {
  std::size_t total = 0;
  for (const auto& cta : assignment.cta_tiles) total += cta.size();
  return total;
}
}

int main() {
  using namespace blackwell_moe;
  try {
    GemmShape shape;
    shape.n = 256;
    shape.tile_m = 128;
    shape.tile_n = 128;
    const std::vector<std::uint32_t> tokens{0, 1, 128, 129};
    const auto tiles = build_expert_tiles(tokens, shape);
    require(tiles.size() == 8, "unexpected expert tile count");
    require(std::none_of(tiles.begin(), tiles.end(),
                         [](const auto& tile) { return tile.expert_id == 0; }),
            "inactive experts must not create tiles");

    const auto baseline = assign_expert_order(tiles, 3);
    require(assigned_tiles(baseline) == tiles.size(),
            "expert-order assignment lost work");

    const auto persistent = assign_static_persistent(tokens, shape, 3);
    require(assigned_tiles(persistent) == tiles.size(),
            "static persistent assignment lost work");
    std::size_t min_load = persistent.cta_tiles.front().size();
    std::size_t max_load = min_load;
    for (const auto& cta : persistent.cta_tiles) {
      min_load = std::min(min_load, cta.size());
      max_load = std::max(max_load, cta.size());
    }
    require(max_load - min_load <= 1,
            "static persistent host assignment should balance tile counts");
    std::cout << "scheduler tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "scheduler test failed: " << error.what() << '\n';
    return 1;
  }
}

