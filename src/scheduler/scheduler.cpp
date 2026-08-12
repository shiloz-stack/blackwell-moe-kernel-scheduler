#include "blackwell_moe/scheduler/scheduler.cuh"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace blackwell_moe {

std::string to_string(SchedulerKind scheduler) {
  switch (scheduler) {
    case SchedulerKind::kExpertOrder: return "expert_order";
    case SchedulerKind::kStaticPersistent: return "static_persistent";
    case SchedulerKind::kDynamicQueue: return "dynamic_queue";
    case SchedulerKind::kCompacted: return "compacted";
    case SchedulerKind::kAuto: return "auto";
  }
  throw std::invalid_argument("unknown scheduler");
}

SchedulerKind parse_scheduler(const std::string& value) {
  if (value == "expert_order" || value == "baseline") {
    return SchedulerKind::kExpertOrder;
  }
  if (value == "static" || value == "static_persistent") {
    return SchedulerKind::kStaticPersistent;
  }
  if (value == "dynamic" || value == "dynamic_queue") {
    return SchedulerKind::kDynamicQueue;
  }
  if (value == "compacted") return SchedulerKind::kCompacted;
  if (value == "auto") return SchedulerKind::kAuto;
  throw std::invalid_argument("unknown scheduler: " + value);
}

std::vector<ExpertTile> build_expert_tiles(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape,
    bool /*compact_inactive*/) {
  if (shape.tile_m == 0 || shape.tile_n == 0) {
    throw std::invalid_argument("tile dimensions must be positive");
  }
  std::vector<ExpertTile> tiles;
  const auto n_tiles = (shape.n + shape.tile_n - 1) / shape.tile_n;
  for (std::uint32_t expert = 0; expert < tokens_per_expert.size(); ++expert) {
    const auto tokens = tokens_per_expert[expert];
    const auto m_tiles = tokens == 0 ? 0 : (tokens + shape.tile_m - 1) / shape.tile_m;
    for (std::uint32_t tile_m = 0; tile_m < m_tiles; ++tile_m) {
      for (std::uint32_t tile_n = 0; tile_n < n_tiles; ++tile_n) {
        tiles.push_back({expert, tile_m, tile_n});
      }
    }
  }
  return tiles;
}

CtaAssignment assign_expert_order(
    const std::vector<ExpertTile>& tiles, std::uint32_t cta_count) {
  if (cta_count == 0) throw std::invalid_argument("cta_count must be positive");
  CtaAssignment assignment{std::vector<std::vector<ExpertTile>>(cta_count)};
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    assignment.cta_tiles[i % cta_count].push_back(tiles[i]);
  }
  return assignment;
}

CtaAssignment assign_static_persistent(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape,
    std::uint32_t cta_count) {
  if (cta_count == 0) throw std::invalid_argument("cta_count must be positive");

  std::vector<std::vector<ExpertTile>> by_expert(tokens_per_expert.size());
  for (const auto& tile : build_expert_tiles(tokens_per_expert, shape)) {
    by_expert[tile.expert_id].push_back(tile);
  }
  std::stable_sort(by_expert.begin(), by_expert.end(),
                   [](const auto& lhs, const auto& rhs) {
                     return lhs.size() > rhs.size();
                   });

  CtaAssignment assignment{std::vector<std::vector<ExpertTile>>(cta_count)};
  std::vector<std::size_t> loads(cta_count, 0);
  for (const auto& expert_tiles : by_expert) {
    for (const auto& tile : expert_tiles) {
      const auto least_loaded = static_cast<std::uint32_t>(
          std::min_element(loads.begin(), loads.end()) - loads.begin());
      assignment.cta_tiles[least_loaded].push_back(tile);
      ++loads[least_loaded];
    }
  }
  return assignment;
}

}  // namespace blackwell_moe

