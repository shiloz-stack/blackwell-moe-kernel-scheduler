#include "blackwell_moe/scheduler/scheduler.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
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
        const auto m_begin = tile_m * shape.tile_m;
        const auto n_begin = tile_n * shape.tile_n;
        const auto valid_m = std::min(shape.tile_m, tokens - m_begin);
        const auto valid_n = std::min(shape.tile_n, shape.n - n_begin);
        tiles.push_back({expert, tile_m, tile_n, valid_m, valid_n});
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
  return assign_expert_order(
      order_by_descending_expert_tiles(tokens_per_expert, shape), cta_count);
}

std::vector<ExpertTile> order_by_descending_expert_tiles(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape) {
  auto tiles = build_expert_tiles(tokens_per_expert, shape);
  std::vector<std::vector<ExpertTile>> by_expert(tokens_per_expert.size());
  for (const auto& tile : tiles) {
    by_expert[tile.expert_id].push_back(tile);
  }

  std::vector<std::uint32_t> expert_order(tokens_per_expert.size());
  std::iota(expert_order.begin(), expert_order.end(), 0);
  std::stable_sort(expert_order.begin(), expert_order.end(),
                   [&by_expert](std::uint32_t lhs, std::uint32_t rhs) {
                     return by_expert[lhs].size() > by_expert[rhs].size();
                   });

  std::vector<ExpertTile> ordered;
  ordered.reserve(tiles.size());
  for (const auto expert : expert_order) {
    ordered.insert(ordered.end(), by_expert[expert].begin(),
                   by_expert[expert].end());
  }
  return ordered;
}

CtaAssignment assign_dynamic_queue(
    const std::vector<ExpertTile>& tiles, std::uint32_t cta_count) {
  if (cta_count == 0) throw std::invalid_argument("cta_count must be positive");

  CtaAssignment assignment{std::vector<std::vector<ExpertTile>>(cta_count)};
  std::vector<std::uint64_t> available_at(cta_count, 0);
  for (const auto& tile : tiles) {
    const auto next_cta = static_cast<std::uint32_t>(
        std::min_element(available_at.begin(), available_at.end()) -
        available_at.begin());
    assignment.cta_tiles[next_cta].push_back(tile);
    // This is an idealized reference cost. It intentionally excludes atomic
    // claim overhead, which must be measured by the future CUDA implementation.
    available_at[next_cta] += std::max<std::uint64_t>(
        1, valid_output_elements(tile));
  }
  return assignment;
}

CtaAssignment simulate_scheduler(
    SchedulerKind scheduler,
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape,
    std::uint32_t cta_count) {
  const auto tiles = build_expert_tiles(tokens_per_expert, shape,
                                        scheduler == SchedulerKind::kCompacted);
  switch (scheduler) {
    case SchedulerKind::kExpertOrder:
    case SchedulerKind::kCompacted:
      return assign_expert_order(tiles, cta_count);
    case SchedulerKind::kStaticPersistent:
      return assign_static_persistent(tokens_per_expert, shape, cta_count);
    case SchedulerKind::kDynamicQueue:
      return assign_dynamic_queue(tiles, cta_count);
    case SchedulerKind::kAuto:
      // Auto-dispatch thresholds require real GPU measurements. Until Phase 5,
      // preserve correctness by falling back to the baseline traversal.
      return assign_expert_order(tiles, cta_count);
  }
  throw std::invalid_argument("unknown scheduler");
}

ScheduleMetrics compute_schedule_metrics(
    const CtaAssignment& assignment, const GemmShape& shape) {
  if (shape.tile_m == 0 || shape.tile_n == 0) {
    throw std::invalid_argument("tile dimensions must be positive");
  }

  ScheduleMetrics metrics;
  if (assignment.cta_tiles.empty()) return metrics;

  metrics.min_tiles_per_cta = std::numeric_limits<std::uint64_t>::max();
  std::vector<double> cta_work;
  cta_work.reserve(assignment.cta_tiles.size());

  const auto padded_elements_per_tile =
      static_cast<std::uint64_t>(shape.tile_m) * shape.tile_n;
  for (const auto& cta : assignment.cta_tiles) {
    const auto tile_count = static_cast<std::uint64_t>(cta.size());
    metrics.total_tiles += tile_count;
    metrics.min_tiles_per_cta =
        std::min(metrics.min_tiles_per_cta, tile_count);
    metrics.max_tiles_per_cta =
        std::max(metrics.max_tiles_per_cta, tile_count);

    std::uint64_t valid_elements = 0;
    for (std::size_t i = 0; i < cta.size(); ++i) {
      valid_elements += valid_output_elements(cta[i]);
      if (i > 0 && cta[i - 1].expert_id != cta[i].expert_id) {
        ++metrics.expert_switches;
      }
    }
    metrics.total_valid_elements += valid_elements;
    cta_work.push_back(static_cast<double>(valid_elements));
  }

  metrics.total_padded_elements =
      metrics.total_tiles * padded_elements_per_tile;
  metrics.mean_cta_work =
      std::accumulate(cta_work.begin(), cta_work.end(), 0.0) /
      static_cast<double>(cta_work.size());
  metrics.max_cta_work =
      *std::max_element(cta_work.begin(), cta_work.end());

  if (metrics.mean_cta_work > 0.0) {
    double squared_deviation = 0.0;
    for (const auto work : cta_work) {
      const auto delta = work - metrics.mean_cta_work;
      squared_deviation += delta * delta;
    }
    const auto standard_deviation =
        std::sqrt(squared_deviation / static_cast<double>(cta_work.size()));
    metrics.cta_work_cv = standard_deviation / metrics.mean_cta_work;
    metrics.tail_ratio = metrics.max_cta_work / metrics.mean_cta_work;
  }
  if (metrics.max_cta_work > 0.0) {
    metrics.estimated_utilization =
        static_cast<double>(metrics.total_valid_elements) /
        (static_cast<double>(cta_work.size()) * metrics.max_cta_work);
  }
  if (metrics.total_padded_elements > 0) {
    metrics.useful_work_ratio =
        static_cast<double>(metrics.total_valid_elements) /
        static_cast<double>(metrics.total_padded_elements);
  }
  return metrics;
}

}  // namespace blackwell_moe
