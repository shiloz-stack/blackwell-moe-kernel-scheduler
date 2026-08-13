#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>
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

std::vector<blackwell_moe::ExpertTile> flatten(
    const blackwell_moe::CtaAssignment& assignment) {
  std::vector<blackwell_moe::ExpertTile> result;
  for (const auto& cta : assignment.cta_tiles) {
    result.insert(result.end(), cta.begin(), cta.end());
  }
  return result;
}

void require_same_tiles(std::vector<blackwell_moe::ExpertTile> actual,
                        std::vector<blackwell_moe::ExpertTile> expected,
                        const char* message) {
  const auto key = [](const auto& tile) {
    return std::tie(tile.expert_id, tile.tile_m, tile.tile_n, tile.valid_m,
                    tile.valid_n);
  };
  const auto less = [&key](const auto& lhs, const auto& rhs) {
    return key(lhs) < key(rhs);
  };
  std::sort(actual.begin(), actual.end(), less);
  std::sort(expected.begin(), expected.end(), less);
  require(actual == expected, message);
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
    const auto expert_three_tail = std::find_if(
        tiles.begin(), tiles.end(), [](const auto& tile) {
          return tile.expert_id == 3 && tile.tile_m == 1 && tile.tile_n == 0;
        });
    require(expert_three_tail != tiles.end(), "missing expert boundary tile");
    require(expert_three_tail->valid_m == 1 &&
                expert_three_tail->valid_n == 128,
            "expert boundary dimensions are incorrect");
    require(valid_output_elements(*expert_three_tail) == 128,
            "valid output element count is incorrect");

    const auto baseline = assign_expert_order(tiles, 3);
    require(assigned_tiles(baseline) == tiles.size(),
            "expert-order assignment lost work");
    require_same_tiles(flatten(baseline), tiles,
                       "expert-order assignment changed tile coverage");

    const auto persistent = assign_static_persistent(tokens, shape, 3);
    require(assigned_tiles(persistent) == tiles.size(),
            "static persistent assignment lost work");
    require_same_tiles(flatten(persistent), tiles,
                       "static persistent assignment changed tile coverage");
    std::size_t min_load = persistent.cta_tiles.front().size();
    std::size_t max_load = min_load;
    for (const auto& cta : persistent.cta_tiles) {
      min_load = std::min(min_load, cta.size());
      max_load = std::max(max_load, cta.size());
    }
    require(max_load - min_load <= 1,
            "static persistent host assignment should balance tile counts");

    const auto ordered = order_by_descending_expert_tiles(tokens, shape);
    require(!ordered.empty() && ordered.front().expert_id == 3,
            "largest expert should appear first in the static tile order");

    const auto dynamic = assign_dynamic_queue(tiles, 3);
    require_same_tiles(flatten(dynamic), tiles,
                       "dynamic queue assignment changed tile coverage");
    require(dynamic.cta_tiles == assign_dynamic_queue(tiles, 3).cta_tiles,
            "dynamic queue simulation must be deterministic");

    GemmShape boundary_shape;
    boundary_shape.n = 129;
    boundary_shape.tile_m = 128;
    boundary_shape.tile_n = 128;
    const auto boundary_tiles =
        build_expert_tiles(std::vector<std::uint32_t>{129}, boundary_shape);
    const auto boundary_assignment = assign_expert_order(boundary_tiles, 2);
    const auto schedule_metrics =
        compute_schedule_metrics(boundary_assignment, boundary_shape);
    require(schedule_metrics.total_tiles == 4,
            "schedule metric tile count mismatch");
    require(schedule_metrics.total_valid_elements == 129ull * 129ull,
            "schedule metric valid element count mismatch");
    require(schedule_metrics.total_padded_elements == 4ull * 128ull * 128ull,
            "schedule metric padded element count mismatch");
    require(schedule_metrics.useful_work_ratio > 0.25 &&
                schedule_metrics.useful_work_ratio < 0.26,
            "schedule metric useful-work ratio mismatch");
    require(schedule_metrics.tail_ratio > 1.0 &&
                schedule_metrics.estimated_utilization > 0.0 &&
                schedule_metrics.estimated_utilization <= 1.0,
            "schedule imbalance metrics are outside their valid ranges");

    // A non-divisible boundary-heavy case verifies that the ideal dynamic
    // queue can react to unequal tile work instead of reproducing grid stride.
    const std::vector<std::uint32_t> irregular_tokens{1, 257, 129};
    const auto irregular_tiles =
        build_expert_tiles(irregular_tokens, boundary_shape);
    const auto irregular_static_metrics = compute_schedule_metrics(
        assign_expert_order(irregular_tiles, 3), boundary_shape);
    const auto irregular_dynamic_metrics = compute_schedule_metrics(
        assign_dynamic_queue(irregular_tiles, 3), boundary_shape);
    require(irregular_dynamic_metrics.tail_ratio <
                irregular_static_metrics.tail_ratio,
            "dynamic reference should reduce tail in the irregular test case");

    bool rejected_zero_ctas = false;
    try {
      (void)assign_dynamic_queue(tiles, 0);
    } catch (const std::invalid_argument&) {
      rejected_zero_ctas = true;
    }
    require(rejected_zero_ctas, "dynamic queue must reject zero CTAs");
    std::cout << "scheduler tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "scheduler test failed: " << error.what() << '\n';
    return 1;
  }
}
