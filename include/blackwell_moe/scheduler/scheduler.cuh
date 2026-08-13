#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "blackwell_moe/problem.hpp"
#include "blackwell_moe/work_tile.cuh"

namespace blackwell_moe {

enum class SchedulerKind {
  kExpertOrder,
  kStaticPersistent,
  kDynamicQueue,
  kCompacted,
  kAuto,
};

struct CtaAssignment {
  std::vector<std::vector<ExpertTile>> cta_tiles;
};

// CPU-side scheduling measurements. Work is expressed in valid output elements;
// multiplying by 2*K converts it to GEMM FLOPs because K is common to all tiles.
struct ScheduleMetrics {
  std::uint64_t total_tiles = 0;
  std::uint64_t total_valid_elements = 0;
  std::uint64_t total_padded_elements = 0;
  std::uint64_t expert_switches = 0;
  std::uint64_t min_tiles_per_cta = 0;
  std::uint64_t max_tiles_per_cta = 0;
  double mean_cta_work = 0.0;
  double max_cta_work = 0.0;
  double cta_work_cv = 0.0;
  double tail_ratio = 0.0;
  double estimated_utilization = 0.0;
  double useful_work_ratio = 0.0;
};

std::string to_string(SchedulerKind scheduler);
SchedulerKind parse_scheduler(const std::string& value);

std::vector<ExpertTile> build_expert_tiles(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape,
    bool compact_inactive = false);

CtaAssignment assign_expert_order(
    const std::vector<ExpertTile>& tiles, std::uint32_t cta_count);

std::vector<ExpertTile> order_by_descending_expert_tiles(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape);

CtaAssignment assign_static_persistent(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape,
    std::uint32_t cta_count);

CtaAssignment assign_dynamic_queue(
    const std::vector<ExpertTile>& tiles, std::uint32_t cta_count);

CtaAssignment simulate_scheduler(
    SchedulerKind scheduler,
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape,
    std::uint32_t cta_count);

ScheduleMetrics compute_schedule_metrics(
    const CtaAssignment& assignment, const GemmShape& shape);

}  // namespace blackwell_moe
