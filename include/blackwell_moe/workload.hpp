#pragma once

#include <cstdint>
#include <vector>

#include "blackwell_moe/problem.hpp"

namespace blackwell_moe {

struct WorkloadMetrics {
  double mean_tokens = 0.0;
  double coefficient_of_variation = 0.0;
  double max_over_mean = 0.0;
  double inactive_expert_ratio = 0.0;
  double small_m_expert_ratio = 0.0;
  double tile_count_cv = 0.0;
  std::uint32_t active_experts = 0;
  std::uint64_t total_tiles = 0;
};

Workload generate_workload(const WorkloadConfig& config);

WorkloadMetrics compute_workload_metrics(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape);

void validate_workload(const Workload& workload);

}  // namespace blackwell_moe

