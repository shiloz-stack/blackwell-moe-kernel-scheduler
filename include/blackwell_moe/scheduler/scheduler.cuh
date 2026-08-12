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

std::string to_string(SchedulerKind scheduler);
SchedulerKind parse_scheduler(const std::string& value);

std::vector<ExpertTile> build_expert_tiles(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape,
    bool compact_inactive = false);

CtaAssignment assign_expert_order(
    const std::vector<ExpertTile>& tiles, std::uint32_t cta_count);

CtaAssignment assign_static_persistent(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape,
    std::uint32_t cta_count);

}  // namespace blackwell_moe

