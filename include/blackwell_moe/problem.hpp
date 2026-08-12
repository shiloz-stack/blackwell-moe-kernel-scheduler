#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace blackwell_moe {

enum class Distribution {
  kUniform,
  kHeavyHitter,
  kSparse,
  kZipf,
};

struct WorkloadConfig {
  std::uint32_t experts = 64;
  std::uint64_t total_tokens = 4096;
  Distribution distribution = Distribution::kUniform;
  std::uint64_t seed = 2026;
  double zipf_alpha = 1.2;
  double heavy_hitter_fraction = 0.1;
  double heavy_hitter_token_share = 0.8;
  double inactive_expert_fraction = 0.5;
};

struct Workload {
  WorkloadConfig config;
  std::vector<std::uint32_t> tokens_per_expert;
};

struct GemmShape {
  std::uint32_t n = 7168;
  std::uint32_t k = 2048;
  std::uint32_t tile_m = 128;
  std::uint32_t tile_n = 128;
};

std::string to_string(Distribution distribution);
Distribution parse_distribution(const std::string& value);

}  // namespace blackwell_moe

