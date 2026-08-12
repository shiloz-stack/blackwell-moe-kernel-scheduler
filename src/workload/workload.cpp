#include "blackwell_moe/workload.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace blackwell_moe {
namespace {

std::vector<double> make_weights(const WorkloadConfig& config) {
  const auto experts = config.experts;
  std::vector<double> weights(experts, 1.0);

  switch (config.distribution) {
    case Distribution::kUniform:
      break;
    case Distribution::kZipf:
      for (std::uint32_t i = 0; i < experts; ++i) {
        weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), config.zipf_alpha);
      }
      break;
    case Distribution::kHeavyHitter: {
      const auto heavy_count = std::max<std::uint32_t>(
          1, static_cast<std::uint32_t>(std::ceil(
                 experts * config.heavy_hitter_fraction)));
      const auto light_count = experts - heavy_count;
      const double heavy_weight = config.heavy_hitter_token_share / heavy_count;
      const double light_weight = light_count == 0
                                      ? 0.0
                                      : (1.0 - config.heavy_hitter_token_share) /
                                            light_count;
      for (std::uint32_t i = 0; i < experts; ++i) {
        weights[i] = i < heavy_count ? heavy_weight : light_weight;
      }
      break;
    }
    case Distribution::kSparse: {
      const auto inactive_count = std::min<std::uint32_t>(
          experts - 1, static_cast<std::uint32_t>(std::floor(
                           experts * config.inactive_expert_fraction)));
      std::fill(weights.begin(), weights.begin() + inactive_count, 0.0);
      break;
    }
  }
  return weights;
}

double coefficient_of_variation(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  if (mean == 0.0) return 0.0;
  double squared_error = 0.0;
  for (double value : values) {
    const double error = value - mean;
    squared_error += error * error;
  }
  return std::sqrt(squared_error / values.size()) / mean;
}

void validate_config(const WorkloadConfig& config) {
  if (config.experts == 0) throw std::invalid_argument("experts must be positive");
  if (config.zipf_alpha <= 0.0) {
    throw std::invalid_argument("zipf_alpha must be positive");
  }
  const auto in_unit_interval = [](double x) { return x >= 0.0 && x <= 1.0; };
  if (!in_unit_interval(config.heavy_hitter_fraction) ||
      config.heavy_hitter_fraction == 0.0 ||
      !in_unit_interval(config.heavy_hitter_token_share) ||
      !in_unit_interval(config.inactive_expert_fraction)) {
    throw std::invalid_argument("distribution fractions must be within [0, 1]");
  }
}

}  // namespace

std::string to_string(Distribution distribution) {
  switch (distribution) {
    case Distribution::kUniform: return "uniform";
    case Distribution::kHeavyHitter: return "heavy_hitter";
    case Distribution::kSparse: return "sparse";
    case Distribution::kZipf: return "zipf";
  }
  throw std::invalid_argument("unknown distribution");
}

Distribution parse_distribution(const std::string& value) {
  if (value == "uniform") return Distribution::kUniform;
  if (value == "heavy_hitter" || value == "heavy") {
    return Distribution::kHeavyHitter;
  }
  if (value == "sparse") return Distribution::kSparse;
  if (value == "zipf") return Distribution::kZipf;
  throw std::invalid_argument("unknown distribution: " + value);
}

Workload generate_workload(const WorkloadConfig& config) {
  validate_config(config);
  auto weights = make_weights(config);
  std::mt19937_64 generator(config.seed);
  std::discrete_distribution<std::uint32_t> choose_expert(
      weights.begin(), weights.end());

  Workload workload{config, std::vector<std::uint32_t>(config.experts, 0)};
  for (std::uint64_t token = 0; token < config.total_tokens; ++token) {
    ++workload.tokens_per_expert[choose_expert(generator)];
  }
  validate_workload(workload);
  return workload;
}

void validate_workload(const Workload& workload) {
  if (workload.tokens_per_expert.size() != workload.config.experts) {
    throw std::invalid_argument("tokens_per_expert size does not match experts");
  }
  const std::uint64_t total = std::accumulate(
      workload.tokens_per_expert.begin(), workload.tokens_per_expert.end(),
      std::uint64_t{0});
  if (total != workload.config.total_tokens) {
    throw std::invalid_argument("tokens_per_expert does not conserve tokens");
  }
}

WorkloadMetrics compute_workload_metrics(
    const std::vector<std::uint32_t>& tokens_per_expert,
    const GemmShape& shape) {
  if (tokens_per_expert.empty()) return {};
  if (shape.tile_m == 0 || shape.tile_n == 0) {
    throw std::invalid_argument("tile dimensions must be positive");
  }

  WorkloadMetrics metrics;
  const std::uint64_t token_sum = std::accumulate(
      tokens_per_expert.begin(), tokens_per_expert.end(), std::uint64_t{0});
  metrics.mean_tokens = static_cast<double>(token_sum) / tokens_per_expert.size();

  std::vector<double> token_values;
  std::vector<double> tile_values;
  token_values.reserve(tokens_per_expert.size());
  tile_values.reserve(tokens_per_expert.size());
  const auto n_tiles = (shape.n + shape.tile_n - 1) / shape.tile_n;
  std::uint32_t inactive = 0;
  std::uint32_t small_m = 0;
  std::uint32_t max_tokens = 0;

  for (const auto tokens : tokens_per_expert) {
    max_tokens = std::max(max_tokens, tokens);
    inactive += tokens == 0 ? 1 : 0;
    small_m += tokens > 0 && tokens < shape.tile_m ? 1 : 0;
    metrics.active_experts += tokens > 0 ? 1 : 0;
    const std::uint64_t m_tiles =
        tokens == 0 ? 0 : (tokens + shape.tile_m - 1) / shape.tile_m;
    const std::uint64_t tiles = m_tiles * n_tiles;
    metrics.total_tiles += tiles;
    token_values.push_back(static_cast<double>(tokens));
    tile_values.push_back(static_cast<double>(tiles));
  }

  metrics.coefficient_of_variation = coefficient_of_variation(token_values);
  metrics.tile_count_cv = coefficient_of_variation(tile_values);
  metrics.max_over_mean = metrics.mean_tokens == 0.0
                              ? 0.0
                              : max_tokens / metrics.mean_tokens;
  metrics.inactive_expert_ratio =
      static_cast<double>(inactive) / tokens_per_expert.size();
  metrics.small_m_expert_ratio =
      static_cast<double>(small_m) / tokens_per_expert.size();
  return metrics;
}

}  // namespace blackwell_moe

