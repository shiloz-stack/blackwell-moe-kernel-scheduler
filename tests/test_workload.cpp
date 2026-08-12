#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "blackwell_moe/workload.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main() {
  using namespace blackwell_moe;
  try {
    for (const auto distribution : {Distribution::kUniform,
                                    Distribution::kHeavyHitter,
                                    Distribution::kSparse,
                                    Distribution::kZipf}) {
      WorkloadConfig config;
      config.experts = 32;
      config.total_tokens = 10000;
      config.distribution = distribution;
      const auto first = generate_workload(config);
      const auto second = generate_workload(config);
      require(first.tokens_per_expert == second.tokens_per_expert,
              "generation must be deterministic for a fixed seed");
      require(first.tokens_per_expert.size() == config.experts,
              "expert count mismatch");
      require(std::accumulate(first.tokens_per_expert.begin(),
                              first.tokens_per_expert.end(), std::uint64_t{0}) ==
                  config.total_tokens,
              "token conservation failed");
    }

    WorkloadConfig uniform;
    uniform.experts = 32;
    uniform.total_tokens = 100000;
    uniform.distribution = Distribution::kUniform;
    WorkloadConfig heavy = uniform;
    heavy.distribution = Distribution::kHeavyHitter;
    const GemmShape shape;
    const auto uniform_metrics = compute_workload_metrics(
        generate_workload(uniform).tokens_per_expert, shape);
    const auto heavy_metrics = compute_workload_metrics(
        generate_workload(heavy).tokens_per_expert, shape);
    require(heavy_metrics.coefficient_of_variation >
                uniform_metrics.coefficient_of_variation,
            "heavy-hitter workload should be more skewed than uniform");

    const auto exact_metrics = compute_workload_metrics({0, 64, 128, 256}, shape);
    require(std::abs(exact_metrics.inactive_expert_ratio - 0.25) < 1e-12,
            "inactive expert ratio mismatch");
    require(std::abs(exact_metrics.small_m_expert_ratio - 0.25) < 1e-12,
            "small-M expert ratio mismatch");
    std::cout << "workload tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "workload test failed: " << error.what() << '\n';
    return 1;
  }
}

