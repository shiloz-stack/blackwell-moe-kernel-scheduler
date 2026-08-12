#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "blackwell_moe/scheduler/scheduler.cuh"
#include "blackwell_moe/workload.hpp"

namespace {

std::string value_after_equals(const std::string& argument) {
  const auto position = argument.find('=');
  if (position == std::string::npos) {
    throw std::invalid_argument("expected --key=value, got: " + argument);
  }
  return argument.substr(position + 1);
}

void print_help() {
  std::cout
      << "Usage: moe_workload_bench [options]\n"
      << "  --distribution=uniform|heavy_hitter|sparse|zipf\n"
      << "  --scheduler=expert_order|static_persistent|dynamic_queue|compacted|auto\n"
      << "  --experts=N --tokens=N --n=N --k=N --tile-m=N --tile-n=N\n"
      << "  --seed=N --csv\n\n"
      << "Phase 1 measures workload and host scheduling metadata only; it does not\n"
      << "report GPU kernel latency.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    blackwell_moe::WorkloadConfig config;
    blackwell_moe::GemmShape shape;
    auto scheduler = blackwell_moe::SchedulerKind::kExpertOrder;
    bool csv = false;

    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument == "--help" || argument == "-h") {
        print_help();
        return EXIT_SUCCESS;
      } else if (argument == "--csv") {
        csv = true;
      } else if (argument.rfind("--distribution=", 0) == 0) {
        config.distribution =
            blackwell_moe::parse_distribution(value_after_equals(argument));
      } else if (argument.rfind("--scheduler=", 0) == 0) {
        scheduler = blackwell_moe::parse_scheduler(value_after_equals(argument));
      } else if (argument.rfind("--experts=", 0) == 0) {
        config.experts = std::stoul(value_after_equals(argument));
      } else if (argument.rfind("--tokens=", 0) == 0) {
        config.total_tokens = std::stoull(value_after_equals(argument));
      } else if (argument.rfind("--seed=", 0) == 0) {
        config.seed = std::stoull(value_after_equals(argument));
      } else if (argument.rfind("--n=", 0) == 0) {
        shape.n = std::stoul(value_after_equals(argument));
      } else if (argument.rfind("--k=", 0) == 0) {
        shape.k = std::stoul(value_after_equals(argument));
      } else if (argument.rfind("--tile-m=", 0) == 0) {
        shape.tile_m = std::stoul(value_after_equals(argument));
      } else if (argument.rfind("--tile-n=", 0) == 0) {
        shape.tile_n = std::stoul(value_after_equals(argument));
      } else {
        throw std::invalid_argument("unknown argument: " + argument);
      }
    }

    const auto workload = blackwell_moe::generate_workload(config);
    const auto metrics = blackwell_moe::compute_workload_metrics(
        workload.tokens_per_expert, shape);
    const auto tiles = blackwell_moe::build_expert_tiles(
        workload.tokens_per_expert, shape,
        scheduler == blackwell_moe::SchedulerKind::kCompacted);

    if (csv) {
      std::cout << "distribution,scheduler,experts,total_tokens,active_experts,"
                   "cv_m,max_over_mean,inactive_ratio,small_m_ratio,tile_cv,"
                   "total_tiles,n,k,tile_m,tile_n,seed\n";
      std::cout << blackwell_moe::to_string(config.distribution) << ','
                << blackwell_moe::to_string(scheduler) << ',' << config.experts
                << ',' << config.total_tokens << ',' << metrics.active_experts
                << ',' << metrics.coefficient_of_variation << ','
                << metrics.max_over_mean << ',' << metrics.inactive_expert_ratio
                << ',' << metrics.small_m_expert_ratio << ','
                << metrics.tile_count_cv << ',' << tiles.size() << ',' << shape.n
                << ',' << shape.k << ',' << shape.tile_m << ',' << shape.tile_n
                << ',' << config.seed << '\n';
    } else {
      std::cout << std::fixed << std::setprecision(4)
                << "distribution:       " << blackwell_moe::to_string(config.distribution) << '\n'
                << "scheduler:          " << blackwell_moe::to_string(scheduler) << '\n'
                << "experts:            " << config.experts << '\n'
                << "total tokens:       " << config.total_tokens << '\n'
                << "active experts:     " << metrics.active_experts << '\n'
                << "CV(M):              " << metrics.coefficient_of_variation << '\n'
                << "max(M) / mean(M):   " << metrics.max_over_mean << '\n'
                << "inactive ratio:     " << metrics.inactive_expert_ratio << '\n'
                << "small-M ratio:      " << metrics.small_m_expert_ratio << '\n'
                << "tile-count CV:      " << metrics.tile_count_cv << '\n'
                << "expert tiles:       " << tiles.size() << '\n';
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

