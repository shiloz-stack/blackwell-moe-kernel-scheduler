#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackwell_moe/scheduler/scheduler.cuh"
#include "blackwell_moe/scheduler/scheduler_probe.cuh"
#include "blackwell_moe/workload.hpp"

namespace {

struct Options {
  blackwell_moe::WorkloadConfig workload;
  blackwell_moe::GemmShape shape;
  blackwell_moe::DeviceSchedulerKind scheduler =
      blackwell_moe::DeviceSchedulerKind::kDynamicQueue;
  std::uint32_t ctas = 120;
  std::uint32_t claim_size = 1;
  std::uint32_t work_scale = 8;
  int warmup = 20;
  int iterations = 200;
  bool csv = false;
};

std::string value(const std::string& argument) {
  const auto equals = argument.find('=');
  if (equals == std::string::npos) {
    throw std::invalid_argument("expected --key=value: " + argument);
  }
  return argument.substr(equals + 1);
}

Options parse_options(int argc, char** argv) {
  using namespace blackwell_moe;
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--csv") {
      options.csv = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "Usage: moe_scheduler_gpu_bench [options]\n"
          << "  --distribution=uniform|heavy_hitter|sparse|zipf\n"
          << "  --scheduler=static|dynamic --experts=N --tokens=N --n=N\n"
          << "  --tile-m=N --tile-n=N --ctas=N --claim-size=N\n"
          << "  --work-scale=N --warmup=N --iterations=N --csv\n";
      std::exit(0);
    } else if (argument.rfind("--distribution=", 0) == 0) {
      options.workload.distribution = parse_distribution(value(argument));
    } else if (argument.rfind("--scheduler=", 0) == 0) {
      const auto name = value(argument);
      if (name == "static" || name == "static_persistent") {
        options.scheduler = DeviceSchedulerKind::kStaticPersistent;
      } else if (name == "dynamic" || name == "dynamic_queue") {
        options.scheduler = DeviceSchedulerKind::kDynamicQueue;
      } else {
        throw std::invalid_argument("unknown device scheduler: " + name);
      }
    } else if (argument.rfind("--experts=", 0) == 0) {
      options.workload.experts = std::stoul(value(argument));
    } else if (argument.rfind("--tokens=", 0) == 0) {
      options.workload.total_tokens = std::stoull(value(argument));
    } else if (argument.rfind("--seed=", 0) == 0) {
      options.workload.seed = std::stoull(value(argument));
    } else if (argument.rfind("--n=", 0) == 0) {
      options.shape.n = std::stoul(value(argument));
    } else if (argument.rfind("--tile-m=", 0) == 0) {
      options.shape.tile_m = std::stoul(value(argument));
    } else if (argument.rfind("--tile-n=", 0) == 0) {
      options.shape.tile_n = std::stoul(value(argument));
    } else if (argument.rfind("--ctas=", 0) == 0) {
      options.ctas = std::stoul(value(argument));
    } else if (argument.rfind("--claim-size=", 0) == 0) {
      options.claim_size = std::stoul(value(argument));
    } else if (argument.rfind("--work-scale=", 0) == 0) {
      options.work_scale = std::stoul(value(argument));
    } else if (argument.rfind("--warmup=", 0) == 0) {
      options.warmup = std::stoi(value(argument));
    } else if (argument.rfind("--iterations=", 0) == 0) {
      options.iterations = std::stoi(value(argument));
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.ctas == 0 || options.claim_size == 0 ||
      options.work_scale == 0 || options.warmup < 0 ||
      options.iterations <= 0) {
    throw std::invalid_argument("invalid scheduler benchmark options");
  }
  return options;
}

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <class T>
struct DeviceArray {
  T* pointer = nullptr;
  ~DeviceArray() { cudaFree(pointer); }
  void allocate(std::size_t count) {
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer),
                            count * sizeof(T)),
                 "cudaMalloc");
  }
};

double percentile(std::vector<float> samples, double probability) {
  std::sort(samples.begin(), samples.end());
  const auto index = static_cast<std::size_t>(
      std::ceil(probability * samples.size()) - 1.0);
  return samples[std::min(index, samples.size() - 1)];
}

struct ObservedBalance {
  double cv = 0.0;
  double tail_ratio = 0.0;
  double utilization = 0.0;
};

ObservedBalance compute_balance(const std::vector<std::uint64_t>& work) {
  ObservedBalance result;
  const double mean = std::accumulate(work.begin(), work.end(), 0.0) /
                      static_cast<double>(work.size());
  const auto maximum = *std::max_element(work.begin(), work.end());
  if (mean == 0.0 || maximum == 0) return result;
  double variance = 0.0;
  for (const auto value : work) {
    const double delta = static_cast<double>(value) - mean;
    variance += delta * delta;
  }
  result.cv = std::sqrt(variance / work.size()) / mean;
  result.tail_ratio = static_cast<double>(maximum) / mean;
  result.utilization = mean / static_cast<double>(maximum);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace blackwell_moe;
  try {
    const auto options = parse_options(argc, argv);
    const auto workload = generate_workload(options.workload);
    auto tiles = options.scheduler == DeviceSchedulerKind::kStaticPersistent
                     ? order_by_descending_expert_tiles(
                           workload.tokens_per_expert, options.shape)
                     : build_expert_tiles(
                           workload.tokens_per_expert, options.shape, true);
    if (tiles.empty()) throw std::runtime_error("workload produced no tiles");

    DeviceArray<ExpertTile> device_tiles;
    DeviceArray<std::uint32_t> next_tile;
    DeviceArray<std::uint32_t> visits;
    DeviceArray<std::uint64_t> checksums;
    DeviceArray<std::uint32_t> cta_tiles;
    DeviceArray<std::uint64_t> cta_work;
    device_tiles.allocate(tiles.size());
    next_tile.allocate(1);
    visits.allocate(tiles.size());
    checksums.allocate(tiles.size());
    cta_tiles.allocate(options.ctas);
    cta_work.allocate(options.ctas);
    require_cuda(cudaMemcpy(device_tiles.pointer, tiles.data(),
                            tiles.size() * sizeof(ExpertTile),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(tiles)");
    require_cuda(cudaMemset(visits.pointer, 0,
                            tiles.size() * sizeof(std::uint32_t)),
                 "cudaMemset(visits)");

    SchedulerProbeArguments arguments;
    arguments.tiles = device_tiles.pointer;
    arguments.tile_count = static_cast<std::uint32_t>(tiles.size());
    arguments.cta_count = options.ctas;
    arguments.claim_size = options.claim_size;
    arguments.work_scale = options.work_scale;
    arguments.next_tile = next_tile.pointer;
    arguments.tile_visits = visits.pointer;
    arguments.tile_checksums = checksums.pointer;
    arguments.cta_tile_counts = cta_tiles.pointer;
    arguments.cta_valid_elements = cta_work.pointer;

    auto launch = [&]() {
      require_cuda(cudaMemsetAsync(next_tile.pointer, 0, sizeof(std::uint32_t)),
                   "cudaMemsetAsync(next_tile)");
      const auto status = launch_scheduler_probe(options.scheduler, arguments);
      if (status != KernelStatus::kSuccess) {
        throw std::runtime_error(kernel_status_string(status));
      }
    };
    for (int i = 0; i < options.warmup; ++i) launch();
    require_cuda(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    require_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    std::vector<float> samples;
    samples.reserve(options.iterations);
    for (int i = 0; i < options.iterations; ++i) {
      require_cuda(cudaMemsetAsync(next_tile.pointer, 0, sizeof(std::uint32_t)),
                   "cudaMemsetAsync(next_tile)");
      require_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
      const auto status = launch_scheduler_probe(options.scheduler, arguments);
      if (status != KernelStatus::kSuccess) {
        throw std::runtime_error(kernel_status_string(status));
      }
      require_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
      require_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
      float elapsed_ms = 0.0f;
      require_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "cudaEventElapsedTime");
      samples.push_back(elapsed_ms);
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    std::vector<std::uint64_t> host_cta_work(options.ctas);
    require_cuda(cudaMemcpy(host_cta_work.data(), cta_work.pointer,
                            host_cta_work.size() * sizeof(std::uint64_t),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(cta_work)");
    const auto balance = compute_balance(host_cta_work);
    const auto median_ms = percentile(samples, 0.5);
    const auto p95_ms = percentile(samples, 0.95);
    if (options.csv) {
      std::cout << "distribution,scheduler,experts,tokens,tiles,ctas,claim_size,"
                   "work_scale,median_ms,p95_ms,observed_cta_work_cv,tail_ratio,"
                   "observed_utilization\n"
                << to_string(options.workload.distribution) << ','
                << device_scheduler_name(options.scheduler) << ','
                << options.workload.experts << ','
                << options.workload.total_tokens << ',' << tiles.size() << ','
                << options.ctas << ',' << options.claim_size << ','
                << options.work_scale << ',' << median_ms << ',' << p95_ms
                << ',' << balance.cv << ',' << balance.tail_ratio << ','
                << balance.utilization << '\n';
    } else {
      std::cout << "distribution=" << to_string(options.workload.distribution)
                << " scheduler=" << device_scheduler_name(options.scheduler)
                << " tiles=" << tiles.size() << " median_ms=" << median_ms
                << " p95_ms=" << p95_ms << " cta_work_cv=" << balance.cv
                << " tail_ratio=" << balance.tail_ratio
                << " utilization=" << balance.utilization << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GPU scheduler benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
