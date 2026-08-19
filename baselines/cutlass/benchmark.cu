#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackwell_moe/kernels/grouped_gemm.cuh"
#include "blackwell_moe/workload.hpp"

#ifndef BLACKWELL_MOE_CUTLASS_REVISION
#define BLACKWELL_MOE_CUTLASS_REVISION "unknown"
#endif

namespace {

struct Options {
  blackwell_moe::WorkloadConfig workload;
  std::uint32_t n = 2048;
  std::uint32_t k = 1024;
  int warmup = 5;
  int iterations = 20;
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
  Options options;
  options.workload.experts = 8;
  options.workload.total_tokens = 1024;
  options.workload.distribution = blackwell_moe::Distribution::kZipf;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--csv") {
      options.csv = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "Usage: moe_cutlass_baseline_bench [options]\n"
          << "  --distribution=uniform|heavy_hitter|sparse|zipf\n"
          << "  --experts=N --tokens=N --n=N --k=N --seed=N\n"
          << "  --warmup=N --iterations=N --csv\n";
      std::exit(0);
    } else if (argument.rfind("--distribution=", 0) == 0) {
      options.workload.distribution =
          blackwell_moe::parse_distribution(value(argument));
    } else if (argument.rfind("--experts=", 0) == 0) {
      options.workload.experts = std::stoul(value(argument));
    } else if (argument.rfind("--tokens=", 0) == 0) {
      options.workload.total_tokens = std::stoull(value(argument));
    } else if (argument.rfind("--seed=", 0) == 0) {
      options.workload.seed = std::stoull(value(argument));
    } else if (argument.rfind("--n=", 0) == 0) {
      options.n = std::stoul(value(argument));
    } else if (argument.rfind("--k=", 0) == 0) {
      options.k = std::stoul(value(argument));
    } else if (argument.rfind("--warmup=", 0) == 0) {
      options.warmup = std::stoi(value(argument));
    } else if (argument.rfind("--iterations=", 0) == 0) {
      options.iterations = std::stoi(value(argument));
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.workload.total_tokens == 0 || options.n == 0 || options.k == 0 ||
      options.n % 8 != 0 ||
      options.k % 8 != 0 || options.warmup < 0 || options.iterations <= 0) {
    throw std::invalid_argument(
        "tokens must be positive, N/K must be positive multiples of 8, and "
        "iteration counts must be valid");
  }
  return options;
}

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

struct DeviceBuffers {
  std::vector<void*> allocations;
  ~DeviceBuffers() {
    for (void* allocation : allocations) cudaFree(allocation);
  }

  void* allocate(std::size_t bytes, bool zero = false) {
    void* pointer = nullptr;
    require_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
    allocations.push_back(pointer);
    if (zero) require_cuda(cudaMemset(pointer, 0, bytes), "cudaMemset");
    return pointer;
  }
};

__global__ void fill_bf16(__nv_bfloat16* data, std::size_t count,
                          std::uint32_t seed) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    std::uint32_t bits = static_cast<std::uint32_t>(index) ^ seed;
    bits ^= bits >> 16;
    bits *= 0x7feb352dU;
    bits ^= bits >> 15;
    const float value = static_cast<float>(static_cast<int>(bits % 33) - 16) /
                        16.0f;
    data[index] = __float2bfloat16(value);
  }
}

void initialize_bf16(void* pointer, std::size_t elements, std::uint32_t seed) {
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(
      std::min<std::size_t>(4096, (elements + kThreads - 1) / kThreads));
  fill_bf16<<<std::max(1, blocks), kThreads>>>(
      static_cast<__nv_bfloat16*>(pointer), elements, seed);
  require_cuda(cudaPeekAtLastError(), "fill_bf16 launch");
}

double percentile(std::vector<float> values, double probability) {
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::ceil(probability * values.size()) - 1.0);
  return values[std::min(index, values.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
  using namespace blackwell_moe;
  try {
    const auto options = parse_options(argc, argv);
    const auto workload = generate_workload(options.workload);

    int device = 0;
    int runtime_version = 0;
    cudaDeviceProp properties{};
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    require_cuda(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties");
    require_cuda(cudaRuntimeGetVersion(&runtime_version),
                 "cudaRuntimeGetVersion");

    DeviceBuffers storage;
    std::vector<const void*> a(options.workload.experts, nullptr);
    std::vector<const void*> b(options.workload.experts, nullptr);
    std::vector<void*> d(options.workload.experts, nullptr);
    std::uint64_t total_fmas = 0;
    constexpr std::size_t kBf16Bytes = 2;
    for (std::uint32_t expert = 0; expert < options.workload.experts; ++expert) {
      const auto m = workload.tokens_per_expert[expert];
      if (m == 0) continue;
      const auto a_elements = static_cast<std::size_t>(m) * options.k;
      const auto b_elements =
          static_cast<std::size_t>(options.n) * options.k;
      const auto d_elements = static_cast<std::size_t>(m) * options.n;
      a[expert] = storage.allocate(a_elements * kBf16Bytes);
      b[expert] = storage.allocate(b_elements * kBf16Bytes);
      d[expert] = storage.allocate(d_elements * kBf16Bytes, true);
      initialize_bf16(const_cast<void*>(a[expert]), a_elements,
                      options.workload.seed + expert * 2);
      initialize_bf16(const_cast<void*>(b[expert]), b_elements,
                      options.workload.seed + expert * 2 + 1);
      total_fmas += static_cast<std::uint64_t>(m) * options.n * options.k;
    }

    GroupedGemmArguments args;
    args.a = a.data();
    args.b = b.data();
    args.d = d.data();
    args.tokens_per_expert = workload.tokens_per_expert.data();
    args.experts = options.workload.experts;
    args.n = options.n;
    args.k = options.k;

    Bf16GroupedGemmPlan plan;
    auto status = plan.initialize(args);
    if (status != KernelStatus::kSuccess) {
      throw std::runtime_error(std::string("plan initialization: ") +
                               kernel_status_string(status));
    }
    for (int i = 0; i < options.warmup; ++i) {
      status = plan.run();
      if (status != KernelStatus::kSuccess) {
        throw std::runtime_error(kernel_status_string(status));
      }
    }
    require_cuda(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    require_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    std::vector<float> samples;
    samples.reserve(options.iterations);
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      require_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
      status = plan.run();
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

    const double median_ms = percentile(samples, 0.5);
    const double p95_ms = percentile(samples, 0.95);
    const double tflops = 2.0 * static_cast<double>(total_fmas) /
                          (median_ms * 1.0e9);
    const auto plan_stats = plan.stats();

    if (options.csv) {
      std::cout
          << "device,compute_capability,cuda_runtime,cutlass,distribution,"
             "experts,active_experts,tokens,n,k,warmup,iterations,median_ms,"
             "p95_ms,tflops,threadblocks,workspace_bytes\n";
      std::cout << properties.name << ',' << properties.major << '.'
                << properties.minor << ',' << runtime_version << ','
                << BLACKWELL_MOE_CUTLASS_REVISION << ','
                << to_string(options.workload.distribution) << ','
                << options.workload.experts << ',' << plan_stats.active_experts
                << ',' << options.workload.total_tokens << ',' << options.n
                << ',' << options.k << ',' << options.warmup << ','
                << options.iterations << ',' << median_ms << ',' << p95_ms
                << ',' << tflops << ',' << plan_stats.threadblock_count << ','
                << plan_stats.workspace_bytes << '\n';
    } else {
      std::cout << std::fixed << std::setprecision(4)
                << "device:             " << properties.name << '\n'
                << "compute capability: " << properties.major << '.'
                << properties.minor << '\n'
                << "CUDA runtime:       " << runtime_version << '\n'
                << "CUTLASS:            " << BLACKWELL_MOE_CUTLASS_REVISION
                << '\n'
                << "distribution:       "
                << to_string(options.workload.distribution) << '\n'
                << "active experts:     " << plan_stats.active_experts << '/'
                << options.workload.experts << '\n'
                << "shape:              [M_i," << options.n << ',' << options.k
                << "]\n"
                << "median latency:     " << median_ms << " ms\n"
                << "p95 latency:        " << p95_ms << " ms\n"
                << "throughput:         " << tflops << " TFLOP/s\n"
                << "threadblocks:       " << plan_stats.threadblock_count
                << '\n'
                << "workspace:          " << plan_stats.workspace_bytes
                << " bytes\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GPU benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
