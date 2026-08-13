#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackwell_moe/kernels/direct_cute_gemm.cuh"

#ifndef BLACKWELL_MOE_CUTLASS_REVISION
#define BLACKWELL_MOE_CUTLASS_REVISION "unknown"
#endif

namespace {

struct Options {
  std::uint32_t m = 128;
  std::uint32_t n = 7168;
  std::uint32_t k = 2048;
  std::uint32_t seed = 2026;
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
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--csv") {
      options.csv = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: direct_cute_bench [options]\n"
                << "  --m=N --n=N --k=N --seed=N\n"
                << "  --warmup=N --iterations=N --csv\n"
                << "  M/N/K must be divisible by 128/256/64.\n";
      std::exit(0);
    } else if (argument.rfind("--m=", 0) == 0) {
      options.m = std::stoul(value(argument));
    } else if (argument.rfind("--n=", 0) == 0) {
      options.n = std::stoul(value(argument));
    } else if (argument.rfind("--k=", 0) == 0) {
      options.k = std::stoul(value(argument));
    } else if (argument.rfind("--seed=", 0) == 0) {
      options.seed = std::stoul(value(argument));
    } else if (argument.rfind("--warmup=", 0) == 0) {
      options.warmup = std::stoi(value(argument));
    } else if (argument.rfind("--iterations=", 0) == 0) {
      options.iterations = std::stoi(value(argument));
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.m == 0 || options.n == 0 || options.k == 0 ||
      options.m % 128 != 0 || options.n % 256 != 0 ||
      options.k % 64 != 0 || options.warmup < 0 || options.iterations <= 0) {
    throw std::invalid_argument(
        "M/N/K must be positive multiples of 128/256/64 and iteration "
        "counts must be valid");
  }
  return options;
}

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

__global__ void fill_bf16(__nv_bfloat16* data, std::size_t count,
                          std::uint32_t seed) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    std::uint32_t bits = static_cast<std::uint32_t>(index) ^ seed;
    bits ^= bits >> 16;
    bits *= 0x7feb352dU;
    bits ^= bits >> 15;
    const float sample = static_cast<float>(static_cast<int>(bits % 33) - 16) /
                         16.0f;
    data[index] = __float2bfloat16(sample);
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

double percentile(std::vector<float> samples, double probability) {
  std::sort(samples.begin(), samples.end());
  const auto index = static_cast<std::size_t>(
      std::ceil(probability * samples.size()) - 1.0);
  return samples[std::min(index, samples.size() - 1)];
}

struct Allocation {
  void* pointer = nullptr;
  ~Allocation() { cudaFree(pointer); }

  void allocate(std::size_t bytes) {
    require_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
  }
};

}  // namespace

int main(int argc, char** argv) {
  using namespace blackwell_moe;
  try {
    const auto options = parse_options(argc, argv);

    int device = 0;
    int runtime_version = 0;
    cudaDeviceProp properties{};
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    require_cuda(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties");
    require_cuda(cudaRuntimeGetVersion(&runtime_version),
                 "cudaRuntimeGetVersion");

    const auto a_elements = static_cast<std::size_t>(options.m) * options.k;
    const auto b_elements = static_cast<std::size_t>(options.n) * options.k;
    const auto d_elements = static_cast<std::size_t>(options.m) * options.n;
    Allocation a;
    Allocation b;
    Allocation d;
    a.allocate(a_elements * sizeof(__nv_bfloat16));
    b.allocate(b_elements * sizeof(__nv_bfloat16));
    d.allocate(d_elements * sizeof(float));
    initialize_bf16(a.pointer, a_elements, options.seed);
    initialize_bf16(b.pointer, b_elements, options.seed + 1);
    require_cuda(cudaMemset(d.pointer, 0, d_elements * sizeof(float)),
                 "cudaMemset(D)");

    DirectCuteGemmArguments arguments;
    arguments.a = a.pointer;
    arguments.b = b.pointer;
    arguments.d = d.pointer;
    arguments.m = options.m;
    arguments.n = options.n;
    arguments.k = options.k;

    DirectCuteGemmStats stats;
    for (int i = 0; i < options.warmup; ++i) {
      const auto status = launch_direct_cute_gemm(arguments, &stats);
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
      const auto status = launch_direct_cute_gemm(arguments, &stats);
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
    const double tflops =
        2.0 * static_cast<double>(options.m) * options.n * options.k /
        (median_ms * 1.0e9);

    if (options.csv) {
      std::cout << "kernel,device,compute_capability,cuda_runtime,cutlass,m,n,k,"
                   "warmup,iterations,median_ms,p95_ms,tflops,tile_m,tile_n,"
                   "tile_k,grid_m,grid_n,dynamic_smem_bytes,output_type\n";
      std::cout << direct_cute_kernel_name() << ',' << properties.name << ','
                << properties.major << '.' << properties.minor << ','
                << runtime_version << ',' << BLACKWELL_MOE_CUTLASS_REVISION
                << ',' << options.m << ',' << options.n << ',' << options.k
                << ',' << options.warmup << ',' << options.iterations << ','
                << median_ms << ',' << p95_ms << ',' << tflops << ','
                << stats.tile_m << ',' << stats.tile_n << ',' << stats.tile_k
                << ',' << stats.grid_m << ',' << stats.grid_n << ','
                << stats.dynamic_smem_bytes << ",f32\n";
    } else {
      std::cout << std::fixed << std::setprecision(4)
                << "kernel:             " << direct_cute_kernel_name() << '\n'
                << "device:             " << properties.name << '\n'
                << "compute capability: " << properties.major << '.'
                << properties.minor << '\n'
                << "CUDA runtime:       " << runtime_version << '\n'
                << "CUTLASS:            " << BLACKWELL_MOE_CUTLASS_REVISION
                << '\n'
                << "shape:              [" << options.m << ',' << options.n
                << ',' << options.k << "]\n"
                << "tile:               [" << stats.tile_m << ','
                << stats.tile_n << ',' << stats.tile_k << "]\n"
                << "grid:               [" << stats.grid_m << ','
                << stats.grid_n << "]\n"
                << "output:             FP32\n"
                << "median latency:     " << median_ms << " ms\n"
                << "p95 latency:        " << p95_ms << " ms\n"
                << "throughput:         " << tflops << " TFLOP/s\n"
                << "dynamic smem:       " << stats.dynamic_smem_bytes
                << " bytes\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Direct CuTe benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
