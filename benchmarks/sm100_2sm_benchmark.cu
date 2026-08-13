#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackwell_moe/kernels/sm100_2sm_gemm.cuh"

#ifndef BLACKWELL_MOE_CUTLASS_REVISION
#define BLACKWELL_MOE_CUTLASS_REVISION "unknown"
#endif

namespace {

struct Options {
  std::uint32_t m = 4096;
  std::uint32_t n = 7168;
  std::uint32_t k = 2048;
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
      std::cout << "Usage: sm100_2sm_bench [--m=N --n=N --k=N "
                   "--warmup=N --iterations=N --csv]\n";
      std::exit(0);
    } else if (argument.rfind("--m=", 0) == 0) {
      options.m = std::stoul(value(argument));
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
  if (options.m == 0 || options.n == 0 || options.k == 0 ||
      options.n % 8 != 0 || options.k % 8 != 0 || options.warmup < 0 ||
      options.iterations <= 0) {
    throw std::invalid_argument("invalid GEMM shape or iteration count");
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
    data[index] = __float2bfloat16(
        static_cast<float>(static_cast<int>(bits % 33) - 16) / 16.0f);
  }
}

struct Allocation {
  void* pointer = nullptr;
  ~Allocation() { cudaFree(pointer); }
  void allocate(std::size_t bytes) {
    require_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
  }
};

double percentile(std::vector<float> samples, double probability) {
  std::sort(samples.begin(), samples.end());
  const auto index = static_cast<std::size_t>(
      std::ceil(probability * samples.size()) - 1.0);
  return samples[std::min(index, samples.size() - 1)];
}

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
    d.allocate(d_elements * sizeof(__nv_bfloat16));
    fill_bf16<<<std::min<std::size_t>(4096, (a_elements + 255) / 256), 256>>>(
        static_cast<__nv_bfloat16*>(a.pointer), a_elements, 2026);
    fill_bf16<<<std::min<std::size_t>(4096, (b_elements + 255) / 256), 256>>>(
        static_cast<__nv_bfloat16*>(b.pointer), b_elements, 2027);
    require_cuda(cudaPeekAtLastError(), "input initialization");

    Sm100DenseGemmArguments arguments;
    arguments.a = a.pointer;
    arguments.b = b.pointer;
    arguments.d = d.pointer;
    arguments.m = options.m;
    arguments.n = options.n;
    arguments.k = options.k;

    Sm100TwoSmGemmPlan plan;
    auto status = plan.initialize(arguments);
    if (status != KernelStatus::kSuccess) {
      throw std::runtime_error(kernel_status_string(status));
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
    for (int i = 0; i < options.iterations; ++i) {
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
    const double tflops =
        2.0 * static_cast<double>(options.m) * options.n * options.k /
        (median_ms * 1.0e9);
    const auto stats = plan.stats();
    if (options.csv) {
      std::cout << "kernel,device,compute_capability,cuda_runtime,cutlass,m,n,k,"
                   "warmup,iterations,median_ms,p95_ms,tflops,tile_m,tile_n,"
                   "tile_k,sm_count,persistent_ctas,workspace_bytes\n"
                << sm100_2sm_kernel_name() << ',' << properties.name << ','
                << properties.major << '.' << properties.minor << ','
                << runtime_version << ',' << BLACKWELL_MOE_CUTLASS_REVISION
                << ',' << options.m << ',' << options.n << ',' << options.k
                << ',' << options.warmup << ',' << options.iterations << ','
                << median_ms << ',' << p95_ms << ',' << tflops << ','
                << stats.tile_m << ',' << stats.tile_n << ',' << stats.tile_k
                << ',' << stats.sm_count << ',' << stats.persistent_ctas << ','
                << stats.workspace_bytes << '\n';
    } else {
      std::cout << "kernel=" << sm100_2sm_kernel_name()
                << " shape=" << options.m << 'x' << options.n << 'x'
                << options.k << " median_ms=" << median_ms
                << " p95_ms=" << p95_ms << " tflops=" << tflops << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "SM100 2-SM benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
