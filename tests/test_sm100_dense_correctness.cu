#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackwell_moe/kernels/sm100_dense_gemm.cuh"
#include "cutlass/bfloat16.h"

namespace {

using Bf16 = cutlass::bfloat16_t;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

struct DeviceBuffers {
  Bf16* a = nullptr;
  Bf16* b = nullptr;
  Bf16* d = nullptr;
  ~DeviceBuffers() {
    cudaFree(d);
    cudaFree(b);
    cudaFree(a);
  }
};

}  // namespace

int main() {
  using namespace blackwell_moe;
  try {
    int device = 0;
    cudaDeviceProp properties{};
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    require_cuda(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties");
    if (properties.major != 10 || properties.minor != 0) {
      std::cout << "SKIP: native tcgen05/TMEM test requires SM100a\n";
      return 0;
    }

    constexpr std::uint32_t kM = 128;
    constexpr std::uint32_t kN = 128;
    constexpr std::uint32_t kK = 64;
    std::vector<Bf16> host_a(static_cast<std::size_t>(kM) * kK);
    std::vector<Bf16> host_b(static_cast<std::size_t>(kN) * kK);
    std::vector<Bf16> output(static_cast<std::size_t>(kM) * kN);

    for (std::size_t i = 0; i < host_a.size(); ++i) {
      host_a[i] = Bf16(static_cast<float>(static_cast<int>(i % 9) - 4) /
                       16.0f);
    }
    for (std::size_t i = 0; i < host_b.size(); ++i) {
      host_b[i] = Bf16(static_cast<float>(static_cast<int>((i * 3) % 11) - 5) /
                       16.0f);
    }

    DeviceBuffers buffers;
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers.a),
                            host_a.size() * sizeof(Bf16)),
                 "cudaMalloc(A)");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers.b),
                            host_b.size() * sizeof(Bf16)),
                 "cudaMalloc(B)");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers.d),
                            output.size() * sizeof(Bf16)),
                 "cudaMalloc(D)");
    require_cuda(cudaMemcpy(buffers.a, host_a.data(),
                            host_a.size() * sizeof(Bf16),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(A)");
    require_cuda(cudaMemcpy(buffers.b, host_b.data(),
                            host_b.size() * sizeof(Bf16),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(B)");
    require_cuda(cudaMemset(buffers.d, 0, output.size() * sizeof(Bf16)),
                 "cudaMemset(D)");

    Sm100DenseGemmArguments arguments;
    arguments.a = buffers.a;
    arguments.b = buffers.b;
    arguments.d = buffers.d;
    arguments.m = kM;
    arguments.n = kN;
    arguments.k = kK;

    Sm100DenseGemmPlan plan;
    auto status = plan.initialize(arguments);
    require(status == KernelStatus::kSuccess, kernel_status_string(status));
    status = plan.run();
    require(status == KernelStatus::kSuccess, kernel_status_string(status));
    require_cuda(cudaDeviceSynchronize(), "SM100 GEMM execution");
    require_cuda(cudaMemcpy(output.data(), buffers.d,
                            output.size() * sizeof(Bf16),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(D)");

    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
    for (std::uint32_t row = 0; row < kM; ++row) {
      for (std::uint32_t column = 0; column < kN; ++column) {
        float reference = 0.0f;
        for (std::uint32_t inner = 0; inner < kK; ++inner) {
          reference += static_cast<float>(host_a[row * kK + inner]) *
                       static_cast<float>(host_b[column * kK + inner]);
        }
        const float actual = static_cast<float>(output[row * kN + column]);
        const float absolute_error = std::abs(actual - reference);
        const float relative_error =
            absolute_error / std::max(1.0f, std::abs(reference));
        max_absolute_error = std::max(max_absolute_error, absolute_error);
        max_relative_error = std::max(max_relative_error, relative_error);
        require(absolute_error <= 0.04f + 0.02f * std::abs(reference),
                "SM100 BF16 GEMM result exceeds tolerance");
      }
    }

    const auto stats = plan.stats();
    require(stats.tile_m == 128 && stats.tile_n == 128 && stats.tile_k == 64,
            "unexpected native MMA tile shape");
    std::cout << "SM100 native BF16 GEMM correctness passed"
              << " (device=" << properties.name
              << ", kernel=" << sm100_dense_kernel_name()
              << ", max_abs_error=" << max_absolute_error
              << ", max_rel_error=" << max_relative_error << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "SM100 native correctness test failed: " << error.what()
              << '\n';
    return 1;
  }
}
