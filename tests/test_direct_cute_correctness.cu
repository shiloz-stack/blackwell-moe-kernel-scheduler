#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackwell_moe/kernels/direct_cute_gemm.cuh"
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
  float* d = nullptr;
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
      std::cout << "SKIP: direct CuTe tcgen05/TMEM test requires SM100a\n";
      return 0;
    }

    constexpr std::uint32_t kM = 128;
    constexpr std::uint32_t kN = 256;
    constexpr std::uint32_t kK = 64;
    std::vector<Bf16> host_a(static_cast<std::size_t>(kM) * kK);
    std::vector<Bf16> host_b(static_cast<std::size_t>(kN) * kK);
    std::vector<float> output(static_cast<std::size_t>(kM) * kN);

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
                            output.size() * sizeof(float)),
                 "cudaMalloc(D)");
    require_cuda(cudaMemcpy(buffers.a, host_a.data(),
                            host_a.size() * sizeof(Bf16),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(A)");
    require_cuda(cudaMemcpy(buffers.b, host_b.data(),
                            host_b.size() * sizeof(Bf16),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(B)");
    require_cuda(cudaMemset(buffers.d, 0, output.size() * sizeof(float)),
                 "cudaMemset(D)");

    DirectCuteGemmArguments arguments;
    arguments.a = buffers.a;
    arguments.b = buffers.b;
    arguments.d = buffers.d;
    arguments.m = kM;
    arguments.n = kN;
    arguments.k = kK;

    DirectCuteGemmStats stats;
    const auto status = launch_direct_cute_gemm(arguments, &stats);
    require(status == KernelStatus::kSuccess, kernel_status_string(status));
    require_cuda(cudaDeviceSynchronize(), "direct CuTe GEMM execution");
    require_cuda(cudaMemcpy(output.data(), buffers.d,
                            output.size() * sizeof(float),
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
        const float actual = output[row * kN + column];
        const float absolute_error = std::abs(actual - reference);
        const float relative_error =
            absolute_error / std::max(1.0f, std::abs(reference));
        max_absolute_error = std::max(max_absolute_error, absolute_error);
        max_relative_error = std::max(max_relative_error, relative_error);
        require(absolute_error <= 0.002f + 0.002f * std::abs(reference),
                "direct CuTe FP32 result exceeds tolerance");
      }
    }

    require(stats.grid_m == 1 && stats.grid_n == 1,
            "unexpected direct CuTe launch grid");
    require(stats.dynamic_smem_bytes > 0,
            "direct CuTe kernel reported no shared memory");
    std::cout << "Direct CuTe SM100 GEMM correctness passed"
              << " (device=" << properties.name
              << ", kernel=" << direct_cute_kernel_name()
              << ", smem_bytes=" << stats.dynamic_smem_bytes
              << ", max_abs_error=" << max_absolute_error
              << ", max_rel_error=" << max_relative_error << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Direct CuTe correctness test failed: " << error.what()
              << '\n';
    return 1;
  }
}
