#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackwell_moe/kernels/grouped_gemm.cuh"
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

struct ExpertBuffers {
  Bf16* a = nullptr;
  Bf16* b = nullptr;
  Bf16* d = nullptr;

  ExpertBuffers() = default;
  ExpertBuffers(const ExpertBuffers&) = delete;
  ExpertBuffers& operator=(const ExpertBuffers&) = delete;

  ~ExpertBuffers() {
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
    if (properties.major < 8) {
      std::cout << "SKIP: BF16 Tensor Core baseline requires compute capability 8.0+\n";
      return 0;
    }

    constexpr std::uint32_t kN = 32;
    constexpr std::uint32_t kK = 32;
    const std::vector<std::uint32_t> tokens{7, 0, 13, 31};
    // Fixed-size storage avoids moving the non-copyable allocation owners.
    ExpertBuffers buffers[4];
    std::vector<const void*> a_ptrs(tokens.size(), nullptr);
    std::vector<const void*> b_ptrs(tokens.size(), nullptr);
    std::vector<void*> d_ptrs(tokens.size(), nullptr);
    std::vector<std::vector<Bf16>> host_a(tokens.size());
    std::vector<std::vector<Bf16>> host_b(tokens.size());

    for (std::size_t expert = 0; expert < tokens.size(); ++expert) {
      const auto m = tokens[expert];
      if (m == 0) continue;
      host_a[expert].resize(static_cast<std::size_t>(m) * kK);
      host_b[expert].resize(static_cast<std::size_t>(kN) * kK);
      for (std::size_t i = 0; i < host_a[expert].size(); ++i) {
        const int value = static_cast<int>((i + expert * 3) % 11) - 5;
        host_a[expert][i] = Bf16(static_cast<float>(value) / 8.0f);
      }
      for (std::size_t i = 0; i < host_b[expert].size(); ++i) {
        const int value = static_cast<int>((i * 3 + expert) % 13) - 6;
        host_b[expert][i] = Bf16(static_cast<float>(value) / 8.0f);
      }

      const auto a_bytes = host_a[expert].size() * sizeof(Bf16);
      const auto b_bytes = host_b[expert].size() * sizeof(Bf16);
      const auto d_bytes = static_cast<std::size_t>(m) * kN * sizeof(Bf16);
      require_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers[expert].a),
                              a_bytes),
                   "cudaMalloc(A)");
      require_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers[expert].b),
                              b_bytes),
                   "cudaMalloc(B)");
      require_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers[expert].d),
                              d_bytes),
                   "cudaMalloc(D)");
      require_cuda(cudaMemcpy(buffers[expert].a, host_a[expert].data(), a_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(A)");
      require_cuda(cudaMemcpy(buffers[expert].b, host_b[expert].data(), b_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(B)");
      require_cuda(cudaMemset(buffers[expert].d, 0, d_bytes), "cudaMemset(D)");
      a_ptrs[expert] = buffers[expert].a;
      b_ptrs[expert] = buffers[expert].b;
      d_ptrs[expert] = buffers[expert].d;
    }

    GroupedGemmArguments args;
    args.a = a_ptrs.data();
    args.b = b_ptrs.data();
    args.d = d_ptrs.data();
    args.tokens_per_expert = tokens.data();
    args.experts = static_cast<std::uint32_t>(tokens.size());
    args.n = kN;
    args.k = kK;

    Bf16GroupedGemmPlan plan;
    auto status = plan.initialize(args);
    require(status == KernelStatus::kSuccess,
            kernel_status_string(status));
    require(plan.stats().active_experts == 3,
            "inactive expert filtering is incorrect");
    status = plan.run();
    require(status == KernelStatus::kSuccess,
            kernel_status_string(status));
    require_cuda(cudaDeviceSynchronize(), "grouped GEMM execution");

    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
    for (std::size_t expert = 0; expert < tokens.size(); ++expert) {
      const auto m = tokens[expert];
      if (m == 0) continue;
      std::vector<Bf16> output(static_cast<std::size_t>(m) * kN);
      require_cuda(cudaMemcpy(output.data(), buffers[expert].d,
                              output.size() * sizeof(Bf16),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(D)");
      for (std::uint32_t row = 0; row < m; ++row) {
        for (std::uint32_t column = 0; column < kN; ++column) {
          float reference = 0.0f;
          for (std::uint32_t inner = 0; inner < kK; ++inner) {
            reference += static_cast<float>(
                             host_a[expert][row * kK + inner]) *
                         static_cast<float>(
                             host_b[expert][column * kK + inner]);
          }
          const float actual = static_cast<float>(output[row * kN + column]);
          const float absolute_error = std::abs(actual - reference);
          const float relative_error =
              absolute_error / std::max(1.0f, std::abs(reference));
          max_absolute_error = std::max(max_absolute_error, absolute_error);
          max_relative_error = std::max(max_relative_error, relative_error);
          require(absolute_error <= 0.08f + 0.02f * std::abs(reference),
                  "BF16 grouped GEMM result exceeds tolerance");
        }
      }
    }

    std::cout << "CUTLASS BF16 grouped GEMM correctness passed"
              << " (device=" << properties.name
              << ", active_experts=" << plan.stats().active_experts
              << ", max_abs_error=" << max_absolute_error
              << ", max_rel_error=" << max_relative_error << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUTLASS correctness test failed: " << error.what() << '\n';
    return 1;
  }
}
