#include "blackwell_moe/kernels/grouped_gemm.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"

namespace blackwell_moe {
namespace {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    ElementB,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

template <typename T>
bool allocate_and_copy(T*& device, const std::vector<T>& host) {
  if (host.empty()) {
    device = nullptr;
    return true;
  }
  const auto bytes = host.size() * sizeof(T);
  if (cudaMalloc(reinterpret_cast<void**>(&device), bytes) != cudaSuccess) {
    device = nullptr;
    return false;
  }
  if (cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cudaFree(device);
    device = nullptr;
    return false;
  }
  return true;
}

}  // namespace

struct Bf16GroupedGemmPlan::Impl {
  Gemm gemm;
  typename Gemm::Arguments* arguments = nullptr;
  std::vector<cutlass::gemm::GemmCoord> host_problem_sizes;
  cutlass::gemm::GemmCoord* problem_sizes = nullptr;
  ElementA const** ptr_a = nullptr;
  ElementB const** ptr_b = nullptr;
  ElementOutput** ptr_c = nullptr;
  ElementOutput** ptr_d = nullptr;
  std::int64_t* lda = nullptr;
  std::int64_t* ldb = nullptr;
  std::int64_t* ldc = nullptr;
  std::int64_t* ldd = nullptr;
  std::uint8_t* workspace = nullptr;
  cudaStream_t default_stream = nullptr;
  GroupedGemmPlanStats plan_stats;
  bool ready = false;

  ~Impl() { release(); }

  void release() {
    delete arguments;
    arguments = nullptr;
    cudaFree(workspace);
    cudaFree(ldd);
    cudaFree(ldc);
    cudaFree(ldb);
    cudaFree(lda);
    cudaFree(ptr_d);
    cudaFree(ptr_c);
    cudaFree(ptr_b);
    cudaFree(ptr_a);
    cudaFree(problem_sizes);
    workspace = nullptr;
    ldd = ldc = ldb = lda = nullptr;
    ptr_d = ptr_c = nullptr;
    ptr_b = nullptr;
    ptr_a = nullptr;
    problem_sizes = nullptr;
    host_problem_sizes.clear();
    plan_stats = {};
    ready = false;
  }
};

const char* kernel_status_string(KernelStatus status) {
  switch (status) {
    case KernelStatus::kSuccess: return "success";
    case KernelStatus::kInvalidArgument: return "invalid argument";
    case KernelStatus::kUnsupportedDataType: return "unsupported data type";
    case KernelStatus::kUnsupportedScheduler: return "unsupported scheduler";
    case KernelStatus::kUnsupportedDevice: return "unsupported device";
    case KernelStatus::kCudaError: return "CUDA error";
    case KernelStatus::kCutlassError: return "CUTLASS error";
  }
  return "unknown status";
}

Bf16GroupedGemmPlan::Bf16GroupedGemmPlan() : impl_(new Impl) {}

Bf16GroupedGemmPlan::~Bf16GroupedGemmPlan() { delete impl_; }

Bf16GroupedGemmPlan::Bf16GroupedGemmPlan(
    Bf16GroupedGemmPlan&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

Bf16GroupedGemmPlan& Bf16GroupedGemmPlan::operator=(
    Bf16GroupedGemmPlan&& other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

KernelStatus Bf16GroupedGemmPlan::initialize(
    const GroupedGemmArguments& args) {
  if (impl_ == nullptr) impl_ = new Impl;
  impl_->release();

  if (args.a == nullptr || args.b == nullptr || args.d == nullptr ||
      args.tokens_per_expert == nullptr || args.experts == 0 || args.n == 0 ||
      args.k == 0 || args.n % 8 != 0 || args.k % 8 != 0) {
    return KernelStatus::kInvalidArgument;
  }
  if (args.data_type != DataType::kBf16) {
    return KernelStatus::kUnsupportedDataType;
  }
  if (args.scheduler != SchedulerKind::kExpertOrder) {
    return KernelStatus::kUnsupportedScheduler;
  }

  cudaDeviceProp properties{};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return KernelStatus::kCudaError;
  }
  if (properties.major < 8) return KernelStatus::kUnsupportedDevice;

  std::vector<ElementA const*> host_a;
  std::vector<ElementB const*> host_b;
  std::vector<ElementOutput*> host_c;
  std::vector<ElementOutput*> host_d;
  std::vector<std::int64_t> host_lda;
  std::vector<std::int64_t> host_ldb;
  std::vector<std::int64_t> host_ldc;
  std::vector<std::int64_t> host_ldd;

  for (std::uint32_t expert = 0; expert < args.experts; ++expert) {
    const auto m = args.tokens_per_expert[expert];
    if (m == 0) continue;
    if (args.a[expert] == nullptr || args.b[expert] == nullptr ||
        args.d[expert] == nullptr) {
      impl_->release();
      return KernelStatus::kInvalidArgument;
    }
    impl_->host_problem_sizes.emplace_back(m, args.n, args.k);
    host_a.push_back(static_cast<ElementA const*>(args.a[expert]));
    host_b.push_back(static_cast<ElementB const*>(args.b[expert]));
    host_c.push_back(static_cast<ElementOutput*>(args.d[expert]));
    host_d.push_back(static_cast<ElementOutput*>(args.d[expert]));
    host_lda.push_back(args.k);
    host_ldb.push_back(args.k);
    host_ldc.push_back(args.n);
    host_ldd.push_back(args.n);
  }

  impl_->plan_stats.active_experts =
      static_cast<std::uint32_t>(impl_->host_problem_sizes.size());
  impl_->default_stream = static_cast<cudaStream_t>(args.stream);
  if (impl_->host_problem_sizes.empty()) {
    impl_->ready = true;
    return KernelStatus::kSuccess;
  }

  const auto block_count =
      Gemm::sufficient(impl_->host_problem_sizes.data(),
                       static_cast<int>(impl_->host_problem_sizes.size()));
  if (block_count <= 0) {
    impl_->release();
    return KernelStatus::kUnsupportedDevice;
  }
  impl_->plan_stats.threadblock_count = block_count;

  const bool metadata_ok =
      allocate_and_copy(impl_->problem_sizes, impl_->host_problem_sizes) &&
      allocate_and_copy(impl_->ptr_a, host_a) &&
      allocate_and_copy(impl_->ptr_b, host_b) &&
      allocate_and_copy(impl_->ptr_c, host_c) &&
      allocate_and_copy(impl_->ptr_d, host_d) &&
      allocate_and_copy(impl_->lda, host_lda) &&
      allocate_and_copy(impl_->ldb, host_ldb) &&
      allocate_and_copy(impl_->ldc, host_ldc) &&
      allocate_and_copy(impl_->ldd, host_ldd);
  if (!metadata_ok) {
    impl_->release();
    return KernelStatus::kCudaError;
  }

  typename Gemm::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);
  impl_->arguments = new typename Gemm::Arguments(
      impl_->problem_sizes,
      static_cast<int>(impl_->host_problem_sizes.size()),
      block_count,
      epilogue,
      impl_->ptr_a,
      impl_->ptr_b,
      impl_->ptr_c,
      impl_->ptr_d,
      impl_->lda,
      impl_->ldb,
      impl_->ldc,
      impl_->ldd,
      impl_->host_problem_sizes.data());

  impl_->plan_stats.workspace_bytes =
      impl_->gemm.get_workspace_size(*impl_->arguments);
  if (impl_->plan_stats.workspace_bytes > 0 &&
      cudaMalloc(reinterpret_cast<void**>(&impl_->workspace),
                 impl_->plan_stats.workspace_bytes) != cudaSuccess) {
    impl_->release();
    return KernelStatus::kCudaError;
  }

  if (Gemm::can_implement(*impl_->arguments) != cutlass::Status::kSuccess) {
    impl_->release();
    return KernelStatus::kInvalidArgument;
  }
  if (impl_->gemm.initialize(*impl_->arguments, impl_->workspace,
                             impl_->default_stream) !=
      cutlass::Status::kSuccess) {
    impl_->release();
    return KernelStatus::kCutlassError;
  }
  impl_->ready = true;
  return KernelStatus::kSuccess;
}

KernelStatus Bf16GroupedGemmPlan::run(void* stream) {
  if (impl_ == nullptr || !impl_->ready) {
    return KernelStatus::kInvalidArgument;
  }
  if (impl_->plan_stats.active_experts == 0) return KernelStatus::kSuccess;
  auto cuda_stream = stream == nullptr ? impl_->default_stream
                                       : static_cast<cudaStream_t>(stream);
  if (impl_->gemm.run(cuda_stream) != cutlass::Status::kSuccess) {
    return KernelStatus::kCutlassError;
  }
  return cudaPeekAtLastError() == cudaSuccess ? KernelStatus::kSuccess
                                               : KernelStatus::kCudaError;
}

GroupedGemmPlanStats Bf16GroupedGemmPlan::stats() const {
  return impl_ == nullptr ? GroupedGemmPlanStats{} : impl_->plan_stats;
}

KernelStatus launch_grouped_gemm_baseline(const GroupedGemmArguments& args) {
  Bf16GroupedGemmPlan plan;
  auto status = plan.initialize(args);
  if (status != KernelStatus::kSuccess) return status;
  return plan.run(args.stream);
}

}  // namespace blackwell_moe
