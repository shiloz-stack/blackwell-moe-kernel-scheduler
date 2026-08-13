#include "blackwell_moe/kernels/sm100_2sm_gemm.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <utility>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

namespace blackwell_moe {
namespace {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int kAlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// A 2-SM tcgen05 instruction has a 256-row MMA extent.  Cluster M must be a
// multiple of two so the leader and peer CTA are resident together.
using MmaTileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

using EpilogueOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD,
    ElementCompute,
    ElementC,
    ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        kAlignmentC,
        ElementD,
        LayoutD,
        kAlignmentD,
        cutlass::epilogue::TmaWarpSpecialized2Sm,
        EpilogueOperation>::CollectiveOp;

// StageCountAutoCarveout fills the remaining shared memory with pipeline
// stages after reserving the TMA epilogue storage.  This is the optimized
// multi-stage, warp-specialized path; the direct CuTe kernel remains the
// intentionally transparent single-stage teaching implementation.
using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        kAlignmentA,
        ElementB,
        LayoutB,
        kAlignmentB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using ProblemShape = typename Gemm::GemmKernel::ProblemShape;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

}  // namespace

struct Sm100TwoSmGemmPlan::Impl {
  Gemm gemm;
  typename Gemm::Arguments* arguments = nullptr;
  std::uint8_t* workspace = nullptr;
  cudaStream_t default_stream = nullptr;
  Sm100DenseGemmPlanStats plan_stats;
  bool ready = false;

  Impl() {
    plan_stats.tile_m = 256;
    plan_stats.tile_n = 128;
    plan_stats.tile_k = 64;
  }

  ~Impl() { release(); }

  void release() {
    delete arguments;
    arguments = nullptr;
    cudaFree(workspace);
    workspace = nullptr;
    default_stream = nullptr;
    const auto tile_m = plan_stats.tile_m;
    const auto tile_n = plan_stats.tile_n;
    const auto tile_k = plan_stats.tile_k;
    plan_stats = {};
    plan_stats.tile_m = tile_m;
    plan_stats.tile_n = tile_n;
    plan_stats.tile_k = tile_k;
    ready = false;
  }
};

Sm100TwoSmGemmPlan::Sm100TwoSmGemmPlan() : impl_(new Impl) {}
Sm100TwoSmGemmPlan::~Sm100TwoSmGemmPlan() { delete impl_; }

Sm100TwoSmGemmPlan::Sm100TwoSmGemmPlan(
    Sm100TwoSmGemmPlan&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

Sm100TwoSmGemmPlan& Sm100TwoSmGemmPlan::operator=(
    Sm100TwoSmGemmPlan&& other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

KernelStatus Sm100TwoSmGemmPlan::initialize(
    const Sm100DenseGemmArguments& args) {
  if (impl_ == nullptr) impl_ = new Impl;
  impl_->release();

  if (args.a == nullptr || args.b == nullptr || args.d == nullptr ||
      args.m == 0 || args.n == 0 || args.k == 0 || args.n % 8 != 0 ||
      args.k % 8 != 0) {
    return KernelStatus::kInvalidArgument;
  }

  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return KernelStatus::kCudaError;
  }
  if (properties.major != 10 || properties.minor != 0) {
    return KernelStatus::kUnsupportedDevice;
  }

  const auto m = static_cast<int>(args.m);
  const auto n = static_cast<int>(args.n);
  const auto k = static_cast<int>(args.k);
  constexpr int l = 1;
  const ProblemShape problem_shape{m, n, k, l};
  const auto stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, l));
  const auto stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n, k, l));
  const auto stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(m, n, l));
  const auto stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(m, n, l));

  cutlass::KernelHardwareInfo hardware_info;
  hardware_info.device_id = device;
  hardware_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  impl_->plan_stats.sm_count = hardware_info.sm_count;
  impl_->default_stream = static_cast<cudaStream_t>(args.stream);

  impl_->arguments = new typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {static_cast<ElementA const*>(args.a), stride_a,
       static_cast<ElementB const*>(args.b), stride_b},
      {{}, static_cast<ElementC const*>(args.d), stride_c,
       static_cast<ElementD*>(args.d), stride_d},
      hardware_info};
  impl_->arguments->epilogue.thread.alpha = 1.0f;
  impl_->arguments->epilogue.thread.beta = 0.0f;

  const auto grid = Gemm::get_grid_shape(*impl_->arguments);
  impl_->plan_stats.persistent_ctas = grid.x * grid.y * grid.z;

  if (impl_->gemm.can_implement(*impl_->arguments) !=
      cutlass::Status::kSuccess) {
    impl_->release();
    return KernelStatus::kInvalidArgument;
  }

  impl_->plan_stats.workspace_bytes =
      Gemm::get_workspace_size(*impl_->arguments);
  if (impl_->plan_stats.workspace_bytes > 0 &&
      cudaMalloc(reinterpret_cast<void**>(&impl_->workspace),
                 impl_->plan_stats.workspace_bytes) != cudaSuccess) {
    impl_->release();
    return KernelStatus::kCudaError;
  }
  if (impl_->gemm.initialize(*impl_->arguments,
                             impl_->workspace,
                             impl_->default_stream) !=
      cutlass::Status::kSuccess) {
    impl_->release();
    return KernelStatus::kCutlassError;
  }
  impl_->ready = true;
  return KernelStatus::kSuccess;
}

KernelStatus Sm100TwoSmGemmPlan::run(void* stream) {
  if (impl_ == nullptr || !impl_->ready) {
    return KernelStatus::kInvalidArgument;
  }
  const auto cuda_stream = stream == nullptr
                               ? impl_->default_stream
                               : static_cast<cudaStream_t>(stream);
  if (impl_->gemm.run(cuda_stream) != cutlass::Status::kSuccess) {
    return KernelStatus::kCutlassError;
  }
  return cudaPeekAtLastError() == cudaSuccess ? KernelStatus::kSuccess
                                               : KernelStatus::kCudaError;
}

Sm100DenseGemmPlanStats Sm100TwoSmGemmPlan::stats() const {
  return impl_ == nullptr ? Sm100DenseGemmPlanStats{} : impl_->plan_stats;
}

KernelStatus launch_sm100_2sm_gemm(const Sm100DenseGemmArguments& args) {
  Sm100TwoSmGemmPlan plan;
  auto status = plan.initialize(args);
  return status == KernelStatus::kSuccess ? plan.run(args.stream) : status;
}

const char* sm100_2sm_kernel_name() {
  return "sm100_2sm_persistent_clc_tma_multicast_tcgen05_bf16_256x128x64";
}

}  // namespace blackwell_moe
