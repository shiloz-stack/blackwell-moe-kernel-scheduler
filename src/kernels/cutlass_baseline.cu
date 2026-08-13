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

// -----------------------------------------------------------------------------
// 1. CUTLASS kernel configuration
// -----------------------------------------------------------------------------
// This file is the host-side wrapper for a CUTLASS grouped GEMM. It does not
// contain a handwritten __global__ GEMM body. Instead, DefaultGemmGrouped uses
// the types and tile shapes below to generate the device kernel at compile time.
//
// At a high level, the kernel evaluates one GEMM for every active MoE expert:
//
//   D_i[M_i, N] = A_i[M_i, K] * B_i[K, N]
//
// M_i varies with the routing result, while N and K are shared by all experts.
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// Important parts of this kernel configuration:
//   - BF16 inputs and output with FP32 accumulation.
//   - Tensor Core execution through the SM80-compatible CUTLASS path.
//   - A 128x128x32 threadblock tile, subdivided into 64x64x32 warp tiles.
//   - BF16 Tensor Core instructions with an m16n8k16 instruction shape.
//   - Four pipeline stages for overlapping memory movement and MMA work.
//   - DeviceOnly grouped scheduling: CUTLASS maps persistent CTAs to the
//     flattened tile space of all expert GEMMs on the GPU.
//
// This is intentionally a portable baseline for Blackwell measurements. It is
// not yet an SM100-native tcgen05/TMA/CLC kernel.
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

// GemmKernel is the generated device-side implementation. Gemm is CUTLASS's
// host-side adapter for creating parameters, initializing workspace, and
// launching that kernel.
using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

// Allocate one device-side metadata array and populate it from its host copy.
// Matrix payloads are not copied here; the arrays contain shapes, strides, or
// device pointers that tell CUTLASS where each expert's matrices already live.
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

// -----------------------------------------------------------------------------
// 2. Reusable grouped-GEMM plan state
// -----------------------------------------------------------------------------
// A plan owns everything that can be prepared outside the timed kernel loop:
// CUTLASS arguments, device-side problem metadata, optional workspace, and the
// selected CUDA stream. Keeping initialize() separate from run() prevents
// allocation and host-to-device metadata copies from polluting benchmark time.
struct Bf16GroupedGemmPlan::Impl {
  Gemm gemm;
  typename Gemm::Arguments* arguments = nullptr;
  // CUTLASS may use the host problem list while constructing the device-side
  // grouped schedule, so it must remain alive for the lifetime of the plan.
  std::vector<cutlass::gemm::GemmCoord> host_problem_sizes;
  // The following arrays live on the GPU. Entry p in every array describes the
  // same active expert problem: its [M, N, K], matrix pointers, and strides.
  cutlass::gemm::GemmCoord* problem_sizes = nullptr;
  ElementA** ptr_a = nullptr;
  ElementB** ptr_b = nullptr;
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

  // Reset the plan to an empty state. cudaFree(nullptr) is valid, which keeps
  // cleanup safe for partially initialized plans and error paths.
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
  // ---------------------------------------------------------------------------
  // 3. Validate the requested workload and target device
  // ---------------------------------------------------------------------------
  if (impl_ == nullptr) impl_ = new Impl;
  impl_->release();

  // AlignmentA/AlignmentB are both eight BF16 elements (16 bytes). Requiring N
  // and K to be multiples of eight keeps the vectorized accesses compatible
  // with this baseline configuration.
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

  // The selected CUTLASS kernel uses an SM80 architecture tag, so compute
  // capability 8.0 is the minimum supported device. On Blackwell this runs as
  // a compatible baseline rather than using the native SM100 instruction path.
  cudaDeviceProp properties{};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return KernelStatus::kCudaError;
  }
  if (properties.major < 8) return KernelStatus::kUnsupportedDevice;

  // The legacy grouped-GEMM Arguments API requires mutable pointer arrays,
  // although the generated mainloop treats the A/B matrix payloads as read-only.
  std::vector<ElementA*> host_a;
  std::vector<ElementB*> host_b;
  std::vector<ElementOutput*> host_c;
  std::vector<ElementOutput*> host_d;
  std::vector<std::int64_t> host_lda;
  std::vector<std::int64_t> host_ldb;
  std::vector<std::int64_t> host_ldc;
  std::vector<std::int64_t> host_ldd;

  // ---------------------------------------------------------------------------
  // 4. Convert the MoE routing result into CUTLASS grouped-GEMM metadata
  // ---------------------------------------------------------------------------
  // Compact inactive experts (M_i == 0). Every remaining entry describes one
  // independent GEMM, and all host vectors preserve the same compacted order.
  for (std::uint32_t expert = 0; expert < args.experts; ++expert) {
    const auto m = args.tokens_per_expert[expert];
    if (m == 0) continue;
    if (args.a[expert] == nullptr || args.b[expert] == nullptr ||
        args.d[expert] == nullptr) {
      impl_->release();
      return KernelStatus::kInvalidArgument;
    }
    // Problem p computes D_p[M_i, N] = A_p[M_i, K] * B_p[K, N].
    impl_->host_problem_sizes.emplace_back(m, args.n, args.k);
    host_a.push_back(const_cast<ElementA*>(
        static_cast<ElementA const*>(args.a[expert])));
    host_b.push_back(const_cast<ElementB*>(
        static_cast<ElementB const*>(args.b[expert])));
    // The epilogue uses beta = 0, so C is never read for its numerical value.
    // Reusing D as C avoids allocating a separate source matrix.
    host_c.push_back(static_cast<ElementOutput*>(args.d[expert]));
    host_d.push_back(static_cast<ElementOutput*>(args.d[expert]));
    // A is row-major [M_i, K]. B is supplied in row-major [N, K] storage,
    // which is memory-equivalent to the column-major [K, N] view used above.
    // C and D are row-major [M_i, N].
    host_lda.push_back(args.k);
    host_ldb.push_back(args.k);
    host_ldc.push_back(args.n);
    host_ldd.push_back(args.n);
  }

  impl_->plan_stats.active_experts =
      static_cast<std::uint32_t>(impl_->host_problem_sizes.size());
  impl_->default_stream = static_cast<cudaStream_t>(args.stream);
  if (impl_->host_problem_sizes.empty()) {
    // A routing result with no active experts is valid and requires no launch.
    impl_->ready = true;
    return KernelStatus::kSuccess;
  }

  // CUTLASS chooses a persistent CTA count from the total grouped tile count
  // and the available GPU resources. A CTA can process multiple tiles during
  // one launch through CUTLASS's device-side ProblemVisitor.
  const auto block_count =
      Gemm::sufficient(impl_->host_problem_sizes.data(),
                       static_cast<int>(impl_->host_problem_sizes.size()));
  if (block_count <= 0) {
    impl_->release();
    return KernelStatus::kUnsupportedDevice;
  }
  impl_->plan_stats.threadblock_count = block_count;

  // Copy only the compacted problem description to the GPU. The actual A/B/D
  // matrix buffers were allocated by the caller and are referenced by pointer.
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

  // ---------------------------------------------------------------------------
  // 5. Build and initialize the CUTLASS launch arguments
  // ---------------------------------------------------------------------------
  // LinearCombination implements D = alpha * accumulator + beta * C.
  // With alpha=1 and beta=0, the accumulated GEMM result is written directly.
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

  // DeviceOnly scheduling normally needs little or no auxiliary workspace, but
  // querying CUTLASS keeps the wrapper correct for the selected kernel policy.
  impl_->plan_stats.workspace_bytes =
      impl_->gemm.get_workspace_size(*impl_->arguments);
  if (impl_->plan_stats.workspace_bytes > 0 &&
      cudaMalloc(reinterpret_cast<void**>(&impl_->workspace),
                 impl_->plan_stats.workspace_bytes) != cudaSuccess) {
    impl_->release();
    return KernelStatus::kCudaError;
  }

  // can_implement is CUTLASS's standard compatibility hook (validation is
  // limited for this grouped-kernel implementation). initialize converts the
  // Arguments object into launch-ready parameters; it does not run the GEMM.
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
  // ---------------------------------------------------------------------------
  // 6. Launch the already initialized grouped GEMM
  // ---------------------------------------------------------------------------
  // This is the path benchmark iterations should time. Inside gemm.run(),
  // CUTLASS launches the generated persistent grouped kernel. Each CTA uses a
  // ProblemVisitor to select an expert tile, performs the Tensor Core MMA and
  // epilogue, then advances through the flattened grouped tile space.
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
  // Convenience path for correctness checks and one-off calls. Benchmarks
  // should construct a plan once and call run() repeatedly instead.
  Bf16GroupedGemmPlan plan;
  auto status = plan.initialize(args);
  if (status != KernelStatus::kSuccess) return status;
  return plan.run(args.stream);
}

}  // namespace blackwell_moe
