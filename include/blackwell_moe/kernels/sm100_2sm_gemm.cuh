#pragma once

#include "blackwell_moe/kernels/sm100_dense_gemm.cuh"

namespace blackwell_moe {

// Reusable reference plan for Blackwell's cooperative two-SM MMA path.
// Compared with Sm100DenseGemmPlan, this variant doubles the MMA M extent,
// forms a 2x1x1 CTA cluster, and selects CUTLASS's 2-SM TMA mainloop and
// epilogue schedules.  CUTLASS lowers those schedules to TMA multicast and
// 2-SM tcgen05 operations while the persistent scheduler obtains output tiles
// through Cluster Launch Control (CLC).
class Sm100TwoSmGemmPlan {
 public:
  Sm100TwoSmGemmPlan();
  ~Sm100TwoSmGemmPlan();

  Sm100TwoSmGemmPlan(const Sm100TwoSmGemmPlan&) = delete;
  Sm100TwoSmGemmPlan& operator=(const Sm100TwoSmGemmPlan&) = delete;
  Sm100TwoSmGemmPlan(Sm100TwoSmGemmPlan&& other) noexcept;
  Sm100TwoSmGemmPlan& operator=(Sm100TwoSmGemmPlan&& other) noexcept;

  KernelStatus initialize(const Sm100DenseGemmArguments& args);
  KernelStatus run(void* stream = nullptr);
  Sm100DenseGemmPlanStats stats() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

KernelStatus launch_sm100_2sm_gemm(const Sm100DenseGemmArguments& args);
const char* sm100_2sm_kernel_name();

}  // namespace blackwell_moe
