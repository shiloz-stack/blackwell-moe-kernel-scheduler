# Benchmark Methodology

## Required metadata

Every published result must include the GPU model, CUDA toolkit, driver, CUTLASS revision, compiler flags, clocks or power mode when controlled, data type, GEMM shapes, routing distribution, warm-up count, measured iterations, and scheduler policy.

## Measurement rules

1. Validate output against a trusted reference before timing a new shape.
2. Time the complete kernel path, including device-side scheduling overhead.
3. Report median and tail latency, not only the best iteration.
4. Keep the workload and math pipeline identical when comparing schedulers.
5. Publish regressions and crossover regions alongside best-case improvements.
6. Separate host preprocessing time from GPU kernel time, then report end-to-end dispatch when the policy needs preprocessing.

Phase 1 reports workload metadata only. Its CLI output must not be presented as GPU throughput or latency.

## Phase 2 baseline boundary

`moe_cutlass_baseline_bench` measures one reusable CUTLASS grouped-GEMM plan with CUDA
Events. Memory allocation, routing generation, active-expert filtering,
metadata copies, and CUTLASS initialization occur before timing. Report this as
kernel-only latency. End-to-end dispatch measurements will be added when host
or device preprocessing policies are compared.
