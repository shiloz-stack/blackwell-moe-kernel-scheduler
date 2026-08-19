# CUTLASS External Baseline

This directory contains the pinned CUTLASS v4.6.0 BF16 grouped-GEMM used only
as an external correctness and performance reference. Project-owned kernels are
implemented in TIRx under `python/blackwell_moe_tirx/kernels`.

The baseline is disabled by default. Build it explicitly with:

```bash
cmake -S . -B build-cutlass \
  -DBLACKWELL_MOE_ENABLE_CUDA=ON \
  -DBLACKWELL_MOE_BUILD_CUTLASS_BASELINE=ON \
  -DBLACKWELL_MOE_BUILD_TESTS=ON
cmake --build build-cutlass -j
```

The relevant targets are `moe_cutlass_baseline_bench` and
`test_cutlass_baseline_correctness`. Keeping the source here makes previously
reported B200 numbers reproducible without making CUTLASS part of the active
TIRx implementation path.
