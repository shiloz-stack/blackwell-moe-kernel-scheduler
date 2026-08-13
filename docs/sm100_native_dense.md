# SM100-Native Dense GEMM Bring-Up

This stage adds a deliberately small Blackwell-native GEMM before changing the
MoE scheduler. It separates two risks:

1. Can the repository build and execute an SM100a TMA/tcgen05/TMEM pipeline?
2. Can that pipeline later be driven by irregular `(expert, tile_m, tile_n)`
   work?

Only the first question is addressed here. The existing grouped-GEMM baseline
remains unchanged and is still the reference for irregular MoE workloads.

## Kernel contract

The kernel computes BF16 output with FP32 accumulation:

```text
D[M, N] = A[M, K] x transpose(B_storage[N, K])
```

The selected policy is intentionally explicit:

| Item | Selection |
| --- | --- |
| Architecture | `cutlass::arch::Sm100` compiled as `sm_100a` |
| Mainloop | `KernelTmaWarpSpecialized1SmSm100` |
| Epilogue | `TmaWarpSpecialized1Sm` |
| Tile scheduler | CUTLASS `PersistentScheduler` using CLC |
| MMA tile | `128 x 128 x 64` |
| Cluster | `1 x 1 x 1` |
| Input / output | BF16 / BF16 |
| Accumulator | FP32 |

CUTLASS owns the collective implementation of TMA transfers, the
warp-specialized producer/consumer pipeline, tcgen05 MMA issue, TMEM accumulator
handling, and persistent CLC tile acquisition. This repository owns the
explicit policy choice, shape, public plan interface, validation, benchmark
methodology, and the later MoE expert-tile integration. This distinction
matters: this stage is a native pipeline bring-up, not yet a handwritten CuTe
kernel or a project-owned scheduler.

## Build on B100/B200/GB200

The `a` suffix is required. `sm_100` alone does not enable architecture-
accelerated tcgen05/TMEM features. The native library therefore adds
`--generate-code=arch=compute_100a,code=sm_100a` explicitly.

```bash
cmake -S . -B build-sm100 -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBLACKWELL_MOE_ENABLE_CUDA=ON \
  -DBLACKWELL_MOE_ENABLE_SM100_NATIVE=ON \
  -DBLACKWELL_MOE_CUDA_ARCHITECTURES=100 \
  -DBLACKWELL_MOE_BUILD_TESTS=ON

cmake --build build-sm100 -j "$(nproc)"
```

This target is intentionally unavailable on SM103 and SM120. The portable
grouped baseline can still be built for those targets with the native option
disabled.

## Correctness gate

```bash
ctest --test-dir build-sm100 --output-on-failure

./build-sm100/test_sm100_dense_correctness \
  2>&1 | tee results/b200_sm100_dense_correctness.log
```

The native test executes a `128 x 128 x 64` GEMM and compares every BF16 output
against an FP32-accumulating CPU reference. Do not publish performance numbers
until this test passes.

## Initial benchmark sweep

First run one smoke case:

```bash
./build-sm100/sm100_dense_bench \
  --m=128 --n=7168 --k=2048 \
  --warmup=20 --iterations=200 --csv \
  2>&1 | tee results/b200_sm100_dense_smoke.csv
```

Then sweep the MoE-sensitive `M` dimension:

```bash
for m in 16 32 64 128 256 512 1024; do
  ./build-sm100/sm100_dense_bench \
    --m="${m}" --n=7168 --k=2048 \
    --seed=2026 --warmup=20 --iterations=200 --csv
done 2>&1 | tee results/b200_sm100_dense_m_sweep.csv
```

These rows establish the native kernel's small-`M` utilization curve. They are
not yet an apples-to-apples replacement for the grouped baseline because each
run contains one dense problem rather than all experts in one launch.

## What comes next

After B200 correctness and timing pass:

1. inspect the generated kernel with Nsight Compute and SASS tools;
2. implement the same dataflow directly with CuTe so TMA barriers, TMEM
   allocation, `tcgen05.mma`, and TMEM-to-register epilogue movement are visible;
3. preserve the validated collective kernel as a reference;
4. connect the native math pipeline to persistent expert-tile work assignment;
5. compare static and dynamic/CLC-assisted scheduling under routing skew.

The CUTLASS configurations follow NVIDIA's
[Blackwell collective-builder example](https://github.com/NVIDIA/cutlass/blob/v4.6.0/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu)
and the direct-CuTe follow-up will use the
[SM100 TMA tutorial](https://github.com/NVIDIA/cutlass/blob/v4.6.0/examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu)
as an API reference.
