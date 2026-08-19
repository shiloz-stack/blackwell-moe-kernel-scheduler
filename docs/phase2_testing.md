# Phase 2: CUTLASS BF16 Baseline Testing

Phase 2 adds the first executable GPU baseline. It uses NVIDIA CUTLASS `v4.6.0` `device::GemmGrouped` with BF16 inputs/output and FP32 accumulation. The kernel is configured with an SM80 Tensor Core mainloop and compiled for the selected target GPU. This provides a portable grouped-GEMM reference on Blackwell; it is not yet a native SM100 `tcgen05` kernel.

## Requirements

- Linux
- CMake 3.24 or newer
- C++17 host compiler
- CUDA Toolkit 12.8 or newer for B100/B200/GB200 (SM100)
- Git access during the first configure so CMake can fetch CUTLASS `v4.6.0`

For B300 (SM103), use CUDA 13.0 or newer. For GeForce RTX 50-series (SM120), override the architecture as shown below; SM100 binaries are not compatible with SM120.

## Configure and build

### B100/B200/GB200 (SM100)

```bash
cmake -S . -B build \
  -DBLACKWELL_MOE_ENABLE_CUDA=ON \
  -DBLACKWELL_MOE_BUILD_CUTLASS_BASELINE=ON \
  -DBLACKWELL_MOE_CUDA_ARCHITECTURES=100 \
  -DBLACKWELL_MOE_BUILD_TESTS=ON
cmake --build build -j
```

### RTX 50-series (SM120)

```bash
cmake -S . -B build \
  -DBLACKWELL_MOE_ENABLE_CUDA=ON \
  -DBLACKWELL_MOE_BUILD_CUTLASS_BASELINE=ON \
  -DBLACKWELL_MOE_CUDA_ARCHITECTURES=120 \
  -DBLACKWELL_MOE_BUILD_TESTS=ON
cmake --build build -j
```

For B300, use the same command with
`-DBLACKWELL_MOE_CUDA_ARCHITECTURES=103` and CUDA 13.0+.

If CMake rejects the architecture number, the installed CUDA toolkit is too old for that target.

## Correctness gate

Run all CPU and GPU tests:

```bash
ctest --test-dir build --output-on-failure
```

Run only the grouped-GEMM test:

```bash
./build/test_cutlass_baseline_correctness
```

The test covers irregular expert sizes `[7, 0, 13, 31]`, filters the inactive expert, computes each active expert GEMM in one grouped launch, and compares BF16 output with an FP32-accumulating CPU reference. Do not collect performance results until this test passes.

## Baseline benchmark

Start with the small default problem:

```bash
./build/moe_cutlass_baseline_bench
```

Then sweep routing distributions using the same matrix dimensions:

```bash
for distribution in uniform heavy_hitter sparse zipf; do
  ./build/moe_cutlass_baseline_bench \
    --distribution=${distribution} \
    --experts=64 \
    --tokens=4096 \
    --n=7168 \
    --k=2048 \
    --warmup=10 \
    --iterations=100 \
    --csv
done
```

The large configuration allocates one `[N, K]` BF16 weight matrix per active expert. Check available GPU memory before running it.

The benchmark initializes A/B with deterministic nonzero BF16 values and reports median and p95 CUDA Event latency, effective TFLOP/s, active experts, launched threadblocks, GPU model, compute capability, CUDA runtime, and CUTLASS revision. Allocation, data initialization, metadata construction, and CUTLASS initialization occur before timing.

## Information to send back

Please return:

1. complete CMake configure output;
2. the first compiler error, including approximately 30 lines around it, if compilation fails;
3. `ctest --test-dir build --output-on-failure` output;
4. benchmark CSV rows;
5. `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv` output.
