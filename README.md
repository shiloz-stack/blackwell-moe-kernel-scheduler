# Blackwell MoE Kernel Scheduler

A kernel-level exploration of **persistent expert-tile scheduling** and **workload-aware dispatch** for irregular Mixture-of-Experts (MoE) inference on NVIDIA Blackwell GPUs.

> **Status:** Work in progress. The CUTLASS BF16 grouped baseline is validated on NVIDIA B200. An independent SM100a-native dense TMA/tcgen05/TMEM path is now implemented and awaiting its B200 correctness/performance gate.

## Motivation

MoE inference routes a different number of tokens to each expert. The resulting grouped GEMMs therefore contain many irregular and often small-`M` problems:

- popular experts receive many tokens while others receive few or none;
- static expert ordering can leave CTAs with uneven amounts of work;
- small expert matrices may not provide enough parallelism to occupy the GPU;
- scheduling overhead can outweigh useful computation in low-token regimes.

This project focuses on **scheduling inside the GPU kernel**. It does not implement a full inference engine or replace the model-level router. The inference engine supplies routed token counts; the kernel scheduler decides how expert output tiles are assigned to persistent CTAs.

## Project Goal

Build and evaluate a grouped-GEMM prototype that represents work as:

```text
(expert_id, tile_m, tile_n)
```

and compares several expert-tile scheduling policies under realistic routing imbalance:

1. baseline expert-order scheduling;
2. tile-count-aware static persistent assignment;
3. dynamic global work distribution;
4. active-expert compaction for sparse and small-`M` workloads;
5. workload-aware dispatch between scheduling policies.

The end goal is not a single hand-tuned kernel for one shape. It is a reproducible study of **when each scheduling strategy wins, why it wins, and what overhead it introduces**.

## Proposed Design

### 1. Routing-aware workload model

Synthetic and trace-driven workloads will be characterized using:

- coefficient of variation of expert token counts, `CV(M)`;
- `max(M) / mean(M)`;
- inactive-expert ratio;
- small-`M` expert ratio;
- expert tile-count imbalance.

Initial workload families will include uniform, heavy-hitter, and Zipf-like routing distributions. Support for real routing traces is planned.

### 2. Expert-tile scheduler

Each grouped GEMM is decomposed into independently schedulable output tiles. Planned policies include:

| Policy | Work assignment | Intended regime |
| --- | --- | --- |
| Expert order | Original grouped-GEMM order | Baseline |
| Sorted static persistent | Assign experts or tiles by estimated work | Moderate, predictable skew |
| Dynamic queue | CTAs acquire the next available expert tile | High routing imbalance |
| Active-expert compaction | Remove inactive experts before scheduling | Sparse or low-token batches |
| CLC-assisted scheduling | Explore cluster-level work redistribution | Blackwell-specific experiment |

Dynamic scheduling is not assumed to be universally faster. Queue traffic, atomics, synchronization, and extra control flow will be measured explicitly.

### 3. Workload-aware dispatch

The project will derive a simple crossover model from measured data instead of reporting only a best-case speedup. A candidate policy has the following form:

```text
if routing_skew < skew_threshold:
    use static_persistent
elif active_tiles < tile_threshold:
    use compacted_small_m
else:
    use dynamic_scheduler
```

Thresholds will be fitted and validated across held-out workload shapes rather than hard-coded from a single benchmark.

## Scope

| Layer | Responsibility in this project |
| --- | --- |
| MoE router / inference engine | Produces expert assignments and token counts; treated as input |
| Host dispatch | Builds grouped-GEMM metadata and selects a kernel policy |
| Kernel scheduler | Maps expert tiles to persistent CTAs |
| Math pipeline | Uses CUTLASS/CuTe primitives where appropriate |
| Evaluation harness | Generates workloads, checks correctness, profiles kernels, and reports results |

CUTLASS and CuTe provide the underlying GEMM and architecture primitives. The original work in this repository will focus on workload construction, expert-tile decomposition, scheduling policies, dispatch logic, and controlled evaluation. Any reused or modified upstream component will be identified explicitly.

## Evaluation Plan

### Baselines

- a CUTLASS grouped-GEMM baseline;
- original expert-order scheduling;
- static persistent scheduling;
- dynamic and compacted variants developed in this project.

### Workloads

Benchmarks will sweep:

- number of experts;
- routed tokens per expert;
- hidden and intermediate dimensions;
- top-`k` routing patterns;
- uniform, skewed, sparse, and Zipf-like expert loads;
- BF16 initially, with FP8 considered after the scheduling study is stable.

### Metrics

- end-to-end kernel latency, including scheduling overhead;
- median and tail latency across repeated runs;
- achieved throughput;
- CTA work imbalance and tail duration;
- active-expert and tile utilization;
- queue/atomic overhead for dynamic policies;
- correctness against a trusted grouped-GEMM reference.

All benchmark reports will record GPU model, clocks or power mode when relevant, CUDA version, CUTLASS revision, build flags, shapes, data type, warm-up procedure, and iteration count.

## Planned Repository Layout

```text
.
├── benchmarks/        # Workload generation and benchmark runners
├── include/           # Scheduler policies and shared kernel interfaces
├── src/               # CUDA/CuTe kernel implementations
├── tests/             # Correctness and scheduler unit tests
├── tools/             # Trace processing and result analysis
└── README.md
```

## Roadmap

- [x] Establish and validate a CUTLASS grouped-GEMM baseline on B200
- [x] Build the routing-aware workload generator
- [x] Record expert-level and tile-level imbalance metrics
- [x] Implement expert-tile decomposition and CPU scheduler simulation
- [ ] Validate the SM100a one-SM TMA/tcgen05/TMEM dense kernel on B200
- [ ] Validate the direct CuTe TMA/tcgen05/TMEM kernel on B200
- [ ] Implement static persistent scheduling
- [ ] Implement dynamic work distribution
- [ ] Add active-expert compaction
- [ ] Evaluate CLC-assisted work redistribution
- [ ] Profile scheduler overhead and CTA tail effects
- [ ] Derive and validate the crossover dispatch model
- [ ] Publish reproducible Blackwell benchmark results

## Success Criteria

The project will be considered successful when it can:

1. reproduce correctness across the full benchmark matrix;
2. explain performance using workload and scheduling measurements;
3. identify crossover regions where static, compacted, or dynamic scheduling is preferable;
4. demonstrate improvements on representative irregular workloads without hiding regressions;
5. reproduce reported results from a clean checkout with documented hardware and software versions.

## References

Implementation references and exact upstream revisions will be added as dependencies are introduced. Likely foundations include NVIDIA CUTLASS/CuTe documentation, grouped-GEMM examples, and Blackwell architecture programming guidance.

## License

A license will be selected before the first public release.


## Getting Started

Phase 1 is dependency-free on the CPU side and requires a C++17 compiler and CMake 3.24 or newer:

```bash
cmake -S . -B build -DBLACKWELL_MOE_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Inspect a workload and its expert-tile imbalance:

```bash
./build/moe_workload_bench \
  --distribution=zipf \
  --scheduler=static_persistent \
  --experts=64 \
  --tokens=4096 \
  --ctas=120
```

This CPU-side model reports tile padding, CTA work imbalance, tail ratio,
estimated utilization, and expert switches without claiming GPU latency. See
[`docs/scheduler_model.md`](docs/scheduler_model.md).

Generate a routing workload without third-party Python packages:

```bash
python3 python/generate_workloads.py \
  --distribution heavy_hitter \
  --experts 64 \
  --tokens 4096 \
  --output workload.json
```

CUDA remains disabled by default so the workload tools build without a CUDA
toolkit. Enable `BLACKWELL_MOE_ENABLE_CUDA` to fetch pinned CUTLASS `v4.6.0`,
build the BF16 grouped-GEMM baseline, GPU correctness test, and CUDA Event
benchmark. See [`docs/phase2_testing.md`](docs/phase2_testing.md).

On B100/B200/GB200, additionally enable
`BLACKWELL_MOE_ENABLE_SM100_NATIVE` to build the one-SM native TMA/tcgen05/TMEM
dense kernel, its CPU-reference correctness test, and its small-`M` benchmark.
See [`docs/sm100_native_dense.md`](docs/sm100_native_dense.md).

## Current Implementation

- [x] CMake project and optional CUDA boundary
- [x] Deterministic uniform, heavy-hitter, sparse, and Zipf workload generation
- [x] Routing and expert-tile imbalance metrics
- [x] Expert-tile decomposition
- [x] Host reference expert-order and static-persistent assignment
- [x] CPU persistent-scheduler simulator and CTA assignment metrics
- [x] Benchmark metadata CLI and unit tests
- [x] Pinned CUTLASS v4.6.0 BF16 grouped-GEMM baseline
- [x] Reusable grouped-GEMM plan with untimed metadata initialization
- [x] GPU correctness test against an FP32-accumulating CPU reference
- [x] CUDA Event benchmark with median/p95 latency and environment metadata
- [x] Explicit SM100a one-SM TMA/warp-specialized dense kernel configuration
- [x] CUTLASS persistent CLC tile-scheduler reference for the native dense path
- [x] Native dense correctness test and small-`M` CUDA Event benchmark
- [x] Direct CuTe TMA barriers, TMEM allocation, tcgen05 MMA, and TMEM-load epilogue
- [x] Direct CuTe CPU-reference correctness test and benchmark harness
- [ ] Project-owned device-side expert-tile persistent schedulers
