# B200 Optimization Suite

This suite separates three questions that must not be collapsed into one
speedup number:

1. Does each kernel produce correct output on SM100a?
2. Which Blackwell math pipeline is best for a given dense expert shape?
3. Does dynamic expert-tile assignment recover enough load balance to pay for
   its queue overhead?

## Implemented paths

| Path | Purpose | Blackwell mechanisms |
| --- | --- | --- |
| CUTLASS grouped baseline | Irregular MoE reference | Portable grouped GEMM |
| 1-SM collective | Optimized native reference | TMA, auto-staged warp specialization, tcgen05, TMEM, TMA epilogue, persistent CLC |
| 2-SM collective | Cooperative native reference | 2-SM tcgen05, TMA multicast, auto-staged warp specialization, TMEM, persistent CLC |
| Direct CuTe | Inspectable teaching kernel | Explicit TMA barriers, TMEM allocation, tcgen05 MMA, TMEM-to-register load |
| Scheduler probe | Project-owned policy experiment | Persistent CTAs, static assignment, atomic dynamic queue, chunked claims |

The scheduler probe intentionally uses deterministic integer work proportional
to `valid_m * valid_n`; it does not execute GEMM. Its latency measures queue and
load-balancing behavior in isolation. It must not be reported as a GEMM
speedup. The native collective benchmarks measure the math pipeline, while a
later integration combines project-owned expert-tile acquisition with that
pipeline.

## Build

```bash
cmake -S . -B build-gpu-make -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBLACKWELL_MOE_ENABLE_CUDA=ON \
  -DBLACKWELL_MOE_ENABLE_SM100_NATIVE=ON \
  -DBLACKWELL_MOE_CUDA_ARCHITECTURES=100 \
  -DBLACKWELL_MOE_BUILD_TESTS=ON

cmake --build build-gpu-make -j "$(nproc)"
```

The native translation units are compiled for `compute_100a/sm_100a`; the
architecture-accelerated suffix is required for tcgen05 and TMEM.

## Run everything once

```bash
./tools/run_b200_optimization_suite.sh \
  build-gpu-make \
  results/b200-optimization-suite
```

For a cheaper smoke run before the full matrix:

```bash
BLACKWELL_MOE_WARMUP=5 BLACKWELL_MOE_ITERATIONS=20 \
  ./tools/run_b200_optimization_suite.sh \
  build-gpu-make \
  results/b200-optimization-smoke
```

The script runs all correctness tests first and stops on the first failure. It
then writes:

- `environment.txt`: GPU, driver, CUDA, and CMake metadata;
- `correctness.log`: CPU, grouped-GEMM, scheduler, and SM100a gates;
- `grouped_baseline.csv`: routing-distribution baseline;
- `native_1sm_vs_2sm.csv`: 1-SM/2-SM crossover sweep;
- `direct_cute.csv`: explicit primitive path;
- `scheduler_probe.csv`: static/dynamic and claim-size sweep;
- a compressed `.tar.gz` archive next to the result directory.

## How to read the scheduler rows

- `median_ms` and `p95_ms` include the probe kernel but exclude host workload
  generation and the queue reset.
- `observed_cta_work_cv` is measured from device counters after execution.
- `tail_ratio = max(CTA work) / mean(CTA work)`; lower is better.
- `observed_utilization = mean(CTA work) / max(CTA work)`; higher is better.
- `claim_size=1` maximizes adaptability and atomic traffic; larger chunks
  amortize atomics but can reintroduce tail imbalance.

The desired result is a crossover, not a universal winner. Uniform workloads
should expose dynamic-queue overhead. Heavy-hitter and Zipf workloads should
show whether reduced tail imbalance compensates for that overhead. Sparse
workloads additionally test active-expert compaction because zero-token experts
produce no device work items.

## Runtime validation boundary

GitHub Actions compile every CUDA target for `sm_100a` in an official CUDA 13
development container, but Actions do not provide a B200. A green CI check
therefore proves compilation only. Correctness, latency, generated SASS, and
performance-counter claims remain unvalidated until this suite runs on B200.
