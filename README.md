# Blackwell MoE Kernel Scheduler

A kernel-level exploration of **persistent expert-tile scheduling** and **workload-aware dispatch** for irregular Mixture-of-Experts (MoE) inference on NVIDIA Blackwell GPUs.

> **Status:** Work in progress. Seven project-owned TIRx versions now span a
> non-persistent double-buffered kernel, static/dynamic/chunked/hybrid
> persistent scheduling, warp specialization, and Blackwell CLC. All seven pass
> TVM 0.26 SM100a lowering and await the ordered B200 numerical/deadlock/
> performance gate. The previously validated
> CUTLASS BF16 grouped GEMM is retained only as an optional external baseline.

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

## Design

### 1. Routing-aware workload model

Synthetic and trace-driven workloads will be characterized using:

- coefficient of variation of expert token counts, `CV(M)`;
- `max(M) / mean(M)`;
- inactive-expert ratio;
- small-`M` expert ratio;
- expert tile-count imbalance.

Initial workload families will include uniform, heavy-hitter, and Zipf-like routing distributions. Support for real routing traces is planned.

### 2. Expert-tile scheduler

Each grouped GEMM is decomposed into independently schedulable output tiles. Implemented experimental policies include:

| Policy | Work assignment | Intended regime |
| --- | --- | --- |
| Expert order | Original grouped-GEMM order | Baseline |
| Sorted static persistent | Assign experts or tiles by estimated work | Moderate, predictable skew |
| Dynamic queue | CTAs acquire the next available expert tile | High routing imbalance |
| Active-expert compaction | Remove inactive experts before scheduling | Sparse or low-token batches |
| CLC-assisted scheduling | Explore cluster-level work redistribution | Blackwell-specific experiment |

Dynamic scheduling is not assumed to be universally faster. Queue traffic, atomics, synchronization, and extra control flow will be measured explicitly.

### 3. Workload-aware dispatch

The project includes a host-launch bootstrap policy and will fit its crossover
thresholds from measured data instead of reporting only a best-case speedup.
The auditable policy has the following form:

```text
if routing_skew < skew_threshold:
    use static_persistent
elif active_tiles < tile_threshold:
    use compacted_small_m
else:
    use dynamic_scheduler
```

The executable scaffold is in `python/blackwell_moe_tirx/dispatch.py` and can
be selected with `--kernel=auto`. Its current thresholds are explicitly
bootstrap values, not performance claims. They will be fitted and validated
across held-out workload shapes rather than hard-coded from a single benchmark.

## Scope

| Layer | Responsibility in this project |
| --- | --- |
| MoE router / inference engine | Produces expert assignments and token counts; treated as input |
| Host dispatch | Builds grouped-GEMM metadata and selects a kernel policy |
| Kernel scheduler | Maps expert tiles to persistent CTAs |
| Math pipeline | Project-owned TIRx using TMA, tcgen05 and TMEM |
| Evaluation harness | Generates workloads, checks correctness, profiles kernels, and reports results |

All active project kernels are written in TIRx. CUTLASS is not part of their
implementation; its pinned source remains under `baselines/cutlass` solely for
reproducible external comparison. Reused TIRx pipeline structures and compiler
abstractions are attributed explicitly.

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
├── baselines/         # Optional external reference implementations
├── benchmarks/        # CPU/CUDA legacy benchmark runners
├── include/           # Scheduler policies and shared kernel interfaces
├── python/            # Versioned TIRx kernels, planner, CLI, and B200 runner
├── src/               # CPU model and legacy CUDA probes/references
├── tests/             # Correctness and scheduler unit tests
├── tools/             # Trace processing and result analysis
└── README.md
```

## Roadmap

- [x] Establish and validate a CUTLASS grouped-GEMM baseline on B200
- [x] Build the routing-aware workload generator
- [x] Record expert-level and tile-level imbalance metrics
- [x] Implement expert-tile decomposition and CPU scheduler simulation
- [x] Implement V0 non-persistent double-buffered TIRx MoE GEMM
- [x] Implement V0.5 static-persistent single-warpgroup TIRx MoE GEMM
- [x] Implement V1 static-persistent warp-specialized TIRx MoE GEMM
- [x] Implement V2 claim-1 dynamic acquisition with CTA work publication
- [x] Implement V3 chunked dynamic acquisition
- [x] Implement V4 coarse-main/fine-tail locality-aware acquisition
- [x] Implement V5 Blackwell CLC acquisition
- [ ] Validate V0 through V5 in order on B200
- [x] Implement and test static persistent GPU tile assignment
- [x] Implement and test dynamic GPU work distribution with chunked claims
- [x] Add active-expert compaction to the generated device work list
- [ ] Measure CLC-assisted work redistribution against atomic queues
- [ ] Profile scheduler overhead and CTA tail effects
- [x] Implement host-side routing features and a bootstrap crossover policy
- [ ] Fit and validate the crossover thresholds from B200 measurements
- [ ] Publish reproducible Blackwell benchmark results

## Success Criteria

The project will be considered successful when it can:

1. reproduce correctness across the full benchmark matrix;
2. explain performance using workload and scheduling measurements;
3. identify crossover regions where static, compacted, or dynamic scheduling is preferable;
4. demonstrate improvements on representative irregular workloads without hiding regressions;
5. reproduce reported results from a clean checkout with documented hardware and software versions.

## Agent-Assisted Optimization

Development follows a **human-in-the-loop, agent-assisted kernel optimization
workflow**. The workload taxonomy, legal transformation space, correctness
gates, and B200 measurement contract are explicit so Codex can propose and
implement bounded experiments without turning benchmark results into unsupported
claims. The human operator remains responsible for trusted B200 execution and
promotion decisions.

See [`docs/agent_assisted_workflow.md`](docs/agent_assisted_workflow.md), the
versioned [`agent/search_space.yaml`](agent/search_space.yaml), and the reusable
[`optimize-blackwell-moe-kernels` skill](skills/optimize-blackwell-moe-kernels/SKILL.md).
This is not presented as a standalone autonomous kernel agent.

Agent optimization is intentionally sequenced after the project-owned TIRx
baseline. The current implementation first fixes the math pipeline and proves
correctness; the later agent experiment will vary only scheduler policy and
queue claim size under a bounded evaluation contract.

## TIRx Blackwell Path

The versioned implementations live in
[`python/blackwell_moe_tirx/kernels`](python/blackwell_moe_tirx/kernels). They
compile routed expert counts into `(expert_id, tile_m, tile_n)` work items and
use BF16 `128x128x64` and `64x128x64` Blackwell math paths while changing
execution, acquisition, and padding policy:

- double-buffered TMA operand movement;
- FP32 `tcgen05` accumulation in Tensor Memory;
- producer/MMA/writeback warp specialization;
- TMEM-to-register BF16 epilogue and TMA store;
- CPU-testable work-list planning and a B200 correctness/median/p95 harness.

The optimization sequence is V0 non-persistent, V0.5 persistent without warp
specialization, V1 static warp-specialized, V2 dynamic claim-1, V3 chunked,
V4 locality-aware coarse/fine, V5 CLC, and V6 padding-aware M=128/M=64
bucketing with tcgen05 Layout F. See
[`docs/tirx_moe_kernel.md`](docs/tirx_moe_kernel.md) for exact responsibility,
evidence boundaries, and B200 commands.

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

CUDA remains disabled by default so workload tools build without a toolkit.
The active GPU path is Python/TIRx and does not require the CMake CUDA targets.
To reproduce the external CUTLASS numbers, explicitly enable both
`BLACKWELL_MOE_ENABLE_CUDA` and `BLACKWELL_MOE_BUILD_CUTLASS_BASELINE`; see
[`baselines/cutlass/README.md`](baselines/cutlass/README.md).

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
- [x] Pinned CUTLASS v4.6.0 external BF16 grouped-GEMM baseline
- [x] Reusable grouped-GEMM plan with untimed metadata initialization
- [x] GPU correctness test against an FP32-accumulating CPU reference
- [x] CUDA Event benchmark with median/p95 latency and environment metadata
- [x] Explicit SM100a one-SM TMA/warp-specialized dense kernel configuration
- [x] CUTLASS persistent CLC tile-scheduler reference for the native dense path
- [x] Native dense correctness test and small-`M` CUDA Event benchmark
- [x] Direct CuTe TMA barriers, TMEM allocation, tcgen05 MMA, and TMEM-load epilogue
- [x] Direct CuTe CPU-reference correctness test and benchmark harness
- [x] Two-SM TMA-multicast/tcgen05 collective reference and benchmark
- [x] Project-owned device-side expert-tile persistent schedulers
- [x] Exact-once GPU scheduler correctness and observed CTA-load metrics
- [x] CPU-tested TIRx MoE work-list planner and active-expert compaction
- [x] Versioned V0/V0.5/V1 TIRx ablations
- [x] V2/V3 atomic acquisition with one scheduler and a shared work mailbox
- [x] V4 coarse expert-major main queue plus claim-1 tail queue
- [x] V5 native TIRx Cluster Launch Control scheduler
- [x] V6 Layout-F M=64 tail kernel and padding-aware M=128/M=64 bucketing
- [x] Unified version-selecting correctness/generated-CUDA/median/p95 harness

Run the active V0-V6 TIRx correctness and performance matrix with
[`tools/run_b200_tirx_suite.sh`](tools/run_b200_tirx_suite.sh). The older CuTe
reference and scheduler-probe matrix remains available through
[`tools/run_b200_optimization_suite.sh`](tools/run_b200_optimization_suite.sh)
and is documented in
[`docs/b200_optimization_suite.md`](docs/b200_optimization_suite.md).
