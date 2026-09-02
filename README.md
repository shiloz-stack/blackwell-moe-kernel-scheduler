# Workload-Aware MoE Kernels on NVIDIA Blackwell

<div align="center">
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&amp;logoColor=white" alt="Python 3.10+"/></a>
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.8+-76B900.svg?logo=nvidia&amp;logoColor=white" alt="CUDA 12.8+"/></a>
<a href="requirements-tirx.txt"><img src="https://img.shields.io/badge/TIRx%20%2F%20TVM-0.26-FF6F00.svg" alt="TIRx / TVM 0.26"/></a>
<a href="#b200-results"><img src="https://img.shields.io/badge/GPU-NVIDIA%20B200-111111.svg?logo=nvidia&amp;logoColor=76B900" alt="GPU: NVIDIA B200"/></a>
</div>

A kernel-level study of how **execution policy, expert-tile scheduling, and tile
shape interact with irregular MoE routing workloads** on NVIDIA B200 GPUs.

The project implements BF16 grouped GEMM kernels in TIRx on the Blackwell
`TMA -> tcgen05 -> TMEM` path. Rather than searching for one universal winner,
it measures where persistent execution, warp specialization, dynamic work
acquisition, CLC, and padding-aware tile selection help or hurt.

> **Status:** Nine project-owned kernel/policy candidates have passed their
> corresponding B200 correctness gates. The completed study contains 52 B200
> benchmark cases and 10,400 timed CUDA Event samples across uniform,
> heavy-hitter, sparse, and Zipf routing workloads. V1 reduced median kernel
> latency by **6.3-7.6%** versus non-persistent V0; later experiments exposed
> workload-dependent crossovers rather than a single best policy.

## Why this problem matters

After MoE routing, expert `i` receives `M_i` tokens and executes a different
GEMM:

```text
Expert i:  [M_i, K] x [K, N] -> [M_i, N]
```

The total token count may stay fixed while the distribution changes
substantially. Heavy-hitter and Zipf routing create many irregular small-`M`
experts; sparse routing leaves some experts inactive. These patterns change:

- the number of output tiles per expert;
- padding and useful-work ratio;
- the amount of work available to each CTA;
- the tradeoff between load balance and scheduling overhead.

This repository studies that tradeoff **inside the grouped-GEMM kernel**. The
model router, token permutation, NCCL expert-parallel communication, and full
inference-engine integration are outside the measured boundary.

## Kernel design

Each routed grouped GEMM is flattened into independent work items:

```text
(expert_id, tile_m, tile_n)
```

Every project-owned kernel uses the same core Blackwell data path:

```text
GMEM --TMA--> SMEM --tcgen05.mma--> TMEM
                                      |
                                      v
                         registers -> SMEM --TMA--> GMEM
```

The versions vary one primary execution or scheduling decision at a time:

| Version | Primary change | Work acquisition |
| --- | --- | --- |
| V0 | Non-persistent, double-buffered baseline | One CTA per expert tile |
| V0.5 | Reuse CTA-local TMEM, SMEM, and barriers | Static grid stride |
| V1 | Add producer/MMA/writeback warp specialization | Static grid stride |
| V2 | Acquire one tile at a time from a global queue | Atomic claim-1 |
| V3 | Amortize queue access with configurable claims | Atomic claim 1/2/4/8 |
| V4 | Preserve locality for coarse work, balance the tail | Hybrid coarse/fine |
| V5 | Use Blackwell Cluster Launch Control | Native TIRx CLC scheduler |
| V6a | Reduce padding with an all-M64 ablation | Static persistent |
| V6b | Keep M128 main work and move 1-64 row tails to M64 | Two-launch M128/M64 |

V0 and V0.5 already use TMA double buffering, `tcgen05`, and TMEM. The version
sequence therefore isolates execution and scheduling effects rather than
comparing a conventional CUDA GEMM with a Tensor Core implementation.

See [the TIRx kernel guide](docs/tirx_moe_kernel.md) for the pipeline,
warpgroup responsibilities, worklist invariants, and exact evidence boundary.

## B200 results

The main comparison uses one NVIDIA B200, BF16 inputs/output, 64 experts,
4096 routed tokens, `N=7168`, `K=2048`, seed 2026, 20 warmups, and 200 timed
iterations per case. Positive percentages below mean lower latency.

### Persistent execution and scheduling

| Routing workload | V1 vs V0 median | V2 dynamic vs V1 | V5 CLC vs V1 |
| --- | ---: | ---: | ---: |
| Uniform | +6.28% | +0.42% | +0.52% |
| Heavy-hitter | +7.64% | +0.70% | +0.83% |
| Sparse | +6.31% | +1.53% | +1.68% |
| Zipf | +6.54% | +1.26% | +1.40% |

Warp-specialized V1 produced the largest consistent gain. Dynamic claim-1 and
CLC improved some skewed cases, but only at roughly the 1% level. Larger V3
claims and the V4 locality hybrid generally regressed, showing that additional
scheduler complexity was not free.

### Padding-aware M128/M64 policy

| Routing workload | M128/M64 median | M128/M64 p95 | All-M64 median |
| --- | ---: | ---: | ---: |
| Uniform | +1.12% | +1.34% | -27.13% |
| Heavy-hitter | +1.97% | +1.94% | -16.76% |
| Sparse | -11.24% | -11.25% | -38.42% |
| Zipf | +0.89% | +1.21% | -12.96% |

The selective M128/M64 path helped heavy-hitter and Zipf, but regressed sparse
routing. The all-M64 ablation increased useful-work ratio while making every
workload slower. Reducing padded FLOPs alone was therefore insufficient: tile
count, launch overhead, pipeline cost, and physical memory traffic determined
the crossover.

For the Zipf split path, Nsight Compute measured **17.7% less logical TMA-load
traffic but 5.3% more DRAM reads** than V1. That tension helps explain why a
large theoretical padding reduction translated into only a 0.9% median gain.

## Optimization trajectory

The experiments produced four main conclusions:

1. **Reuse and pipeline specialization mattered first.** Persistent execution
   and warp specialization delivered a repeatable 6.3-7.6% improvement over V0.
2. **More dynamic scheduling was not automatically better.** Claim-1 and CLC
   offered modest gains; larger claims and hybrid control flow often cost more
   than the imbalance they recovered.
3. **The first hypothesis exposed a different bottleneck.** After scheduling
   gains plateaued, workload metrics showed only 37.6% and 39.0% useful work for
   M128 heavy-hitter and Zipf cases. This evidence expanded the original search
   space to tile-size specialization.
4. **Less padding still did not imply lower latency.** All-M64 was decisively
   slower, and selective M128/M64 had a workload-dependent crossover. The
   resulting direction is measured launch-time policy selection, not a
   universal replacement kernel.

The full hypothesis and evidence log is maintained in
[the optimization trajectory](docs/optimization_trajectory.md).

## Human-in-the-loop kernel-agent workflow

The project uses Codex as a bounded implementation and analysis partner. It is
not presented as a standalone autonomous kernel agent.

```text
knowledge + search space + evaluation contract
                    |
                    v
      hypothesize -> implement -> run cheap gates
                    |
                    v
       human-operated B200 benchmark / NCU
                    |
                    v
       analyze -> record -> accept, prune, or revise
```

The reusable [optimization skill](skills/optimize-blackwell-moe-kernels/SKILL.md)
defines the loop. [The search space](agent/search_space.yaml) bounds legal
transformations, and [the evaluation contract](agent/evaluation_contract.md)
separates CPU, compile-only, correctness, timing, and profiler evidence.

Deterministic tests—not the language model—decide correctness. The B200 operator
controls scarce hardware and final promotion. Automating remote job submission,
NCU ingestion, candidate ranking, and experiment-budget allocation remains
future work.

## Reproduce the experiments

### CPU-side workload and scheduler tests

```bash
cmake -S . -B build \
  -DBLACKWELL_MOE_ENABLE_CUDA=OFF \
  -DBLACKWELL_MOE_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The CPU model reports routing features, tile padding, CTA assignment, and
exact-once coverage. It does not claim GPU GEMM speedup.

### TIRx environment

```bash
python3 -m venv .venv-tirx
source .venv-tirx/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-tirx.txt
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

### Full B200 version suite

```bash
BLACKWELL_MOE_WARMUP=20 \
BLACKWELL_MOE_ITERATIONS=200 \
BLACKWELL_MOE_CASE_TIMEOUT=600 \
bash tools/run_b200_tirx_suite.sh results/b200-tirx-full
```

The suite records environment metadata, correctness logs, generated CUDA,
median/p95 CSV rows, and a compressed result archive.

### Focused padding experiment

```bash
BLACKWELL_MOE_WARMUP=20 \
BLACKWELL_MOE_ITERATIONS=200 \
BLACKWELL_MOE_CASE_TIMEOUT=600 \
bash tools/run_b200_padding_suite.sh results/b200-padding-aware
```

Use [the NCU helper](tools/profile_ncu.sh) for profiler collection. Detailed
methodology and timing boundaries are documented in
[benchmark methodology](docs/benchmark_methodology.md).

## Measurement boundary

- Reported numbers are **kernel-only CUDA Event latency**.
- Workload generation, allocation, and input initialization are outside timing.
- Queue initialization is outside timing; in-kernel acquisition is included.
- Both launches of the V6b M128/M64 path are inside one timed interval.
- Results do not include routing, token permutation, communication, or complete
  inference-engine overhead.
- The current `--kernel=auto` thresholds are bootstrap rules, not a validated
  production dispatcher. Held-out workload validation remains required.

## Repository map

```text
agent/       Search-space and evaluation contracts
baselines/   Optional pinned CUTLASS reference
benchmarks/  Workload configurations and legacy benchmark entry points
docs/        Architecture, experiment trajectory, and testing guides
python/      Versioned TIRx kernels, worklist planner, dispatch, and CLI
tests/       CPU/reference correctness and scheduler tests
tools/       B200 benchmark, profiling, and result-packaging scripts
```

CUTLASS v4.6.0 is retained only as an attributed external correctness and
performance baseline. All active MoE kernels in the version study are
project-owned TIRx implementations.

## Current limitations and next steps

- publish a compact, checksummed result bundle with CSV and extracted NCU data;
- fit dispatch thresholds on training workloads and validate them on held-out
  shapes and routing traces;
- include host dispatch and metadata preparation in an end-to-end measurement;
- automate B200 job submission, profiler ingestion, and experiment-state
  transitions while retaining human promotion approval.

## References

- [Modern GPU Programming for ML Systems](https://mlc.ai/modern-gpu-programming-for-mlsys/)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [Project architecture](docs/architecture.md)
- [Agent-assisted workflow](docs/agent_assisted_workflow.md)
