# Optimization Trajectory

This log preserves the reasoning boundary between implemented code and measured
evidence. Add one entry per causal experiment; keep failed hypotheses.

## T0 — Routing-aware baseline

**Hypothesis.** Grouped GEMM efficiency depends materially on expert routing
shape even when total tokens and `N/K` are fixed.

**Implementation.** Deterministic uniform, heavy-hitter, sparse, and Zipf
workloads; workload metrics; CUTLASS v4.6.0 BF16 grouped GEMM baseline.

**B200 evidence.** On B200 with 64 experts, 4096 total tokens, `N=7168`,
`K=2048`, 20 warm-ups, and 200 iterations:

| Distribution | Active experts | Median | Effective TFLOP/s |
| --- | ---: | ---: | ---: |
| Uniform | 64 | 0.543552 ms | 221.247 |
| Heavy-hitter | 64 | 0.701792 ms | 171.360 |
| Sparse | 32 | 0.394688 ms | 304.694 |
| Zipf | 64 | 0.688448 ms | 174.681 |

Correctness passed against the CPU reference with zero maximum error in the
recorded smoke case. These rows characterize the CUTLASS baseline scheduler;
they do not measure project-owned persistent scheduling.

**Interpretation.** Equal logical FLOPs do not imply equal tile efficiency.
Heavy-hitter and Zipf produce more padding and irregular small-`M` work, so the
project needs workload-aware scheduling and dispatch rather than a single
best-shape claim.

## T1 — Expert-tile scheduling model

**Hypothesis.** Flattening work to `(expert_id, tile_m, tile_n)` makes load
imbalance observable and lets persistent CTAs redistribute skewed expert work.

**Implementation.** CPU reference policies, exact expert-tile decomposition,
CTA work metrics, active-expert compaction, static persistent assignment, and a
dynamic GPU queue with chunked claims.

**Evidence level.** CPU tests and the GPU scheduler-probe harness are present.
The probe performs deterministic integer work, not GEMM, so no GEMM speedup is
claimed from it.

**Next gate.** Run the static/dynamic claim-size matrix on B200 and explain the
latency-versus-tail-balance crossover.

## T2 — Blackwell math-pipeline candidates

**Hypothesis.** Direct TMA/tcgen05/TMEM pipelines and 1-SM/2-SM specialization
can outperform the portable grouped baseline in different expert-size regimes.

**Implementation.** Direct CuTe teaching kernel, native 1-SM collective, native
2-SM TMA-multicast collective, warp-specialized/persistent configurations, and
correctness/benchmark harnesses.

**Evidence level.** Source and SM100a compile paths exist. Do not claim runtime
correctness or speed until the complete B200 optimization suite is archived.

**Next gate.** Validate each native kernel, measure the 1-SM/2-SM crossover,
then integrate the project-owned ExpertTile acquisition loop with a validated
GEMM pipeline.

## Entry template

```text
ID / date:
Target workload class:
Falsifiable hypothesis:
Single changed axis:
Fixed controls:
Correctness evidence:
Median / p95 evidence:
Profiler or balance evidence:
Regression / crossover:
Decision:
Next experiment:
```
