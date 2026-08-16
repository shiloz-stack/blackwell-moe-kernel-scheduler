# Evaluation Contract

This file defines the trusted evaluator for the agent-assisted optimization
workflow. Codex may propose and implement candidates, but a human operator runs
the scarce B200 gate and decides whether evidence is accepted.

## Evidence levels

| Label | What it proves | What it does not prove |
| --- | --- | --- |
| CPU model | Workload and assignment logic | GPU latency or Tensor Core behavior |
| SM100a compile | Source and architecture APIs compile | Runtime correctness or speed |
| Scheduler probe | Exact-once coverage, queue cost, CTA balance | GEMM throughput |
| B200 correctness | Numerical validity for tested shapes | General performance superiority |
| B200 timing | Latency for recorded conditions | A universal crossover rule |
| NCU/NSYS profile | Mechanistic evidence for a bottleneck | End-to-end serving benefit by itself |

## Correctness gate

Reject a candidate before performance comparison if any of these fail:

- trusted CPU or CUTLASS reference comparison;
- exact-once visit count for every generated expert tile;
- boundary shapes and inactive experts;
- finite output and documented BF16/FP32 tolerances;
- CUDA error checks and sanitizer checks when available.

## Timing gate

Record GPU model, compute capability, driver, CUDA, CUTLASS revision, build
flags, clocks or power mode, dtype, full shape, routing seed/distribution,
warm-up count, measured iterations, and timing boundary.

Use the same generated workload and math path when isolating a scheduler
change. Report median and p95. Include preprocessing or dispatch overhead when
the candidate requires it in the production path; otherwise label the number
as kernel-only.

## Promotion decision

Promote a candidate only when:

1. all correctness gates pass;
2. the repeated timing result is stable enough to distinguish the change;
3. the claimed benefit appears in its target workload class;
4. regressions and crossover points are recorded;
5. a profiler or balance metric supports the causal explanation;
6. a dispatch rule derived from results is checked on held-out shapes.

Do not delete negative results. They constrain the search space and teach the
next Codex iteration which hypothesis failed.

## Result handoff

The B200 operator should return the raw CSV/log files plus environment metadata
and a compressed archive. The next optimization decision must cite those
artifacts, not a remembered best number. Never expose access tokens, SSH keys,
or cloud credentials in logs or commits.
