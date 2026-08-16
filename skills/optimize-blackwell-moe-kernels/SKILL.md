---
name: optimize-blackwell-moe-kernels
description: Guide a measured, human-in-the-loop optimization cycle for this Blackwell MoE kernel project. Use when Codex plans, implements, benchmarks, profiles, or interprets changes to workload taxonomy, expert-tile scheduling, CuTe/TMA/tcgen05/TMEM pipelines, persistent kernels, CLC, 1-SM/2-SM variants, or workload-aware dispatch in this repository.
---

# Optimize Blackwell MoE Kernels

Treat kernel optimization as constrained experimental research. Keep the human
responsible for B200 execution and acceptance; use Codex to structure the search,
implement candidates, check invariants, and interpret returned evidence.

## Establish the experiment contract

1. Read `agent/search_space.yaml` before proposing a transformation.
2. Read `agent/evaluation_contract.md` before changing a benchmark or reporting
   a result.
3. Read `docs/optimization_trajectory.md` before selecting the next experiment.
4. Inspect the relevant implementation and tests; do not reason from filenames
   or roadmap checkboxes alone.
5. State one falsifiable hypothesis and its target workload class.

If the proposed change is outside the declared search space, explain why the
space should expand before implementing it.

## Preserve project boundaries

- Treat routing decisions and expert-parallel collectives as inference-engine
  inputs, not features of this kernel repository.
- Distinguish host launch-level policy selection from device-side tile
  scheduling.
- Distinguish the scheduler probe from GEMM. The probe measures assignment and
  queue behavior using synthetic integer work; never report it as GEMM speedup.
- Keep the validated CUTLASS grouped GEMM as a correctness and performance
  baseline while developing project-owned Blackwell paths.

## Run one optimization cycle

1. **Classify the workload.** Record expert count, token count, distribution,
   `CV(M)`, `max(M)/mean(M)`, inactive ratio, small-`M` ratio, tile count, and
   useful-work ratio where available.
2. **Select one axis.** Change scheduler policy, queue claim size, compaction,
   tile shape, pipeline depth, 1-SM/2-SM mode, or another declared axis. Avoid
   changing multiple causal variables in one comparison.
3. **Predict the tradeoff.** Name both the expected benefit and cost, such as
   lower CTA tail imbalance versus more atomics.
4. **Implement with invariants.** Preserve exact-once tile coverage, boundary
   predication, barrier phase ownership, TMA visibility, TMEM lifetime, stream
   semantics, and the benchmark timing boundary.
5. **Pass the cheapest gates first.** Run CPU tests, reference correctness, and
   SM100a compilation before requesting scarce B200 time.
6. **Produce a B200 command packet.** Give copy-paste build/test commands,
   expected output files, and archive instructions. Do not assume direct access
   to the user's remote machine.
7. **Interpret evidence.** Compare median and p95 latency, correctness, useful
   throughput, CTA balance, queue overhead, and profiler counters appropriate to
   the hypothesis.
8. **Record the decision.** Update `docs/optimization_trajectory.md` with the
   hypothesis, immutable configuration, result, interpretation, and next action.

## Promotion rules

- Reject any candidate that fails correctness, exact-once coverage, or produces
  non-finite output.
- Require repeated measurements and identical workloads for latency claims.
- Report regressions and crossover regions, not only the best row.
- Hold out workload shapes when fitting dispatch thresholds.
- Count dispatch, compaction, metadata, and workspace costs when they are part of
  the proposed production path.
- Label compile-only, simulator-only, scheduler-probe, and B200-measured evidence
  separately.

## Blackwell review checklist

For direct CuTe or TIRx-style kernels, verify the whole movement chain:

```text
GMEM --TMA--> SMEM --tcgen05.mma--> TMEM --copy--> registers --store--> GMEM
```

Check producer/consumer warp roles, mbarrier arrival counts and phase changes,
pipeline-slot reuse, cluster/multicast assumptions, TMEM allocation/deallocation,
tail-tile validity, and architecture target (`sm_100a` where required).

For persistent scheduling, verify work-item construction, active-expert
compaction, CTA termination, queue reset outside the timed region when intended,
claim-size semantics, and that every tile is visited exactly once.

## Reporting language

Describe the process as a **human-in-the-loop, agent-assisted kernel optimization
workflow**. Do not claim that this repository implements a standalone autonomous
kernel agent. Claim a measured optimization only after its result is committed
with the evaluation conditions needed to reproduce it.
