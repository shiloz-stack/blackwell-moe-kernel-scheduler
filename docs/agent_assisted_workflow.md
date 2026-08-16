# Agent-Assisted Optimization Workflow

This repository uses Codex as a structured implementation and analysis partner;
it does not contain a standalone autonomous kernel agent.

```text
human kernel knowledge
        |
        v
workload taxonomy + legal search space + evaluation contract
        |
        v
Codex: hypothesize -> implement -> run cheap gates -> prepare B200 packet
        |                                      |
        |                                      v
        +<---- measured logs/CSV ---- human-operated B200 evaluator
        |
        v
interpret -> preserve trajectory -> choose the next bounded experiment
```

## Division of responsibility

The human supplies architecture judgment, defines useful workload classes,
controls access to scarce hardware, reviews anomalies, and accepts or rejects
claims. Codex searches within explicit constraints, edits candidates, maintains
tests, prepares reproducible commands, and turns profiler or benchmark results
into the next falsifiable hypothesis.

The reusable policy is encoded in
[`skills/optimize-blackwell-moe-kernels/SKILL.md`](../skills/optimize-blackwell-moe-kernels/SKILL.md).
Legal transformations and objectives live in
[`agent/search_space.yaml`](../agent/search_space.yaml), while
[`agent/evaluation_contract.md`](../agent/evaluation_contract.md) protects the
correctness and measurement boundary.

## Why this is agent-ready

The workflow exposes the pieces a kernel agent needs:

- a workload taxonomy instead of one favorite shape;
- a bounded transformation space instead of arbitrary code mutation;
- hard correctness gates before performance reward;
- persistent experiment history, including failures;
- workload-specific objectives and held-out dispatch validation;
- a trusted hardware evaluator whose outputs can feed the next iteration.

This is intentionally described as **human-in-the-loop, agent-assisted kernel
optimization**. Automating remote job submission, result ingestion, candidate
ranking, and budget allocation would be required before calling it a standalone
kernel agent.
