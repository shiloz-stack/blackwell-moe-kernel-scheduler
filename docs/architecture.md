# Architecture

The project separates routing, scheduling, and GEMM execution so scheduling policies can be compared without changing the workload or correctness reference.

```text
routing workload -> expert tile decomposition -> scheduler policy -> grouped GEMM
                         |                          |
                         +-> workload metrics      +-> CTA assignment metrics
```

The inference engine is outside the project boundary. It supplies token counts and data pointers. Host dispatch may select a policy, while the kernel scheduler maps `(expert_id, tile_m, tile_n)` work to persistent CTAs.

## Phase ownership

- Phase 1: deterministic workload construction, metrics, tile decomposition, host reference schedulers, and stable kernel arguments.
- Phase 2: pinned CUTLASS grouped-GEMM baseline plus device correctness checks.
- Phase 3: static persistent kernel scheduler.
- Phase 4: active-expert compaction and dynamic queue.
- Phase 5: measured crossover model and automatic dispatch.
- Phase 6: Blackwell-specific CLC, cluster, FP8, and pipeline experiments.

The Phase 1 CUDA entry point returns `kNotImplemented` deliberately. This prevents workload-only runs from being mistaken for GPU performance results.

