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
- Phase 2: pinned CUTLASS v4.6.0 BF16 grouped-GEMM baseline, reusable
  device-metadata plan, CUDA Event benchmark, and CPU-reference correctness.
- Phase 3: static persistent kernel scheduler.
- Phase 4: active-expert compaction and dynamic queue.
- Phase 5: measured crossover model and automatic dispatch.
- Phase 6: Blackwell-specific CLC, cluster, FP8, and pipeline experiments.

The Phase 2 baseline consumes host-side expert token counts and arrays of device
pointers. Initialization filters zero-token experts, copies CUTLASS problem
metadata to the GPU, and creates a reusable plan. Timed `run()` calls contain
only the grouped kernel launch. The baseline uses an SM80 Tensor Core kernel
compiled for the selected Blackwell target; native SM100 `tcgen05` and CLC are
reserved for later experiments.
