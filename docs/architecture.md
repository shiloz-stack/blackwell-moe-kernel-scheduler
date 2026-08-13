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
- Phase 3A: SM100a-native dense collective bring-up with an explicit one-SM
  TMA/warp-specialized policy, tcgen05 math, and TMEM accumulation.
- Phase 3B: direct CuTe implementation that exposes TMA barriers, TMEM
  allocation, tcgen05 issue, and the epilogue data movement in project code.
  The first bring-up version accepts complete `128x256x64` tiles and writes
  FP32 so architecture synchronization can be validated before predication and
  output conversion are introduced.
- Phase 4: connect the validated native math pipeline to a static persistent
  expert-tile scheduler, followed by active-expert compaction and a dynamic
  queue.
- Phase 5: CLC-assisted work redistribution, measured crossover modeling, and
  automatic dispatch.
- Phase 6: 2SM clusters, FP8/block-scaled math, and additional pipeline tuning.

The Phase 2 baseline consumes host-side expert token counts and arrays of device
pointers. Initialization filters zero-token experts, copies CUTLASS problem
metadata to the GPU, and creates a reusable plan. Timed `run()` calls contain
only the grouped kernel launch. The baseline uses an SM80 Tensor Core kernel
compiled for the selected Blackwell target. Phase 3A keeps that baseline intact
and adds an independent SM100a-only dense kernel. CUTLASS implements its native
collective internals; project-owned expert-tile scheduling begins after the
native data path has passed correctness and profiling gates.
