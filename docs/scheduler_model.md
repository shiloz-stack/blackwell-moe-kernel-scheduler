# CPU Scheduler Model

The CPU scheduler model makes expert-tile policy behavior testable without a
GPU. It is a deterministic reference for work coverage, ordering, persistent CTA
traversal, and assignment metrics. It does not predict kernel latency.

## Tile model

Each active expert GEMM is decomposed into output tiles:

```text
(expert_id, tile_m, tile_n, valid_m, valid_n)
```

`valid_m` and `valid_n` describe the logical output region inside a tile. They
are smaller than the configured tile shape at matrix boundaries. The model uses
valid output elements as a work proxy; because all experts share `K`, multiplying
this value by `2*K` yields logical GEMM FLOPs.

## Reference policies

- `expert_order` preserves the input expert order and assigns flattened tiles
  with grid-stride persistent traversal (`tile_id % cta_count`).
- `static_persistent` orders experts by descending tile count, then uses the same
  grid-stride traversal. This isolates ordering effects from traversal effects.
- `dynamic_queue` assigns the next tile to the earliest available simulated CTA.
  It is an idealized reference and excludes atomic claim and metadata overhead.
- `compacted` currently shares the expert-order traversal because zero-token
  experts already produce no tiles. Device-side metadata compaction remains a
  later CUDA implementation step.
- `auto` falls back to expert order until measured GPU crossover thresholds are
  available.

## Metrics

The CLI reports:

- `CTA work CV`: coefficient of variation of valid output elements per CTA;
- `tail ratio`: maximum CTA work divided by mean CTA work;
- `estimated utilization`: total valid work divided by the product of CTA count
  and maximum CTA work;
- `useful-work ratio`: valid output elements divided by padded tile capacity;
- `expert switches`: adjacent changes of expert ID within each CTA's work list.

These values explain scheduling structure and padding, not hardware throughput.
The CUDA implementation must separately measure queue overhead, memory locality,
Tensor Core utilization, and CTA tail latency.

## Example

```bash
./build/moe_workload_bench \
  --distribution=zipf \
  --scheduler=static_persistent \
  --experts=64 \
  --tokens=4096 \
  --ctas=120
```
