# Versioned TIRx Blackwell MoE Kernels

Project-owned GPU kernels are implemented in TIRx. CUTLASS is retained only as
an optional external baseline under `baselines/cutlass`.

## Common problem contract

The host planner compacts inactive experts and creates an expert-major list:

```text
(expert_id, tile_m_index, tile_n_index)
```

Each kernel multiplies the indices by its compile-time M/N tile size before
forming TMA coordinates; this keeps their alignment provable even though the
metadata is loaded at runtime. V0-V5 compute BF16 `128x128x64` work tiles.
V6 adds a BF16 `64x128x64` tcgen05 Layout-F tail path and combines it with V1
through two host-built buckets. Only valid routed rows enter logical-FLOP and
correctness metrics.

## Optimization journey

| Version | CTA lifetime | Warp specialization | Acquisition |
| --- | --- | --- | --- |
| V0 | One work tile | No | `tile_id = blockIdx.x` |
| V0.5 | Persistent | No | Static grid stride |
| V1 | Persistent | Yes | Static grid stride |
| V2 | Persistent | Yes | Atomic claim-1 queue |
| V3 | Persistent | Yes | Atomic chunk queue |
| V4 | Persistent | Yes | Coarse expert-major chunks plus claim-1 tail |
| V5 | CLC workers | Yes | Blackwell Cluster Launch Control |
| V6 | Persistent, two launches | Yes | Host M=128 main / M=64 tail buckets |

V0 and V0.5 retain TMA double buffering, tcgen05, TMEM and the TMA epilogue.
They remove only warp specialization and/or persistent execution.

V1 splits a CTA into two warpgroups:

| Role | Threads | Responsibility |
| --- | --- | --- |
| TMA producer | WG1 warp 3 | Load per-expert A/B tiles into double-buffered SMEM |
| MMA consumer | WG1 warp 0 | BF16 tcgen05 with FP32 TMEM accumulation |
| Writeback | WG0 | TMEM to registers to BF16 SMEM, then TMA store |

V2-V4 reserve WG1 warp 2 for one scheduler lane. That lane is the only code
allowed to update the queue. It publishes one `tile_id` through a CTA-local
mailbox guarded by a full/empty mbarrier pipeline. Loader, MMA and writeback
must all consume the same mailbox generation before it can be overwritten.
This prevents the fatal design error in which the roles independently claim
different tiles.

V4 divides the expert-major worklist into two disjoint regions. The main region
uses a coarse atomic chunk to preserve contiguous expert locality and amortize
atomics; the reserved tail uses claim-1 acquisition. Queue heads are initialized
to `(0, tail_begin)`, so no tile is duplicated or skipped.

V5 launches one CTA coordinate per work item. A resident CTA computes its own
coordinate while a CLC request attempts to cancel a pending CTA launch. On
success, the resident CTA inherits that coordinate as its next work item. It
uses TIRx `ClusterLaunchControlScheduler`; it is not an emulated atomic queue.

V6 keeps complete 128-row regions and 65--127 row tails on the validated V1
path. A final 1--64 row expert tail is moved to a separate M=64 worklist. Its
tcgen05 accumulator uses TMEM Layout F and a matching `.16x256b` register
fragment for the TMEM-to-SMEM epilogue. Both kernels share A/B/D, and CUDA
Event timing covers both launches. This removes half-tile padding without
forcing large experts to use twice as many M=64 work items.

## Evidence boundary

CPU tests cover worklist compaction, exact-once static/chunked/hybrid planning,
M=64 indexing, disjoint bucket coverage, version metadata and queue
initialization. V0-V5 have passed the B200 runtime gate. V6 remains a candidate
until its Layout-F source compiles, both standalone and composite correctness
checks pass, and the four-distribution B200 benchmark is archived. Do not claim
a V6 speedup from its theoretical padding reduction alone.

## Environment

```bash
python3 -m venv .venv-tirx
source .venv-tirx/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-tirx.txt

python3 -c "import torch, tvm, tvm.tirx; print(torch.__version__, tvm.__version__, torch.cuda.get_device_name())"
```

The CLC version requires the TIRx release containing
`tvm.tirx.lang.tile_scheduler.ClusterLaunchControlScheduler`.

## Test one version

Start with V0, then proceed in order so a synchronization failure has a small
diff surface:

```bash
PYTHONPATH=python python3 -m blackwell_moe_tirx.cli \
  --kernel=v0_nonpersistent \
  --smoke \
  --correctness-only \
  --dump-cuda=results/v0_nonpersistent.cu
```

Example V3 benchmark:

```bash
PYTHONPATH=python python3 -m blackwell_moe_tirx.cli \
  --kernel=v3_chunked \
  --claim-size=4 \
  --distribution=zipf \
  --experts=64 --tokens=4096 --n=7168 --k=2048 \
  --warmup=20 --iterations=200 --csv
```

Run the complete correctness and version matrix:

```bash
BLACKWELL_MOE_WARMUP=5 BLACKWELL_MOE_ITERATIONS=20 \
  ./tools/run_b200_tirx_suite.sh results/b200-tirx-smoke
```

After the smoke suite passes, rerun with the default 20 warmups and 200 timed
iterations. The script archives environment data, generated CUDA, correctness
logs and one combined CSV. Each case has a 600-second safety timeout; override
it with `BLACKWELL_MOE_CASE_TIMEOUT` if compilation on the machine is slower.

For the focused V1 versus V6 experiment:

```bash
BLACKWELL_MOE_WARMUP=20 BLACKWELL_MOE_ITERATIONS=200 \
  ./tools/run_b200_padding_suite.sh results/b200-padding-aware
```

The focused CSV contains V1, an all-M=64 ablation, and the production candidate
that launches the M=128 main bucket followed by the M=64 tail bucket.

## Measurement contract

CSV latency includes the selected GPU kernel and its in-kernel acquisition
overhead. Queue initialization occurs before the CUDA start event and is
therefore excluded; this is stated explicitly so an end-to-end dispatch study
can measure it separately. Report median, p95, effective logical TFLOP/s and
useful-work ratio for every routing distribution. V3 must sweep claim sizes
`1,2,4,8`.

The CSV also records `CV(M)`, maximum-over-mean M, inactive and small-M expert
ratios, expert-tile CV, maximum-over-mean expert tile count, and tiles per CTA.
These are the inputs used to fit the launch-level crossover model.

## Host launch dispatch

`python/blackwell_moe_tirx/dispatch.py` computes routing features without TVM
or CUDA and exposes `select_kernel`. The CLI accepts `--kernel=auto`, but its
current thresholds are bootstrap rules only. V5 is deliberately excluded from
automatic selection until the B200 matrix establishes whether and where CLC
beats the atomic and static paths.

## Upstream attribution

The warp-specialized math pipeline is adapted from Step 7 of the Apache-2.0
licensed MLC tutorial, *Modern GPU Programming for MLSys*. The MoE tensor
interface, expert worklist, atomic mailbox protocol, adaptive coarse/fine queue,
benchmark matrix and dispatch study are project-owned. V5 uses Apache TVM's
`ClusterLaunchControlScheduler` abstraction.
