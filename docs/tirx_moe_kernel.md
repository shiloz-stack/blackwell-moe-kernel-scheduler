# TIRx MoE Kernel Baseline

This repository now contains a project-owned Blackwell MoE grouped-GEMM path
in TIRx. It is intentionally the implementation baseline that comes **before**
agent-driven tuning.

## What is implemented

The host planner converts routed token counts into a compact work list:

```text
(expert_id, tile_m_offset, tile_n_offset)
```

Inactive experts generate no work. Every active expert is decomposed into
`128 x 128` output tiles. The initial kernel launches 148 persistent CTAs (one
per B200 SM), and CTA `bx` walks tiles `bx`, `bx + 148`, and so on. This is the
canonical **static persistent** policy; it has no global atomic queue.

Inside each CTA, two warpgroups have specialized roles:

| Role | Threads | Responsibility |
| --- | --- | --- |
| TMA producer | warpgroup 1, warp 3 | Move expert A/B tiles from HBM to double-buffered SMEM |
| MMA consumer | warpgroup 1, warp 0 | Issue BF16 `tcgen05` MMA and accumulate FP32 in TMEM |
| Writeback | warpgroup 0 | Move TMEM to registers, cast to BF16, stage in SMEM, and TMA-store to HBM |

Four full/empty barriers coordinate the pipeline:

```text
TMA --tma2mma--> MMA --mma2ld--> writeback
TMA <--mma2tma-- MMA <--ld2mma-- writeback
```

The input A tensor is padded with zeros to a 128-row boundary. That lets the
first kernel use full TMA tiles; only valid routed-token rows participate in the
correctness comparison and logical-FLOP metric.

## Evidence boundary

The workload planner and its metadata invariants are covered by CPU-only unit
tests. The TIRx source can only be lowered and executed with a TIRx-enabled TVM
build and a Blackwell GPU. Until the B200 gate below passes, the repository
must describe the kernel as **implemented and awaiting hardware validation**,
not as correct or faster than the CUTLASS baseline.

Dynamic atomic claiming, CLC, two-CTA collectives, and agent parameter search
are deliberately outside this first gate. They should be added only after the
same static kernel passes generated-code inspection and numerical correctness.

## B200 environment

Use the CUDA 13 B200 machine and install the TIRx compiler dependencies in a
virtual environment. Install a CUDA-enabled PyTorch wheel appropriate for the
machine image separately.

```bash
python3 -m venv .venv-tirx
source .venv-tirx/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-tirx.txt

python3 -c "import torch, tvm, tvm.tirx; print(torch.__version__, tvm.__version__, torch.cuda.get_device_name())"
```

## Validation order

Run the low-memory shape first. It compiles the kernel, checks the result, and
dumps generated CUDA so `tcgen05`, TMEM, TMA, and the barrier protocol can be
inspected.

```bash
PYTHONPATH=python python3 -m blackwell_moe_tirx.cli \
  --smoke \
  --correctness-only \
  --dump-cuda results/tirx_static_persistent_sm100a.cu
```

Then benchmark one representative shape:

```bash
PYTHONPATH=python python3 -m blackwell_moe_tirx.cli \
  --distribution=zipf \
  --experts=64 \
  --tokens=4096 \
  --n=7168 \
  --k=2048 \
  --warmup=20 \
  --iterations=200 \
  --csv
```

The complete four-distribution run is wrapped by
`tools/run_b200_tirx_baseline.sh`. Each CSV records logical throughput and
useful-work ratio so padded small-M computation is not hidden.

## Next optimization boundary

Once this gate is green, keep the math pipeline fixed and change only work
claiming:

```text
static:  tile_id = cta_id; tile_id += cta_count
dynamic: tile_id = atomicAdd(queue_head, claim_size)
```

That produces the controlled experiment needed later: compare static versus
dynamic scheduling under uniform, heavy-hitter, and Zipf routing while sweeping
only `claim_size` in `{1, 2, 4, 8}`.

## Upstream reference

The warp-specialized pipeline is adapted from Step 7 of the Apache-2.0 licensed
MLC tutorial, [*Modern GPU Programming for
MLSys*](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_advanced/index.html).
The MoE tensor interface, work-list contract, padding policy, benchmark harness,
and subsequent scheduler experiments are owned by this project.
