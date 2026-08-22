"""Blackwell BF16 MoE grouped GEMM written directly in TIRx.

This first project-owned kernel deliberately keeps scheduling simple: a fixed
set of persistent CTAs walks a host-built expert-tile list with a grid-stride
loop.  Its purpose is to establish a correct math and data-movement baseline
before the dynamic atomic queue becomes an optimization variable.

The pipeline structure follows the Apache-2.0 tutorial implementation in
``mlc-ai/modern-gpu-programming-for-mlsys`` (Step 7, warp specialization),
adapted from dense 2-D tiles to 3-D per-expert tensors and a metadata worklist.
"""

from __future__ import annotations

from typing import Any

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.tirx.lang.pipeline import MBarrier, PipelineState, TCGen05Bar, TMABar
from tvm.tirx.layout import (
    S,
    TCol,
    TLane,
    TileLayout,
    tcgen05_atom_layout,
    tid_in_wg,
    tmem_datapath_layout,
)

from ..config import MoEWorkloadPlan, TIRxMoESpec


BF16_SIZE = 2


def build_static_warp_specialized_kernel(
    spec: TIRxMoESpec,
    plan: MoEWorkloadPlan,
    *,
    small_m: bool = False,
):
    """Return a specialized static-persistent TIRx PrimFunc.

    Tensor shapes and work-list length are compile-time constants.  The three
    int32 columns in ``WorkTiles`` are expert id, M tile index, and N tile index.
    ``A`` is zero padded to ``m_capacity`` so the kernel can use full 128-row
    TMA loads even when an expert's routed-token count is not tile aligned.
    """

    spec.validate()
    expected_tile_m = 64 if small_m else 128
    if spec.tile_m != expected_tile_m or plan.tile_m != expected_tile_m:
        raise ValueError(
            f"this kernel specialization requires tile_m={expected_tile_m}"
        )
    if len(plan.tokens_per_expert) != spec.experts:
        raise ValueError("plan and spec disagree on expert count")
    if not plan.tiles:
        raise ValueError("at least one active expert tile is required")

    # ---------------------------------------------------------------------
    # 0. MoE problem specialization and host-built work list
    # ---------------------------------------------------------------------
    # Unlike a dense GEMM grid, an MoE launch contains a different M for each
    # expert.  The host planner flattens those ragged matrices into TILE_COUNT
    # records of (expert_id, m_tile_index, n_tile_index).  All values below are compile-
    # time constants for one specialized TIRx module; WorkTiles itself remains
    # a runtime GPU buffer.
    E = spec.experts
    M_CAPACITY = plan.m_capacity
    N = spec.n
    K = spec.k
    TILE_COUNT = len(plan.tiles)
    BLK_M, BLK_N, BLK_K = spec.tile_m, spec.tile_n, spec.tile_k
    K_TILES = K // BLK_K
    CTA_COUNT = spec.cta_count
    PIPE_DEPTH = spec.pipe_depth
    WG_NUMBER = 2

    # ---------------------------------------------------------------------
    # 1. CTA tile shape, data types, and shared-memory layouts
    # ---------------------------------------------------------------------
    # The logical operation is D[e] = A[e] @ B[e].T.  B is stored as [N, K],
    # so the transpose is implied by the tcgen05 operand contract rather than
    # materialized as a separate kernel.
    a_type = tvm.DataType(spec.dtype)
    b_type = tvm.DataType(spec.dtype)
    d_type = tvm.DataType(spec.dtype)
    acc_type = tvm.DataType("float32")

    # TMA and tcgen05 must agree on the physical SMEM layout.  The 128-byte
    # swizzle reduces bank conflicts and is part of that cross-engine contract.
    # PIPE_DEPTH is the leading dimension because A/B each have two ring slots.
    A_layout = mma_shared_layout(
        a_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (PIPE_DEPTH, BLK_M, BLK_K),
    )
    B_layout = mma_shared_layout(
        b_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (PIPE_DEPTH, BLK_N, BLK_K),
    )
    D_layout = mma_shared_layout(
        d_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (BLK_M, BLK_N),
    )
    # tcgen05 M=128 uses the direct Layout-D row mapping.  M=64 uses
    # Layout F, which scatters four groups of 16 logical rows across the four
    # 32-lane TMEM regions.  The epilogue fragment must carry the matching
    # .16x256b register layout when it reads the M=64 accumulator back.
    tmem_layout = tmem_datapath_layout(
        "F" if small_m else "D", BLK_M, 512
    )
    small_m_reg_layout = tcgen05_atom_layout(
        "16x256b", (BLK_M, BLK_N), "float32"
    )

    @T.prim_func
    def kernel(
        A: T.Buffer((E, M_CAPACITY, K), a_type),
        B: T.Buffer((E, N, K), b_type),
        WorkTiles: T.Buffer((TILE_COUNT, 3), "int32"),
        D: T.Buffer((E, M_CAPACITY, N), d_type),
    ):
        T.device_entry()

        # -----------------------------------------------------------------
        # 2. Execution scope
        # -----------------------------------------------------------------
        # bx identifies one persistent CTA.  The CTA has two warpgroups:
        #   WG 1: warp 3 issues TMA loads; warp 0 issues tcgen05 MMA.
        #   WG 0: all 128 threads perform the epilogue/writeback.
        # lane_id selects the single elected issuer where a hardware operation
        # must be launched exactly once.
        bx = T.cta_id([CTA_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # -----------------------------------------------------------------
        # 3. Storage and synchronization objects
        # -----------------------------------------------------------------
        # Data follows this memory path:
        #   A/B: HBM --TMA--> double-buffered SMEM
        #   D:   TMEM --tcgen05.ld--> registers --> SMEM --TMA--> HBM
        #
        # The four barriers form two forward readiness edges and two backward
        # buffer-release edges:
        #
        #   TMA --tma2mma--> MMA --mma2ld--> writeback
        #   TMA <--mma2tma-- MMA <--ld2mma-- writeback
        #
        # TMABar is completed by the TMA engine after the expected bytes land.
        # TCGen05Bar is completed by tcgen05 commit.  MBarrier is completed by
        # the 128 writeback threads after they have copied the accumulator out.
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld = TCGen05Bar(pool, 1)
        ld2mma = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc(
            (PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout
        )
        Bsmem = pool.alloc(
            (PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout
        )
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

        # A single elected producer/consumer thread arrives on the first three
        # barriers.  ld2mma expects all 128 threads in writeback warpgroup 0.
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)
        pool.commit()

        # TMEM is not normal shared memory: tcgen05.alloc returns an address at
        # runtime.  We publish that address and the mbarrier initialization to
        # the whole CTA before constructing the typed/layout-aware TMEM view.
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (BLK_M, 512),
            acc_type,
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=tmem_layout,
        )

        # -----------------------------------------------------------------
        # 4-6. TMA producer, Tensor Core consumer, and software pipeline
        # -----------------------------------------------------------------
        # Warpgroup 1 contains both asynchronous issuers.  They execute on
        # different warps and communicate only through the four barriers.
        # Both independently follow bx, bx+CTA_COUNT, ... so they agree on the
        # persistent CTA's current expert tile without sharing a queue counter.
        if wg_id == 1:
            if warp_id == 3:
                # ---------------------------------------------------------
                # 4. TMA producer (WG 1, warp 3, one elected lane)
                # ---------------------------------------------------------
                # phase=1 means the empty ring buffer is immediately reusable.
                # For each K tile:
                #   1) wait until MMA releases the current SMEM stage;
                #   2) launch asynchronous A and B copies into that stage;
                #   3) tell TMABar how many bytes must arrive;
                #   4) advance stage 0 -> 1 -> 0 ...
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)
                tile_id: T.int32 = bx
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_id < TILE_COUNT:
                        # These three values turn the ragged MoE work item into
                        # ordinary 2-D A/B/D slices for this CTA iteration.
                        expert_id = T.meta_var(WorkTiles[tile_id, 0])
                        m_st = T.meta_var(WorkTiles[tile_id, 1] * BLK_M)
                        n_st = T.meta_var(WorkTiles[tile_id, 2] * BLK_N)
                        for k_tile in range(K_TILES):
                            # Backward edge: do not overwrite a stage until the
                            # previous tcgen05 MMA has finished reading it.
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            k_st = T.meta_var(k_tile * BLK_K)
                            Tx.copy_async(
                                Asmem[tma_ps.stage, :, :],
                                A[
                                    expert_id,
                                    m_st : m_st + BLK_M,
                                    k_st : k_st + BLK_K,
                                ],
                                # Runtime expert/tile coordinates use the
                                # explicit TMA mapping path.
                                dispatch="tma_auto",
                                cta_group=1,
                                mbar=tma2mma.ptr_to([tma_ps.stage]),
                            )
                            Tx.copy_async(
                                Bsmem[tma_ps.stage, :, :],
                                B[
                                    expert_id,
                                    n_st : n_st + BLK_N,
                                    k_st : k_st + BLK_K,
                                ],
                                dispatch="tma_auto",
                                cta_group=1,
                                mbar=tma2mma.ptr_to([tma_ps.stage]),
                            )
                            tma2mma.arrive(
                                tma_ps.stage,
                                (BLK_M * BLK_K + BLK_N * BLK_K) * BF16_SIZE,
                            )
                            tma_ps.advance()
                        tile_id = tile_id + CTA_COUNT

            elif warp_id == 0:
                # ---------------------------------------------------------
                # 5. Tensor Core consumer (WG 1, warp 0, one elected lane)
                # ---------------------------------------------------------
                # mma_ps starts at phase=0, so its first tma2mma wait blocks
                # until the producer has filled stage 0.  ld_ps starts ready
                # because no previous output tile occupies TMEM at launch.
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)
                tile_id: T.int32 = bx
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_id < TILE_COUNT:
                        # Backward edge from epilogue: the previous output must
                        # leave TMEM before K=0 overwrites the accumulator.
                        ld2mma.wait(ld_ps.stage, ld_ps.phase)
                        ld_ps.advance()
                        for k_tile in range(K_TILES):
                            # Forward edge from TMA: operands are now complete
                            # and visible in the current shared-memory stage.
                            tma2mma.wait(mma_ps.stage, mma_ps.phase)
                            Tx.gemm_async(
                                tmem[:, :BLK_N],
                                Asmem[mma_ps.stage, :, :],
                                Bsmem[mma_ps.stage, :, :],
                                # K tile 0 initializes FP32 TMEM; later K tiles
                                # accumulate into the same 128x128 result tile.
                                accum=(k_tile != 0),
                                dispatch="tcgen05",
                                cta_group=1,
                            )
                            mma2tma.arrive(
                                mma_ps.stage, cta_group=1, cta_mask=0
                            )
                            mma_ps.advance()
                        # Forward edge to writeback: the full K reduction is
                        # complete and the FP32 accumulator may be consumed.
                        mma2ld.arrive(0, cta_group=1, cta_mask=0)
                        tile_id = tile_id + CTA_COUNT

        # -----------------------------------------------------------------
        # 7. Epilogue / writeback (WG 0, all 128 threads)
        # -----------------------------------------------------------------
        # The warpgroup drains FP32 TMEM into registers, casts to BF16,
        # assembles D in SMEM, and lets one lane issue the final TMA store.
        # M=128 maps one row to each thread.  M=64 instead uses Layout F and a
        # layout-aware register-to-SMEM deposit because its rows are scattered
        # across half of every 32-lane TMEM partition.
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            if not small_m:
                reg_bf16 = T.alloc_local((BLK_N,), d_type)
            tile_id: T.int32 = bx
            while tile_id < TILE_COUNT:
                expert_id = T.meta_var(WorkTiles[tile_id, 0])
                m_st = T.meta_var(WorkTiles[tile_id, 1] * BLK_M)
                n_st = T.meta_var(WorkTiles[tile_id, 2] * BLK_N)

                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                if small_m:
                    reg = T.alloc_buffer(
                        (BLK_M, BLK_N),
                        acc_type,
                        scope="local",
                        layout=small_m_reg_layout,
                    )
                    reg_bf16_small = T.alloc_buffer(
                        (BLK_M, BLK_N),
                        d_type,
                        scope="local",
                        layout=small_m_reg_layout,
                    )
                    Tx.wg.copy_async(reg[:, :], tmem[:, :BLK_N])
                    T.ptx.tcgen05.wait.ld()
                    ld2mma.arrive(0)
                    Tx.wg.cast(reg_bf16_small[:, :], reg[:, :])
                    Tx.wg.copy(Dsmem[:, :], reg_bf16_small[:, :])
                else:
                    reg = T.alloc_local((BLK_N,), acc_type)
                    reg_wg = reg.view(
                        128,
                        BLK_N,
                        layout=TileLayout(
                            S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
                        ),
                    )
                    # TMEM -> per-thread registers. tcgen05.wait.ld is
                    # required before the registers can be read by the cast.
                    Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
                    T.ptx.tcgen05.wait.ld()
                    ld2mma.arrive(0)
                    Tx.cast(reg_bf16[:], reg[:])
                    Tx.copy(
                        Dsmem[warp_id * 32 + lane_id, :], reg_bf16[:]
                    )
                # Publish all register -> SMEM writes before lane 0 launches
                # the shared -> global TMA store.
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    if lane_id == 0:
                        Tx.copy_async(
                            D[
                                expert_id,
                                m_st : m_st + BLK_M,
                                n_st : n_st + BLK_N,
                            ],
                            Dsmem[:, :],
                            dispatch="tma_auto",
                        )
                        # TMA stores do not signal TMABar.  The issuing lane
                        # commits and waits for its bulk-copy group explicitly.
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(10)
                tile_id = tile_id + CTA_COUNT

        # -----------------------------------------------------------------
        # 8. CTA cleanup
        # -----------------------------------------------------------------
        # Every specialized role must finish before the allocating warp returns
        # the CTA's Tensor Memory columns to the hardware pool.
        T.cuda.cta_sync()
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


def build_kernel(spec: TIRxMoESpec, plan: MoEWorkloadPlan):
    """Build the validated M=128 V1 path."""

    return build_static_warp_specialized_kernel(spec, plan, small_m=False)


def compile_static_persistent_moe(
    spec: TIRxMoESpec,
    plan: MoEWorkloadPlan,
    *,
    target: Any = None,
):
    """Lower and compile the specialized kernel through the TIRx pipeline."""

    target_object = tvm.target.Target(
        target or {"kind": "cuda", "arch": "sm_100a"}
    )
    kernel = build_kernel(spec, plan)
    module = tvm.IRModule({"main": kernel})
    with target_object:
        return tvm.compile(module, target=target_object, tir_pipeline="tirx")
