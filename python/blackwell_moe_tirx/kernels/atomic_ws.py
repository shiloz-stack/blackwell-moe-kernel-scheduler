"""Warp-specialized TIRx MoE GEMM with V2-V4 atomic schedulers."""

from __future__ import annotations

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.backend.cuda.tile_primitive.tma_utils import (
    SwizzleMode,
    mma_shared_layout,
)
from tvm.tirx.lang.pipeline import MBarrier, PipelineState, TCGen05Bar, TMABar
from tvm.tirx.layout import S, TCol, TLane, TileLayout, tid_in_wg

from ..config import MoEWorkloadPlan, TIRxMoESpec
from .atomic_scheduler import AtomicWorkScheduler


BF16_SIZE = 2


def build_atomic_warp_specialized_kernel(
    spec: TIRxMoESpec,
    plan: MoEWorkloadPlan,
    *,
    mode: str,
):
    """Build V2, V3 or V4 while keeping the V1 math pipeline fixed."""

    spec.validate()
    if not plan.tiles:
        raise ValueError("at least one active expert tile is required")
    if mode == "dynamic":
        claim_size = 1
    elif mode == "chunked":
        claim_size = spec.claim_size
    elif mode == "hybrid":
        claim_size = spec.hybrid_main_claim_size
    else:
        raise ValueError(f"unsupported atomic scheduler mode: {mode}")

    E = spec.experts
    M_CAPACITY = plan.m_capacity
    N, K = spec.n, spec.k
    TILE_COUNT = len(plan.tiles)
    BLK_M, BLK_N, BLK_K = spec.tile_m, spec.tile_n, spec.tile_k
    K_TILES = K // BLK_K
    CTA_COUNT = spec.cta_count
    PIPE_DEPTH = spec.pipe_depth
    WG_NUMBER = 2
    HYBRID_TAIL = min(TILE_COUNT, spec.hybrid_tail_tiles)
    HYBRID_MAIN_END = TILE_COUNT - HYBRID_TAIL

    a_type = tvm.DataType(spec.dtype)
    b_type = tvm.DataType(spec.dtype)
    d_type = tvm.DataType(spec.dtype)
    acc_type = tvm.DataType("float32")

    A_layout = mma_shared_layout(
        a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K)
    )
    B_layout = mma_shared_layout(
        b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K)
    )
    D_layout = mma_shared_layout(
        d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N)
    )

    @T.prim_func
    def kernel(
        A: T.Buffer((E, M_CAPACITY, K), a_type),
        B: T.Buffer((E, N, K), b_type),
        WorkTiles: T.Buffer((TILE_COUNT, 3), "int32"),
        QueueHeads: T.Buffer((2,), "int32"),
        D: T.Buffer((E, M_CAPACITY, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([CTA_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld = TCGen05Bar(pool, 1)
        ld2mma = MBarrier(pool, 1)
        work_scheduler = AtomicWorkScheduler(
            pool,
            QueueHeads,
            tile_count=TILE_COUNT,
            mode=mode,
            claim_size=claim_size,
            main_end=HYBRID_MAIN_END,
        )
        pool.move_base_to(1024)
        Asmem = pool.alloc(
            (PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout
        )
        Bsmem = pool.alloc(
            (PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout
        )
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)
        pool.commit()

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(
                    T.address_of(tmem_addr), n_cols=512, cta_group=1
                )
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512),
            acc_type,
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]),
        )

        if wg_id == 1:
            if warp_id == 3:
                # The loader consumes the scheduler's published tile and owns
                # all A/B coordinates for that generation.
                ld = work_scheduler.worker("loader_work")
                ld.reset()
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while ld.valid():
                        ld.consume()
                        if ld.valid():
                            tile_id: T.int32 = ld.tile_id
                            expert_id = T.meta_var(WorkTiles[tile_id, 0])
                            m_st = T.meta_var(WorkTiles[tile_id, 1] * BLK_M)
                            n_st = T.meta_var(WorkTiles[tile_id, 2] * BLK_N)
                            for k_tile in range(K_TILES):
                                mma2tma.wait(tma_ps.stage, tma_ps.phase)
                                k_st = T.meta_var(k_tile * BLK_K)
                                Tx.copy_async(
                                    Asmem[tma_ps.stage, :, :],
                                    A[
                                        expert_id,
                                        m_st : m_st + BLK_M,
                                        k_st : k_st + BLK_K,
                                    ],
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
                                    (
                                        BLK_M * BLK_K
                                        + BLK_N * BLK_K
                                    )
                                    * BF16_SIZE,
                                )
                                tma_ps.advance()

            elif warp_id == 2:
                # Exactly one lane owns QueueHeads.  The other roles never
                # atomicAdd independently, so A/B/MMA/D always refer to the
                # same expert tile.
                work_scheduler.run_scheduler(lane_id)

            elif warp_id == 0:
                mm = work_scheduler.worker("mma_work")
                mm.reset()
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while mm.valid():
                        mm.consume()
                        if mm.valid():
                            ld2mma.wait(ld_ps.stage, ld_ps.phase)
                            ld_ps.advance()
                            for k_tile in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, :BLK_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k_tile != 0),
                                    dispatch="tcgen05",
                                    cta_group=1,
                                )
                                mma2tma.arrive(
                                    mma_ps.stage,
                                    cta_group=1,
                                    cta_mask=0,
                                )
                                mma_ps.advance()
                            mma2ld.arrive(0, cta_group=1, cta_mask=0)

        elif wg_id == 0:
            wb = work_scheduler.worker("writeback_work")
            wb.reset()
            wb_ps = PipelineState(1, phase=0)
            reg_bf16 = T.alloc_local((BLK_N,), d_type)
            while wb.valid():
                wb.consume_wg(wg_id, warp_id, lane_id)
                if wb.valid():
                    tile_id: T.int32 = wb.tile_id
                    expert_id = T.meta_var(WorkTiles[tile_id, 0])
                    m_st = T.meta_var(WorkTiles[tile_id, 1] * BLK_M)
                    n_st = T.meta_var(WorkTiles[tile_id, 2] * BLK_N)

                    mma2ld.wait(wb_ps.stage, wb_ps.phase)
                    wb_ps.advance()
                    T.ptx.tcgen05.fence.after_thread_sync()
                    reg = T.alloc_local((BLK_N,), acc_type)
                    reg_wg = reg.view(
                        128,
                        BLK_N,
                        layout=TileLayout(
                            S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
                        ),
                    )
                    Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
                    T.ptx.tcgen05.wait.ld()
                    ld2mma.arrive(0)

                    Tx.cast(reg_bf16[:], reg[:])
                    Tx.copy(
                        Dsmem[warp_id * 32 + lane_id, :], reg_bf16[:]
                    )
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
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

        T.cuda.cta_sync()
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(
                    tmem_addr[0], n_cols=512, cta_group=1
                )

    return kernel
