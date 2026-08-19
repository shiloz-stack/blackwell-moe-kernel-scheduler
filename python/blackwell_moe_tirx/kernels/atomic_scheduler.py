"""CTA-local publication protocol for V2-V4 atomic work acquisition."""

from __future__ import annotations

from tvm.script import tirx as T
from tvm.tirx.lang.pipeline import Pipeline, PipelineState


@T.meta_class
class _AtomicWorker:
    """Per-role view of one scheduler-published work item."""

    def __init__(self, scheduler, prefix):
        self._scheduler = scheduler
        self._ready = PipelineState(1, 0)
        self.tile_id = T.local_scalar("int32")
        self._done = T.local_scalar("int32")

    @T.inline
    def reset(self):
        self._done = 0

    def valid(self):
        return self._done == 0

    @T.inline
    def consume(self):
        """Consume from one elected thread and release one mailbox arrival."""

        self._scheduler.work_pipe.full.wait(0, self._ready.phase)
        self._ready.advance()
        self.tile_id = self._scheduler.mailbox[0]
        self._scheduler.work_pipe.empty.arrive(0)
        if self.tile_id < 0:
            self._done = 1

    @T.inline
    def consume_wg(self, wg_id, warp_id, lane_id):
        """Let all 128 writeback threads read before one lane releases the slot."""

        self._scheduler.work_pipe.full.wait(0, self._ready.phase)
        self._ready.advance()
        self.tile_id = self._scheduler.mailbox[0]
        T.cuda.warpgroup_sync(wg_id + 12)
        if (warp_id == 0) & (lane_id == 0):
            self._scheduler.work_pipe.empty.arrive(0)
        if self.tile_id < 0:
            self._done = 1


@T.meta_class
class AtomicWorkScheduler:
    """One atomic claimant publishes work to loader, MMA and writeback roles.

    `mode` is a parse-time string:

    * ``dynamic``: claim one item at a time;
    * ``chunked``: claim contiguous expert-major batches;
    * ``hybrid``: claim coarse batches from the main region, then claim one
      item at a time from a separately initialized tail queue.
    """

    def __init__(
        self,
        pool,
        queue_heads,
        *,
        tile_count: int,
        mode: str,
        claim_size: int,
        main_end: int = 0,
    ):
        if mode not in {"dynamic", "chunked", "hybrid"}:
            raise ValueError(f"unsupported atomic scheduler mode: {mode}")
        self.queue_heads = queue_heads
        self.tile_count = tile_count
        self.mode = mode
        self.claim_size = claim_size
        self.main_end = main_end
        # Three consumers release each generation: loader, MMA and writeback.
        self.work_pipe = Pipeline(
            pool, 1, full="mbar", empty="mbar", init_empty=3
        )
        self.mailbox = pool.alloc((1,), "int32", align=4)

    def worker(self, prefix):
        return _AtomicWorker(self, prefix)

    @T.inline
    def run_scheduler(self, lane_id):
        """Run on one elected lane in the otherwise idle scheduler warp."""

        if T.filter(lane_id, T.ptx.elect_sync()):
            empty = PipelineState(1, 1)
            chunk_next = T.local_scalar("int32")
            chunk_end = T.local_scalar("int32")
            begin = T.local_scalar("int32")
            next_tile = T.local_scalar("int32")
            using_tail = T.local_scalar("int32")
            done = T.local_scalar("int32")
            chunk_next = 0
            chunk_end = 0
            using_tail = 0
            done = 0

            while done == 0:
                self.work_pipe.empty.wait(0, empty.phase)
                empty.advance()

                if chunk_next >= chunk_end:
                    if self.mode == "hybrid":
                        if using_tail == 0:
                            begin = T.cuda.atomic_add(
                                T.address_of(self.queue_heads[0]),
                                self.claim_size,
                            )
                            if begin < self.main_end:
                                chunk_next = begin
                                chunk_end = T.min(
                                    begin + self.claim_size, self.main_end
                                )
                            else:
                                using_tail = 1
                                begin = T.cuda.atomic_add(
                                    T.address_of(self.queue_heads[1]), 1
                                )
                                chunk_next = begin
                                chunk_end = T.min(begin + 1, self.tile_count)
                        else:
                            begin = T.cuda.atomic_add(
                                T.address_of(self.queue_heads[1]), 1
                            )
                            chunk_next = begin
                            chunk_end = T.min(begin + 1, self.tile_count)
                    else:
                        begin = T.cuda.atomic_add(
                            T.address_of(self.queue_heads[0]), self.claim_size
                        )
                        chunk_next = begin
                        chunk_end = T.min(
                            begin + self.claim_size, self.tile_count
                        )

                if chunk_next < self.tile_count:
                    next_tile = chunk_next
                    chunk_next = chunk_next + 1
                else:
                    next_tile = -1
                    done = 1

                self.mailbox[0] = next_tile
                # mbarrier publication is the control edge.  The fence keeps
                # the generic shared-memory mailbox write ordered before the
                # three role-local waits observe the new generation.
                T.cuda.thread_fence()
                self.work_pipe.full.arrive(0)
