"""Hardware-independent workload planning for the TIRx MoE kernel.

The host planner turns routed token counts into the one canonical unit of GPU
work used by every scheduler variant: ``(expert_id, tile_m, tile_n)``.
Keeping this module free of TVM, CUDA, and PyTorch makes the planning contract
unit-testable on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal, Sequence


Distribution = Literal["uniform", "heavy_hitter", "sparse", "zipf"]


@dataclass(frozen=True)
class TIRxMoESpec:
    """Compile-time shape and Blackwell pipeline configuration."""

    experts: int = 64
    tokens: int = 4096
    n: int = 7168
    k: int = 2048
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    cta_count: int = 148
    pipe_depth: int = 2
    claim_size: int = 4
    hybrid_main_claim_size: int = 8
    hybrid_tail_tiles: int = 296
    dtype: str = "bfloat16"

    def validate(self) -> None:
        if self.experts <= 0:
            raise ValueError("experts must be positive")
        if self.tokens < 0:
            raise ValueError("tokens must be non-negative")
        if min(self.n, self.k, self.tile_m, self.tile_n, self.tile_k) <= 0:
            raise ValueError("matrix and tile dimensions must be positive")
        if self.n % self.tile_n:
            raise ValueError("n must be divisible by tile_n in the current TIRx family")
        if self.k % self.tile_k:
            raise ValueError("k must be divisible by tile_k in the current TIRx family")
        if (self.tile_m, self.tile_n, self.tile_k) != (128, 128, 64):
            raise ValueError("the current tcgen05 family supports only a 128x128x64 CTA tile")
        if self.cta_count <= 0:
            raise ValueError("cta_count must be positive")
        if self.pipe_depth != 2:
            raise ValueError("the current kernel family requires pipe_depth=2")
        if self.claim_size <= 0:
            raise ValueError("claim_size must be positive")
        if self.hybrid_main_claim_size <= 0:
            raise ValueError("hybrid_main_claim_size must be positive")
        if self.hybrid_tail_tiles < 0:
            raise ValueError("hybrid_tail_tiles must be non-negative")
        if self.dtype != "bfloat16":
            raise ValueError("the current TIRx family supports BF16 inputs and output")


@dataclass(frozen=True)
class WorkTile:
    """One expert output tile; offsets are expressed in matrix elements."""

    expert_id: int
    tile_m: int
    tile_n: int
    valid_m: int
    valid_n: int


@dataclass(frozen=True)
class MoEWorkloadPlan:
    distribution: Distribution
    seed: int
    tokens_per_expert: tuple[int, ...]
    tiles: tuple[WorkTile, ...]
    m_capacity: int
    logical_flops: int
    padded_flops: int

    @property
    def active_experts(self) -> int:
        return sum(count > 0 for count in self.tokens_per_expert)

    @property
    def useful_work_ratio(self) -> float:
        return 1.0 if self.padded_flops == 0 else self.logical_flops / self.padded_flops

    def device_worklist(self) -> list[list[int]]:
        """Return ``(expert, m_tile_index, n_tile_index)`` int32 metadata.

        Storing tile indices instead of element offsets preserves the alignment
        proof needed by TIRx's runtime-coordinate TMA dispatcher.
        """

        # The first kernel contract fixes both M and N tile extents at 128.
        return [
            [tile.expert_id, tile.tile_m // 128, tile.tile_n // 128]
            for tile in self.tiles
        ]


def _weights(
    distribution: Distribution,
    experts: int,
    *,
    zipf_alpha: float,
    heavy_fraction: float,
    heavy_share: float,
    inactive_fraction: float,
) -> list[float]:
    if distribution == "uniform":
        return [1.0] * experts
    if distribution == "zipf":
        return [1.0 / ((rank + 1) ** zipf_alpha) for rank in range(experts)]
    if distribution == "heavy_hitter":
        heavy = max(1, math.ceil(experts * heavy_fraction))
        light = experts - heavy
        heavy_weight = heavy_share / heavy
        light_weight = 0.0 if light == 0 else (1.0 - heavy_share) / light
        return [heavy_weight if expert < heavy else light_weight for expert in range(experts)]
    if distribution == "sparse":
        inactive = min(experts - 1, math.floor(experts * inactive_fraction))
        return [0.0] * inactive + [1.0] * (experts - inactive)
    raise ValueError(f"unsupported distribution: {distribution}")


def generate_token_counts(
    spec: TIRxMoESpec,
    distribution: Distribution,
    *,
    seed: int = 2026,
    zipf_alpha: float = 1.2,
    heavy_fraction: float = 0.1,
    heavy_share: float = 0.8,
    inactive_fraction: float = 0.5,
) -> tuple[int, ...]:
    """Generate the same deterministic routing families as the C++ baseline."""

    spec.validate()
    if zipf_alpha <= 0:
        raise ValueError("zipf_alpha must be positive")
    if not 0.0 < heavy_fraction <= 1.0:
        raise ValueError("heavy_fraction must be in (0, 1]")
    if not 0.0 <= heavy_share <= 1.0:
        raise ValueError("heavy_share must be in [0, 1]")
    if not 0.0 <= inactive_fraction < 1.0:
        raise ValueError("inactive_fraction must be in [0, 1)")

    weights = _weights(
        distribution,
        spec.experts,
        zipf_alpha=zipf_alpha,
        heavy_fraction=heavy_fraction,
        heavy_share=heavy_share,
        inactive_fraction=inactive_fraction,
    )
    rng = random.Random(seed)
    counts = [0] * spec.experts
    for expert in rng.choices(range(spec.experts), weights=weights, k=spec.tokens):
        counts[expert] += 1
    return tuple(counts)


def build_workload_plan(
    spec: TIRxMoESpec,
    distribution: Distribution = "uniform",
    *,
    seed: int = 2026,
    token_counts: Sequence[int] | None = None,
) -> MoEWorkloadPlan:
    """Build an active-expert-compacted, expert-major tile work list."""

    spec.validate()
    counts = (
        generate_token_counts(spec, distribution, seed=seed)
        if token_counts is None
        else tuple(int(count) for count in token_counts)
    )
    if len(counts) != spec.experts:
        raise ValueError("token_counts length must equal experts")
    if any(count < 0 for count in counts):
        raise ValueError("token counts must be non-negative")
    if sum(counts) != spec.tokens:
        raise ValueError("token counts must sum to spec.tokens")

    max_m = max(counts, default=0)
    m_capacity = max(spec.tile_m, math.ceil(max_m / spec.tile_m) * spec.tile_m)
    tiles: list[WorkTile] = []
    for expert_id, expert_m in enumerate(counts):
        for tile_m in range(0, expert_m, spec.tile_m):
            valid_m = min(spec.tile_m, expert_m - tile_m)
            for tile_n in range(0, spec.n, spec.tile_n):
                tiles.append(
                    WorkTile(
                        expert_id=expert_id,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        valid_m=valid_m,
                        valid_n=min(spec.tile_n, spec.n - tile_n),
                    )
                )

    logical_flops = 2 * sum(counts) * spec.n * spec.k
    padded_flops = len(tiles) * 2 * spec.tile_m * spec.tile_n * spec.k
    return MoEWorkloadPlan(
        distribution=distribution,
        seed=seed,
        tokens_per_expert=counts,
        tiles=tuple(tiles),
        m_capacity=m_capacity,
        logical_flops=logical_flops,
        padded_flops=padded_flops,
    )


def static_cta_assignments(tile_count: int, cta_count: int) -> tuple[tuple[int, ...], ...]:
    """Mirror the kernel's grid-stride scheduler for CPU-side verification."""

    if tile_count < 0:
        raise ValueError("tile_count must be non-negative")
    if cta_count <= 0:
        raise ValueError("cta_count must be positive")
    return tuple(
        tuple(range(cta_id, tile_count, cta_count))
        for cta_id in range(cta_count)
    )


def chunked_claims(tile_count: int, claim_size: int) -> tuple[tuple[int, ...], ...]:
    """Return the contiguous batches produced by a monotonic atomic queue."""

    if tile_count < 0:
        raise ValueError("tile_count must be non-negative")
    if claim_size <= 0:
        raise ValueError("claim_size must be positive")
    return tuple(
        tuple(range(begin, min(begin + claim_size, tile_count)))
        for begin in range(0, tile_count, claim_size)
    )


def hybrid_claims(
    tile_count: int,
    main_claim_size: int,
    tail_tiles: int,
) -> tuple[tuple[int, ...], ...]:
    """Model V4: coarse expert-major chunks followed by claim-1 tail work."""

    if tile_count < 0:
        raise ValueError("tile_count must be non-negative")
    if main_claim_size <= 0:
        raise ValueError("main_claim_size must be positive")
    if tail_tiles < 0:
        raise ValueError("tail_tiles must be non-negative")
    tail_begin = tile_count - min(tile_count, tail_tiles)
    main = tuple(
        tuple(range(begin, min(begin + main_claim_size, tail_begin)))
        for begin in range(0, tail_begin, main_claim_size)
    )
    tail = tuple((tile,) for tile in range(tail_begin, tile_count))
    return main + tail
