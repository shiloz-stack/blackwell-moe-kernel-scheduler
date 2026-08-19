"""Host-launch workload features and an explicit bootstrap dispatch policy.

The policy here is intentionally not presented as a performance result.  It
provides a stable launch-level interface whose thresholds can be fitted after
the B200 version sweep has produced evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import MoEWorkloadPlan, TIRxMoESpec


@dataclass(frozen=True)
class RoutingFeatures:
    mean_m: float
    cv_m: float
    max_over_mean_m: float
    inactive_expert_ratio: float
    small_m_expert_ratio: float
    expert_tile_cv: float
    max_expert_tiles_over_mean: float
    active_tiles: int
    tiles_per_cta: float
    useful_work_ratio: float


@dataclass(frozen=True)
class DispatchThresholds:
    """Replace these bootstrap values with thresholds fitted on B200 CSVs."""

    nonpersistent_tiles_per_cta: float = 1.0
    static_cv_m: float = 0.25
    dynamic_min_tiles_per_cta: float = 2.0
    chunked_min_tiles_per_cta: float = 4.0
    high_skew_cv_m: float = 1.0
    sparse_expert_ratio: float = 0.5


def routing_features(spec: TIRxMoESpec, plan: MoEWorkloadPlan) -> RoutingFeatures:
    """Summarize the routed expert rows before a kernel is launched."""

    counts = plan.tokens_per_expert
    mean_m = (sum(counts) / len(counts)) if counts else 0.0
    if mean_m:
        variance = sum((count - mean_m) ** 2 for count in counts) / len(counts)
        cv_m = math.sqrt(variance) / mean_m
        max_over_mean = max(counts) / mean_m
    else:
        cv_m = 0.0
        max_over_mean = 0.0

    active = [count for count in counts if count > 0]
    inactive_ratio = 1.0 - (len(active) / len(counts)) if counts else 0.0
    small_ratio = (
        sum(count <= spec.tile_m for count in active) / len(active) if active else 0.0
    )
    n_tiles = spec.n // spec.tile_n
    expert_tiles = [math.ceil(count / spec.tile_m) * n_tiles for count in counts]
    mean_tiles = sum(expert_tiles) / len(expert_tiles) if expert_tiles else 0.0
    if mean_tiles:
        tile_variance = sum(
            (tile_count - mean_tiles) ** 2 for tile_count in expert_tiles
        ) / len(expert_tiles)
        tile_cv = math.sqrt(tile_variance) / mean_tiles
        max_tiles_over_mean = max(expert_tiles) / mean_tiles
    else:
        tile_cv = 0.0
        max_tiles_over_mean = 0.0
    return RoutingFeatures(
        mean_m=mean_m,
        cv_m=cv_m,
        max_over_mean_m=max_over_mean,
        inactive_expert_ratio=inactive_ratio,
        small_m_expert_ratio=small_ratio,
        expert_tile_cv=tile_cv,
        max_expert_tiles_over_mean=max_tiles_over_mean,
        active_tiles=len(plan.tiles),
        tiles_per_cta=len(plan.tiles) / spec.cta_count,
        useful_work_ratio=plan.useful_work_ratio,
    )


def select_kernel(
    spec: TIRxMoESpec,
    plan: MoEWorkloadPlan,
    thresholds: DispatchThresholds = DispatchThresholds(),
) -> str:
    """Select a candidate kernel at launch level using auditable rules.

    V5 CLC is excluded until measurements establish a crossover.  Sparse
    experts are already compacted out of every version's device work list.
    """

    features = routing_features(spec, plan)
    if features.tiles_per_cta <= thresholds.nonpersistent_tiles_per_cta:
        return "v0_nonpersistent"
    if features.cv_m <= thresholds.static_cv_m:
        return "v1_static_ws"
    if features.tiles_per_cta < thresholds.dynamic_min_tiles_per_cta:
        return "v1_static_ws"
    if (
        features.tiles_per_cta >= thresholds.chunked_min_tiles_per_cta
        and (
            features.cv_m >= thresholds.high_skew_cv_m
            or features.inactive_expert_ratio >= thresholds.sparse_expert_ratio
        )
    ):
        return "v4_hybrid"
    if features.tiles_per_cta >= thresholds.chunked_min_tiles_per_cta:
        return "v3_chunked"
    return "v2_dynamic"
