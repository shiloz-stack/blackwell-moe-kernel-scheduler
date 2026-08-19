"""TIRx implementation of the Blackwell MoE grouped-GEMM prototype."""

from .config import (
    MoEWorkloadPlan,
    TIRxMoESpec,
    WorkTile,
    build_workload_plan,
    chunked_claims,
    hybrid_claims,
    static_cta_assignments,
)
from .dispatch import DispatchThresholds, RoutingFeatures, routing_features, select_kernel

__all__ = [
    "MoEWorkloadPlan",
    "TIRxMoESpec",
    "WorkTile",
    "build_workload_plan",
    "chunked_claims",
    "hybrid_claims",
    "static_cta_assignments",
    "DispatchThresholds",
    "RoutingFeatures",
    "routing_features",
    "select_kernel",
]
