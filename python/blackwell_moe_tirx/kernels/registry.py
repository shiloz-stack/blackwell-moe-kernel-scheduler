"""Lazy registry for the optimization-journey kernels.

This module intentionally does not import TVM.  The CLI and CPU tests can
inspect version metadata on machines without the Blackwell compiler runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from ..config import MoEWorkloadPlan, TIRxMoESpec


@dataclass(frozen=True)
class KernelDescriptor:
    name: str
    module: str
    implementation: str
    persistent: bool
    warp_specialized: bool
    acquisition: str
    requires_queue: bool = False
    requires_clc: bool = False

    def build(self, spec: TIRxMoESpec, plan: MoEWorkloadPlan):
        builder: Callable = getattr(import_module(self.module), "build_kernel")
        return builder(spec, plan)

    def launch_ctas(self, spec: TIRxMoESpec, plan: MoEWorkloadPlan) -> int:
        if self.name in {"v0_nonpersistent", "v5_clc"}:
            return len(plan.tiles)
        return spec.cta_count

    def queue_initial_values(
        self, spec: TIRxMoESpec, plan: MoEWorkloadPlan
    ) -> tuple[int, int]:
        if self.name == "v4_hybrid":
            tail_tiles = min(len(plan.tiles), spec.hybrid_tail_tiles)
            return (0, len(plan.tiles) - tail_tiles)
        return (0, 0)


KERNEL_VERSIONS: dict[str, KernelDescriptor] = {
    "v0_nonpersistent": KernelDescriptor(
        name="v0_nonpersistent",
        module="blackwell_moe_tirx.kernels.v0_nonpersistent",
        implementation="tirx_v0_nonpersistent_double_buffered",
        persistent=False,
        warp_specialized=False,
        acquisition="one CTA per work tile",
    ),
    "v0_5_persistent": KernelDescriptor(
        name="v0_5_persistent",
        module="blackwell_moe_tirx.kernels.v0_5_persistent",
        implementation="tirx_v0_5_static_persistent",
        persistent=True,
        warp_specialized=False,
        acquisition="static grid stride",
    ),
    "v1_static_ws": KernelDescriptor(
        name="v1_static_ws",
        module="blackwell_moe_tirx.kernels.v1_static_ws",
        implementation="tirx_v1_static_persistent_ws",
        persistent=True,
        warp_specialized=True,
        acquisition="static grid stride",
    ),
    "v2_dynamic": KernelDescriptor(
        name="v2_dynamic",
        module="blackwell_moe_tirx.kernels.v2_dynamic",
        implementation="tirx_v2_dynamic_claim1_ws",
        persistent=True,
        warp_specialized=True,
        acquisition="global atomic queue, claim 1",
        requires_queue=True,
    ),
    "v3_chunked": KernelDescriptor(
        name="v3_chunked",
        module="blackwell_moe_tirx.kernels.v3_chunked",
        implementation="tirx_v3_chunked_dynamic_ws",
        persistent=True,
        warp_specialized=True,
        acquisition="global atomic queue, compile-time chunk",
        requires_queue=True,
    ),
    "v4_hybrid": KernelDescriptor(
        name="v4_hybrid",
        module="blackwell_moe_tirx.kernels.v4_hybrid",
        implementation="tirx_v4_locality_hybrid_ws",
        persistent=True,
        warp_specialized=True,
        acquisition="expert-major coarse chunks plus fine-grained tail",
        requires_queue=True,
    ),
    "v5_clc": KernelDescriptor(
        name="v5_clc",
        module="blackwell_moe_tirx.kernels.v5_clc",
        implementation="tirx_v5_clc_ws",
        persistent=True,
        warp_specialized=True,
        acquisition="Blackwell Cluster Launch Control",
        requires_clc=True,
    ),
}


def list_versions() -> tuple[str, ...]:
    return tuple(KERNEL_VERSIONS)


def get_descriptor(version: str) -> KernelDescriptor:
    try:
        return KERNEL_VERSIONS[version]
    except KeyError as error:
        choices = ", ".join(KERNEL_VERSIONS)
        raise ValueError(f"unknown kernel version {version!r}; choose one of: {choices}") from error
