"""V2: warp-specialized persistent GEMM with claim-1 atomic acquisition."""

from .atomic_ws import build_atomic_warp_specialized_kernel


def build_kernel(spec, plan):
    return build_atomic_warp_specialized_kernel(spec, plan, mode="dynamic")
