"""V4: coarse expert-major chunks followed by a claim-1 dynamic tail."""

from .atomic_ws import build_atomic_warp_specialized_kernel


def build_kernel(spec, plan):
    return build_atomic_warp_specialized_kernel(spec, plan, mode="hybrid")
