"""V0: one CTA per expert tile, double buffering, no warp specialization."""

from .single_wg import build_single_warpgroup_kernel


def build_kernel(spec, plan):
    return build_single_warpgroup_kernel(spec, plan, persistent=False)
