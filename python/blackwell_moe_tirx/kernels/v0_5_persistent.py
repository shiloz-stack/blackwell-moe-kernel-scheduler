"""V0.5: static persistent CTAs without warp specialization."""

from .single_wg import build_single_warpgroup_kernel


def build_kernel(spec, plan):
    return build_single_warpgroup_kernel(spec, plan, persistent=True)
