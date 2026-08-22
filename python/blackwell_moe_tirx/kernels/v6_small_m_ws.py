"""V6 microkernel: static persistent M=64 tcgen05 Layout-F path."""

from .v1_static_ws import build_static_warp_specialized_kernel


def build_kernel(spec, plan):
    """Build the M=64 tail kernel while retaining V1's TMA/MMA pipeline."""

    return build_static_warp_specialized_kernel(spec, plan, small_m=True)
