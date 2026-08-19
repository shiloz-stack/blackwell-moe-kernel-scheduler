"""Backward-compatible access to the V1 static warp-specialized kernel.

New code should select a version through :mod:`blackwell_moe_tirx.kernels`.
"""

from .kernels.v1_static_ws import build_kernel

build_static_persistent_moe_kernel = build_kernel

__all__ = ["build_kernel", "build_static_persistent_moe_kernel"]
