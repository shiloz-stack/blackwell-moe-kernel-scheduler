# NVIDIA Blackwell architecture selection is intentionally explicit. The Phase 2
# baseline uses a portable SM80 CUTLASS kernel compiled into the selected binary.
# SM100a-native targets use a separate explicit compute_100a flag in the root
# CMake file because tcgen05/TMEM are architecture-accelerated features. Override
# the portable target for another Blackwell family (for example SM120) with:
#   cmake -S . -B build -DBLACKWELL_MOE_ENABLE_CUDA=ON \
#         -DBLACKWELL_MOE_CUDA_ARCHITECTURES=<arch>
set(BLACKWELL_MOE_CUDA_ARCHITECTURES "100" CACHE STRING
    "CUDA architecture list used for Blackwell kernels")
