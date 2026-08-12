# NVIDIA Blackwell architecture selection is intentionally explicit. The Phase 2
# baseline uses a portable SM80 CUTLASS kernel compiled into an SM100 binary; it
# does not use architecture-conditional tcgen05 instructions yet. Override for
# another Blackwell family (for example SM120) at configure time:
#   cmake -S . -B build -DBLACKWELL_MOE_ENABLE_CUDA=ON \
#         -DBLACKWELL_MOE_CUDA_ARCHITECTURES=<arch>
set(BLACKWELL_MOE_CUDA_ARCHITECTURES "100" CACHE STRING
    "CUDA architecture list used for Blackwell kernels")
