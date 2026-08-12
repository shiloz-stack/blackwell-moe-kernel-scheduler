# NVIDIA Blackwell architecture selection is intentionally explicit. Override at
# configure time when the CUDA toolkit uses a different architecture identifier:
#   cmake -S . -B build -DBLACKWELL_MOE_ENABLE_CUDA=ON \
#         -DBLACKWELL_MOE_CUDA_ARCHITECTURES=<arch>
set(BLACKWELL_MOE_CUDA_ARCHITECTURES "100a" CACHE STRING
    "CUDA architecture list used for Blackwell kernels")

