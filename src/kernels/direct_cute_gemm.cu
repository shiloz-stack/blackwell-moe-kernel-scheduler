/***************************************************************************************************
 * Adapted from NVIDIA CUTLASS's SM100 CuTe tutorial 02:
 * examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu (CUTLASS v4.6.0).
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/

#include "blackwell_moe/kernels/direct_cute_gemm.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

#include "cutlass/arch/barrier.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"

// Keep the CuTe include order aligned with the SM100 tutorials.  Several
// algorithm headers consume Tensor and Copy_Atom declarations but do not own
// those declarations themselves.
#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/algorithm/cooperative_copy.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"

namespace blackwell_moe {
namespace {

using namespace cute;

constexpr std::uint32_t kTileM = 128;
constexpr std::uint32_t kTileN = 256;
constexpr std::uint32_t kTileK = 64;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// One shared-memory stage holds the A and B input tiles. Two mbarriers protect
// the stage in opposite directions: TMA completion before MMA, then MMA
// completion before the stage is overwritten by the next K tile.
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct DirectCuteSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> a;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> b;

  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_a() {
    return make_tensor(make_smem_ptr(a.begin()), ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_b() {
    return make_tensor(make_smem_ptr(b.begin()), BSmemLayout{});
  }
};

template <class SharedStorage,
          class ATensor,
          class BTensor,
          class DTensor,
          class MmaTiler,
          class TiledMma,
          class ClusterShape,
          class TmaAtomA,
          class TmaAtomB>
__global__ void direct_cute_gemm_device(
    ATensor matrix_a,
    BTensor matrix_b,
    DTensor matrix_d,
    MmaTiler mma_tiler,
    TiledMma tiled_mma,
    ClusterShape cluster_shape,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_a,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_b) {
  // ---------------------------------------------------------------------------
  // 1. Select this CTA's complete 128x256 output tile.
  // ---------------------------------------------------------------------------
  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape),
      make_tile(typename TiledMma::AtomThrID{}));
  auto mma_coord_vmnk = make_coord(
      blockIdx.x % size<0>(cluster_layout_vmnk),
      blockIdx.x / size<0>(cluster_layout_vmnk),
      blockIdx.y,
      _);
  auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);

  Tensor g_a = local_tile(
      matrix_a, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor g_b = local_tile(
      matrix_b, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor g_d = local_tile(
      matrix_d, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor s_a = storage.tensor_a();
  Tensor s_b = storage.tensor_b();

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor t_c_g_a = cta_mma.partition_A(g_a);
  Tensor t_c_g_b = cta_mma.partition_B(g_b);
  Tensor t_c_g_d = cta_mma.partition_C(g_d);

  // SMEM descriptor fragments are the operands passed to tcgen05.mma.
  Tensor t_c_r_a = cta_mma.make_fragment_A(s_a);
  Tensor t_c_r_b = cta_mma.make_fragment_B(s_b);

  // ---------------------------------------------------------------------------
  // 2. Allocate TMEM for the FP32 accumulator.
  // ---------------------------------------------------------------------------
  // make_fragment_C creates a TMEM-backed tensor layout. Allocator1Sm grants
  // this CTA the full one-SM TMEM column allocation and returns its base address
  // through shared memory so every warp observes the same allocation.
  Tensor t_c_t_acc = cta_mma.make_fragment_C(t_c_g_d);
  const std::uint32_t elected_thread = cute::elect_one_sync();
  const std::uint32_t elected_warp = threadIdx.x / 32 == 0;

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (elected_warp) {
    tmem_allocator.allocate(
        TmemAllocator::Sm100TmemCapacityColumns,
        &storage.tmem_base_ptr);
  }
  __syncthreads();
  t_c_t_acc.data() = storage.tmem_base_ptr;

  // ---------------------------------------------------------------------------
  // 3. Partition TMA transfers and initialize their completion barriers.
  // ---------------------------------------------------------------------------
  auto [t_a_g_a, t_a_s_a] = tma_partition(
      tma_atom_a,
      Int<0>{},
      Layout<_1>{},
      group_modes<0, 3>(s_a),
      group_modes<0, 3>(t_c_g_a));
  auto [t_b_g_b, t_b_s_b] = tma_partition(
      tma_atom_b,
      Int<0>{},
      Layout<_1>{},
      group_modes<0, 3>(s_b),
      group_modes<0, 3>(t_c_g_b));

  const int tma_transaction_bytes =
      sizeof(make_tensor_like(t_a_s_a)) + sizeof(make_tensor_like(t_b_s_b));

  if (elected_warp && elected_thread) {
    cute::initialize_barrier(storage.mma_barrier, 1);
    cute::initialize_barrier(storage.tma_barrier, 1);
  }
  int mma_phase = 0;
  int tma_phase = 0;
  __syncthreads();

  // ---------------------------------------------------------------------------
  // 4. Mainloop: TMA GMEM->SMEM, then tcgen05.mma SMEM->TMEM.
  // ---------------------------------------------------------------------------
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  for (int k_tile = 0; k_tile < size<3>(t_c_g_a); ++k_tile) {
    // One elected thread programs both asynchronous tensor transfers and their
    // expected byte count. All threads wait on the transaction barrier.
    if (elected_warp && elected_thread) {
      cute::set_barrier_transaction_bytes(
          storage.tma_barrier, tma_transaction_bytes);
      copy(tma_atom_a.with(storage.tma_barrier),
           t_a_g_a(_, k_tile),
           t_a_s_a);
      copy(tma_atom_b.with(storage.tma_barrier),
           t_b_g_b(_, k_tile),
           t_b_s_b);
    }
    cute::wait_barrier(storage.tma_barrier, tma_phase);
    tma_phase ^= 1;

    // CuTe lowers each gemm call to the selected SM100 tcgen05.mma atom. The
    // first instruction zeros the TMEM accumulator; subsequent instructions
    // accumulate into the same TMEM fragment.
    if (elected_warp) {
      for (int k_block = 0; k_block < size<2>(t_c_r_a); ++k_block) {
        gemm(tiled_mma,
             t_c_r_a(_, _, k_block),
             t_c_r_b(_, _, k_block),
             t_c_t_acc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&storage.mma_barrier);
    }
    cute::wait_barrier(storage.mma_barrier, mma_phase);
    mma_phase ^= 1;
  }

  // ---------------------------------------------------------------------------
  // 5. Epilogue: tcgen05.ld TMEM->RMEM, then RMEM->GMEM.
  // ---------------------------------------------------------------------------
  // SM100_TMEM_LOAD_32dp32b1x selects the 32-bit TMEM load atom. CuTe's copy
  // lowers this operation to tcgen05.ld and distributes the FP32 accumulator
  // values across the 128 CTA threads before the regular global stores.
  TiledCopy tmem_to_register =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, t_c_t_acc);
  ThrCopy thread_tmem_to_register =
      tmem_to_register.get_slice(threadIdx.x);
  Tensor t_d_t_acc = thread_tmem_to_register.partition_S(t_c_t_acc);
  Tensor t_d_g_d = thread_tmem_to_register.partition_D(t_c_g_d);
  using Accumulator = typename decltype(t_c_t_acc)::value_type;
  Tensor t_d_r_acc = make_tensor<Accumulator>(shape(t_d_g_d));

  copy(tmem_to_register, t_d_t_acc, t_d_r_acc);
  copy(t_d_r_acc, t_d_g_d);
  __syncthreads();

  // TMEM allocations are explicitly lifetime-managed. Releasing the allocation
  // lock first lets another CTA acquire TMEM before this CTA frees its columns.
  if (elected_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(
        storage.tmem_base_ptr,
        TmemAllocator::Sm100TmemCapacityColumns);
  }
}

KernelStatus launch_direct_cute_gemm_impl(
    const DirectCuteGemmArguments& args,
    DirectCuteGemmStats* stats) {
  using TypeA = cutlass::bfloat16_t;
  using TypeB = cutlass::bfloat16_t;
  using TypeD = float;

  const int m = static_cast<int>(args.m);
  const int n = static_cast<int>(args.n);
  const int k = static_cast<int>(args.k);

  auto layout_a = make_layout(
      make_shape(m, k), make_stride(k, Int<1>{}));
  auto layout_b = make_layout(
      make_shape(n, k), make_stride(k, Int<1>{}));
  auto layout_d = make_layout(
      make_shape(m, n), make_stride(n, Int<1>{}));

  Tensor matrix_a = make_tensor(
      make_gmem_ptr(static_cast<TypeA const*>(args.a)), layout_a);
  Tensor matrix_b = make_tensor(
      make_gmem_ptr(static_cast<TypeB const*>(args.b)), layout_b);
  Tensor matrix_d = make_tensor(
      make_gmem_ptr(static_cast<TypeD*>(args.d)), layout_d);

  // This atom is the explicit 1SM BF16xBF16->FP32 tcgen05.mma selection.
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeA,
                           TypeB,
                           TypeD,
                           128,
                           256,
                           UMMA::Major::K,
                           UMMA::Major::K>{});
  auto tile_m = tile_size<0>(tiled_mma);
  auto tile_n = tile_size<1>(tiled_mma);
  auto tile_k = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(tile_m, tile_n, tile_k);

  auto mma_shape_a = partition_shape_A(
      tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_b = partition_shape_B(
      tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  auto smem_layout_a = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_a);
  auto smem_layout_b = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_b);
  using SharedStorage = DirectCuteSharedStorage<
      TypeA,
      TypeB,
      decltype(smem_layout_a),
      decltype(smem_layout_b)>;

  auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape),
      make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  Copy_Atom tma_atom_a = make_tma_atom(
      SM90_TMA_LOAD{}, matrix_a, smem_layout_a, select<0, 2>(mma_tiler));
  Copy_Atom tma_atom_b = make_tma_atom(
      SM90_TMA_LOAD{}, matrix_b, smem_layout_b, select<1, 2>(mma_tiler));
  Tensor matrix_a_tma = tma_atom_a.get_tma_tensor(shape(matrix_a));
  Tensor matrix_b_tma = tma_atom_b.get_tma_tensor(shape(matrix_b));

  dim3 block(128);
  dim3 cluster(
      size<0>(cluster_shape),
      size<1>(cluster_shape),
      size<2>(cluster_shape));
  dim3 grid(
      size(ceil_div(m, tile_m * size<1>(cluster_layout_vmnk))) * cluster.x,
      size(ceil_div(n, tile_n * size<2>(cluster_layout_vmnk))) * cluster.y);
  const int smem_bytes = sizeof(SharedStorage);

  auto* kernel = &direct_cute_gemm_device<
      SharedStorage,
      decltype(matrix_a_tma),
      decltype(matrix_b_tma),
      decltype(matrix_d),
      decltype(mma_tiler),
      decltype(tiled_mma),
      decltype(cluster_shape),
      decltype(tma_atom_a),
      decltype(tma_atom_b)>;

  if (cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_bytes) != cudaSuccess) {
    return KernelStatus::kCudaError;
  }

  cutlass::ClusterLaunchParams launch_params{
      grid,
      block,
      cluster,
      smem_bytes,
      static_cast<cudaStream_t>(args.stream)};
  const auto status = cutlass::launch_kernel_on_cluster(
      launch_params,
      (void const*)kernel,
      matrix_a_tma,
      matrix_b_tma,
      matrix_d,
      mma_tiler,
      tiled_mma,
      cluster_shape,
      tma_atom_a,
      tma_atom_b);
  if (status != cutlass::Status::kSuccess) {
    return KernelStatus::kCutlassError;
  }

  if (stats != nullptr) {
    stats->grid_m = grid.x;
    stats->grid_n = grid.y;
    stats->dynamic_smem_bytes = smem_bytes;
  }
  return cudaPeekAtLastError() == cudaSuccess ? KernelStatus::kSuccess
                                               : KernelStatus::kCudaError;
}

#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

}  // namespace

KernelStatus launch_direct_cute_gemm(
    const DirectCuteGemmArguments& args,
    DirectCuteGemmStats* stats) {
  if (args.a == nullptr || args.b == nullptr || args.d == nullptr ||
      args.m == 0 || args.n == 0 || args.k == 0 ||
      args.m % kTileM != 0 || args.n % kTileN != 0 ||
      args.k % kTileK != 0 ||
      args.m > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      args.n > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      args.k > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return KernelStatus::kInvalidArgument;
  }

  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return KernelStatus::kCudaError;
  }
  if (properties.major != 10 || properties.minor != 0) {
    return KernelStatus::kUnsupportedDevice;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return launch_direct_cute_gemm_impl(args, stats);
#else
  (void)stats;
  return KernelStatus::kUnsupportedDevice;
#endif
}

const char* direct_cute_kernel_name() {
  return "direct_cute_sm100_tma_tcgen05_tmem_bf16_f32_128x256x64";
}

}  // namespace blackwell_moe
