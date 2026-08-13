#include "blackwell_moe/scheduler/scheduler_probe.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>

#include "blackwell_moe/scheduler/dynamic_queue.cuh"

namespace blackwell_moe {
namespace {

constexpr std::uint32_t kThreads = 256;

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

// All threads execute deterministic integer work proportional to the tile's
// valid output area.  This is deliberately not a GEMM: it isolates queue and
// load-balancing behavior without conflating it with a particular MMA shape.
__device__ void process_tile(const SchedulerProbeArguments& args,
                             std::uint32_t tile_id,
                             std::uint64_t* reduction,
                             std::uint32_t& local_tiles,
                             std::uint64_t& local_valid_elements) {
  const ExpertTile tile = args.tiles[tile_id];
  const std::uint64_t valid = valid_output_elements(tile);
  const std::uint32_t rounds = max(
      1U,
      args.work_scale * static_cast<std::uint32_t>((valid + 4095) / 4096));

  std::uint64_t value =
      (static_cast<std::uint64_t>(tile.expert_id) << 48) ^
      (static_cast<std::uint64_t>(tile.tile_m) << 32) ^
      (static_cast<std::uint64_t>(tile.tile_n) << 16) ^ threadIdx.x;
  for (std::uint32_t round = 0; round < rounds; ++round) {
    value = mix64(value + round + 0x9e3779b97f4a7c15ULL);
  }
  reduction[threadIdx.x] = value;
  __syncthreads();
  for (std::uint32_t offset = kThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] ^= reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(args.tile_visits + tile_id, 1U);
    args.tile_checksums[tile_id] = reduction[0];
    ++local_tiles;
    local_valid_elements += valid;
  }
  __syncthreads();
}

__global__ void static_persistent_probe(SchedulerProbeArguments args) {
  __shared__ std::uint64_t reduction[kThreads];
  std::uint32_t local_tiles = 0;
  std::uint64_t local_valid_elements = 0;
  for (std::uint32_t tile_id = blockIdx.x; tile_id < args.tile_count;
       tile_id += gridDim.x) {
    process_tile(args, tile_id, reduction, local_tiles, local_valid_elements);
  }
  if (threadIdx.x == 0) {
    args.cta_tile_counts[blockIdx.x] = local_tiles;
    args.cta_valid_elements[blockIdx.x] = local_valid_elements;
  }
}

__global__ void dynamic_queue_probe(SchedulerProbeArguments args) {
  __shared__ std::uint64_t reduction[kThreads];
  __shared__ std::uint32_t claimed_begin;
  __shared__ std::uint32_t claimed_end;
  std::uint32_t local_tiles = 0;
  std::uint64_t local_valid_elements = 0;
  const DynamicQueueState queue{
      args.next_tile, args.tile_count, args.claim_size};

  while (true) {
    if (threadIdx.x == 0) {
      claimed_begin = claim_dynamic_tiles(queue);
      claimed_end = min(claimed_begin + queue.claim_size, queue.tile_count);
    }
    __syncthreads();
    if (claimed_begin >= args.tile_count) break;
    for (std::uint32_t tile_id = claimed_begin; tile_id < claimed_end;
         ++tile_id) {
      process_tile(args, tile_id, reduction, local_tiles,
                   local_valid_elements);
    }
  }
  if (threadIdx.x == 0) {
    args.cta_tile_counts[blockIdx.x] = local_tiles;
    args.cta_valid_elements[blockIdx.x] = local_valid_elements;
  }
}

}  // namespace

KernelStatus launch_scheduler_probe(
    DeviceSchedulerKind scheduler,
    const SchedulerProbeArguments& args) {
  if (args.tiles == nullptr || args.tile_count == 0 || args.cta_count == 0 ||
      args.claim_size == 0 || args.work_scale == 0 ||
      args.tile_visits == nullptr || args.tile_checksums == nullptr ||
      args.cta_tile_counts == nullptr || args.cta_valid_elements == nullptr ||
      (scheduler == DeviceSchedulerKind::kDynamicQueue &&
       args.next_tile == nullptr)) {
    return KernelStatus::kInvalidArgument;
  }
  const auto stream = static_cast<cudaStream_t>(args.stream);
  const dim3 grid(args.cta_count);
  const dim3 block(kThreads);
  switch (scheduler) {
    case DeviceSchedulerKind::kStaticPersistent:
      static_persistent_probe<<<grid, block, 0, stream>>>(args);
      break;
    case DeviceSchedulerKind::kDynamicQueue:
      dynamic_queue_probe<<<grid, block, 0, stream>>>(args);
      break;
    default:
      return KernelStatus::kInvalidArgument;
  }
  return cudaPeekAtLastError() == cudaSuccess ? KernelStatus::kSuccess
                                               : KernelStatus::kCudaError;
}

const char* device_scheduler_name(DeviceSchedulerKind scheduler) {
  switch (scheduler) {
    case DeviceSchedulerKind::kStaticPersistent:
      return "static_persistent_round_robin";
    case DeviceSchedulerKind::kDynamicQueue:
      return "dynamic_global_queue";
  }
  return "unknown";
}

}  // namespace blackwell_moe
