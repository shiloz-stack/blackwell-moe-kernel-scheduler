#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackwell_moe/scheduler/scheduler_probe.cuh"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <class T>
struct DeviceArray {
  T* pointer = nullptr;
  ~DeviceArray() { cudaFree(pointer); }
  void allocate(std::size_t count) {
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer),
                            count * sizeof(T)),
                 "cudaMalloc");
  }
};

void verify_scheduler(blackwell_moe::DeviceSchedulerKind scheduler,
                      const std::vector<blackwell_moe::ExpertTile>& tiles,
                      std::uint32_t cta_count,
                      std::uint32_t claim_size) {
  using namespace blackwell_moe;
  DeviceArray<ExpertTile> device_tiles;
  DeviceArray<std::uint32_t> next_tile;
  DeviceArray<std::uint32_t> visits;
  DeviceArray<std::uint64_t> checksums;
  DeviceArray<std::uint32_t> cta_tiles;
  DeviceArray<std::uint64_t> cta_work;
  device_tiles.allocate(tiles.size());
  next_tile.allocate(1);
  visits.allocate(tiles.size());
  checksums.allocate(tiles.size());
  cta_tiles.allocate(cta_count);
  cta_work.allocate(cta_count);
  require_cuda(cudaMemcpy(device_tiles.pointer, tiles.data(),
                          tiles.size() * sizeof(ExpertTile),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(tiles)");
  require_cuda(cudaMemset(next_tile.pointer, 0, sizeof(std::uint32_t)),
               "cudaMemset(next_tile)");
  require_cuda(cudaMemset(visits.pointer, 0,
                          tiles.size() * sizeof(std::uint32_t)),
               "cudaMemset(visits)");
  require_cuda(cudaMemset(cta_tiles.pointer, 0,
                          cta_count * sizeof(std::uint32_t)),
               "cudaMemset(cta_tiles)");
  require_cuda(cudaMemset(cta_work.pointer, 0,
                          cta_count * sizeof(std::uint64_t)),
               "cudaMemset(cta_work)");

  SchedulerProbeArguments arguments;
  arguments.tiles = device_tiles.pointer;
  arguments.tile_count = static_cast<std::uint32_t>(tiles.size());
  arguments.cta_count = cta_count;
  arguments.claim_size = claim_size;
  arguments.work_scale = 2;
  arguments.next_tile = next_tile.pointer;
  arguments.tile_visits = visits.pointer;
  arguments.tile_checksums = checksums.pointer;
  arguments.cta_tile_counts = cta_tiles.pointer;
  arguments.cta_valid_elements = cta_work.pointer;
  const auto status = launch_scheduler_probe(scheduler, arguments);
  require(status == KernelStatus::kSuccess, kernel_status_string(status));
  require_cuda(cudaDeviceSynchronize(), "scheduler probe execution");

  std::vector<std::uint32_t> host_visits(tiles.size());
  std::vector<std::uint32_t> host_cta_tiles(cta_count);
  std::vector<std::uint64_t> host_cta_work(cta_count);
  require_cuda(cudaMemcpy(host_visits.data(), visits.pointer,
                          host_visits.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(visits)");
  require_cuda(cudaMemcpy(host_cta_tiles.data(), cta_tiles.pointer,
                          host_cta_tiles.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(cta_tiles)");
  require_cuda(cudaMemcpy(host_cta_work.data(), cta_work.pointer,
                          host_cta_work.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(cta_work)");

  for (const auto visit_count : host_visits) {
    require(visit_count == 1, "scheduler did not visit every tile exactly once");
  }
  const auto scheduled_tiles = std::accumulate(
      host_cta_tiles.begin(), host_cta_tiles.end(), std::uint64_t{0});
  require(scheduled_tiles == tiles.size(), "CTA tile counts do not sum");
  std::uint64_t expected_work = 0;
  for (const auto& tile : tiles) expected_work += valid_output_elements(tile);
  const auto scheduled_work = std::accumulate(
      host_cta_work.begin(), host_cta_work.end(), std::uint64_t{0});
  require(scheduled_work == expected_work, "CTA work counters do not sum");
}

}  // namespace

int main() {
  using namespace blackwell_moe;
  try {
    std::vector<ExpertTile> tiles;
    for (std::uint32_t index = 0; index < 37; ++index) {
      tiles.push_back({index % 7,
                       index / 7,
                       index % 4,
                       index % 5 == 0 ? 17U : 128U,
                       index % 3 == 0 ? 65U : 128U});
    }
    verify_scheduler(DeviceSchedulerKind::kStaticPersistent, tiles, 8, 1);
    verify_scheduler(DeviceSchedulerKind::kDynamicQueue, tiles, 8, 1);
    verify_scheduler(DeviceSchedulerKind::kDynamicQueue, tiles, 8, 4);
    std::cout << "GPU scheduler probe correctness passed"
              << " (tiles=" << tiles.size()
              << ", policies=static,dynamic[claim=1,4])\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GPU scheduler probe correctness failed: " << error.what()
              << '\n';
    return 1;
  }
}
