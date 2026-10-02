// The full provider query is an independent oracle for every published fact.
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "runtime/cuda_device_facts.hpp"

using generativeqc::runtime::CudaTargetInfo;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

int main() {
  try {
    int count = 0, original = -1;
    check(cudaGetDeviceCount(&count));
    check(cudaGetDevice(&original));
    if (count == 0) throw std::runtime_error("no allocated device");
    for (int device = 0; device < count; ++device) {
      cudaDeviceProp full{};
      check(cudaGetDeviceProperties(&full, device));
      for (int repeat = 0; repeat < 2; ++repeat) {
        CudaTargetInfo actual{};
        char name[256]{};
        check(generativeqc::runtime::cuda_device_facts(device, actual, name));
        const std::uint64_t expected_values[] = {std::uint64_t(full.major),
                                                 std::uint64_t(full.minor),
                                                 std::uint64_t(full.warpSize),
                                                 std::uint64_t(full.maxThreadsPerBlock),
                                                 std::uint64_t(full.maxThreadsPerMultiProcessor),
                                                 std::uint64_t(full.maxBlocksPerMultiProcessor),
                                                 std::uint64_t(full.regsPerMultiprocessor),
                                                 full.sharedMemPerBlock,
                                                 full.sharedMemPerBlockOptin,
                                                 full.sharedMemPerMultiprocessor,
                                                 std::uint64_t(full.multiProcessorCount),
                                                 full.totalGlobalMem};
        const std::uint64_t actual_values[] = {std::uint64_t(actual.compute_capability_major),
                                               std::uint64_t(actual.compute_capability_minor),
                                               actual.warp_size,
                                               actual.maximum_threads_per_block,
                                               actual.maximum_threads_per_sm,
                                               actual.maximum_blocks_per_sm,
                                               actual.registers_per_sm,
                                               actual.shared_memory_per_block,
                                               actual.shared_memory_per_block_optin,
                                               actual.shared_memory_per_sm,
                                               actual.multiprocessor_count,
                                               actual.total_global_memory};
        if (std::strcmp(name, full.name) ||
            std::memcmp(expected_values, actual_values, sizeof(expected_values)))
          throw std::runtime_error("device facts differ from complete CUDA property query");
      }
    }
    CudaTargetInfo invalid{};
    invalid.warp_size = 999;
    char name[256];
    std::memset(name, 'x', sizeof(name));
    if (generativeqc::runtime::cuda_device_facts(count, invalid, name) == cudaSuccess ||
        invalid.warp_size != 0 || invalid.total_global_memory != 0)
      throw std::runtime_error("invalid ordinal retained qualified device facts");
    for (char c : name)
      if (c != 0) throw std::runtime_error("invalid ordinal retained partial device name");
    // Clear the intentionally injected invalid-device error before this query.
    cudaGetLastError();
    int current = -1;
    check(cudaGetDevice(&current));
    if (current != original) throw std::runtime_error("metadata query changed current device");
    std::puts("CUDA facts: full-property oracle, invalid ordinal and device preservation passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
