// Standalone real-device scheduler gate. It does NOT qualify ERI/force arithmetic.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "scf/cuda/direct_warp_queue.cuh"

namespace gpu = generativeqc::scf::cuda_execution;
namespace detail = generativeqc::scf::detail;

__global__ void queue_gate(const unsigned* classes, unsigned* visits, unsigned* failures,
                           unsigned count, unsigned rounds) {
  __shared__ detail::BoundedClassWarpStorage<true, 55, 256> queue;
  for (unsigned round = 0; round < rounds; ++round) {
    gpu::reset_direct_warp_queue(queue);
    __syncthreads();
    for (unsigned slot = threadIdx.x; slot < count; slot += blockDim.x) {
      gpu::publish_direct_warp_task(queue, classes[slot], slot);
    }
    __syncthreads();
    unsigned preferred = threadIdx.x / 32U;
    for (;;) {
      const auto slot = gpu::pull_direct_warp_task(queue, preferred);
      if (slot == queue.empty) break;
      if (slot >= count) {
        if ((threadIdx.x & 31U) == 0U) atomicAdd(failures, 1U);
        break;
      }
      // All lanes must receive the same valid slot, including inactive math lanes.
      if (slot != __shfl_sync(0xffffffffU, slot, 0)) atomicAdd(failures, 1U);
      if ((threadIdx.x & 31U) == 0U) {
        const auto cls = classes[slot] < 55U ? classes[slot] : 55U;
        if (preferred != cls) atomicAdd(failures, 1U);
        atomicAdd(visits + slot, 1U);
      }
      __syncwarp();
    }
    // Batch reuse must wait for every consumer; removing this is a data race.
    __syncthreads();
  }
}

bool ok(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
  return false;
}

int main() {
  unsigned *classes = nullptr, *visits = nullptr, *failures = nullptr;
  if (!ok(cudaMallocManaged(&classes, 256 * sizeof(unsigned)), "allocate classes")) return 1;
  if (!ok(cudaMallocManaged(&visits, 256 * sizeof(unsigned)), "allocate visits")) {
    cudaFree(classes);
    return 1;
  }
  if (!ok(cudaMallocManaged(&failures, sizeof(unsigned)), "allocate failures")) {
    cudaFree(visits);
    cudaFree(classes);
    return 1;
  }
  int result = 0;
  unsigned batches = 0;
  for (unsigned block : {64U, 128U, 256U}) {
    for (unsigned count : {0U, 1U, 31U, 32U, 33U, 63U, 64U, 65U, 127U, 128U, 129U, 255U, 256U}) {
      for (unsigned pattern = 0; pattern < 3; ++pattern) {
        std::fill_n(visits, 256, 0U);
        *failures = 0;
        for (unsigned slot = 0; slot < 256; ++slot) {
          classes[slot] = pattern == 0 ? 20U : (17U * slot + pattern) % (pattern == 1 ? 55U : 61U);
        }
        constexpr unsigned rounds = 7;
        // Several independent CTAs share only the diagnostic output counters.
        constexpr unsigned blocks = 9;
        queue_gate<<<blocks, block>>>(classes, visits, failures, count, rounds);
        if (!ok(cudaGetLastError(), "launch") || !ok(cudaDeviceSynchronize(), "synchronize")) {
          result = 1;
          goto cleanup;
        }
        for (unsigned slot = 0; slot < 256; ++slot) {
          if (visits[slot] != (slot < count ? blocks * rounds : 0U)) ++*failures;
        }
        if (*failures != 0U) {
          std::cerr << "Ownership/collective failure at block/count/pattern " << block << '/'
                    << count << '/' << pattern << '\n';
          result = 1;
          goto cleanup;
        }
        ++batches;
      }
    }
  }
  std::cout << batches << " device scheduler configurations passed\n";
cleanup:
  if (!ok(cudaFree(failures), "free failures")) result = 1;
  if (!ok(cudaFree(visits), "free visits")) result = 1;
  if (!ok(cudaFree(classes), "free classes")) result = 1;
  return result;
}
