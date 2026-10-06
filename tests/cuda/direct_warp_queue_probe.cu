// Real-device ownership/synchronization probe. This does NOT validate chemistry.
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <vector>

#include "scf/cuda/direct_warp_queue.cuh"

using namespace generativeqc::scf::cuda_execution;
constexpr unsigned capacity = 256U;
constexpr unsigned classes = 55U;
constexpr unsigned blocks = 3U;
constexpr unsigned epochs = 3U;

void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct DeviceWords {
  unsigned* pointer = nullptr;
  explicit DeviceWords(unsigned count) { checked(cudaMalloc(&pointer, count * sizeof(unsigned))); }
  ~DeviceWords() { cudaFree(pointer); }
  DeviceWords(const DeviceWords&) = delete;
  DeviceWords& operator=(const DeviceWords&) = delete;
};

template <bool Dynamic>
__global__ void ownership_probe(unsigned count, unsigned mode, unsigned* visits,
                                unsigned* errors) {
  __shared__ BoundedWarpQueueStorage<Dynamic, capacity, classes> queue;
  __shared__ unsigned tile_state[capacity];
  const unsigned lane = threadIdx.x % 32U;
  const unsigned warp = threadIdx.x / 32U;
  for (unsigned epoch = 0U; epoch < epochs; ++epoch) {
    if constexpr (Dynamic) {
      for (unsigned slot = threadIdx.x; slot < count; slot += blockDim.x) {
        queue.classes[slot] = mode == 0U ? classes - 1U : (slot * 17U + epoch * 11U) % classes;
        if (mode == 2U && slot == 0U) queue.classes[slot] = classes;
      }
    }
    __syncthreads();
    if constexpr (Dynamic) {
      if (threadIdx.x == 0U) queue.prepare(count);
    }
    __syncthreads();
    BoundedWarpTaskCursor<Dynamic, capacity, classes> cursor(warp, blockDim.x / 32U);
    for (auto slot = cursor.next(queue, lane, count); slot < count;
         slot = cursor.next(queue, lane, count)) {
      const auto offset = (epoch * blocks + blockIdx.x) * capacity + slot;
      if (lane == 0U) atomicAdd(visits + offset, 1U);
      // Different task lengths and mutable task state exercise independent warp
      // progress. Every lane must see its OWN task's tile at every iteration.
      for (unsigned tile = 0U; tile <= (slot * 13U + epoch) % 11U; ++tile) {
        if (lane == 0U) tile_state[slot] = epoch * 1000U + tile;
        __syncwarp();
        if (tile_state[slot] != epoch * 1000U + tile) atomicAdd(errors, 1U);
        __syncwarp();
      }
    }
    // Packet storage cannot be recycled while any warp still owns a task.
    __syncthreads();
  }
}

int main() {
  try {
    int devices = 0;
    checked(cudaGetDeviceCount(&devices));
    if (devices == 0) throw std::runtime_error("real CUDA device required");
    constexpr unsigned words = capacity * blocks * epochs;
    DeviceWords visits(words), errors(1U);
    std::vector<unsigned> actual(words);
    unsigned launches = 0;
    for (unsigned threads : {128U, 256U}) {
      for (bool dynamic : {false, true}) {
        for (unsigned mode : {0U, 1U, 2U}) {
          for (unsigned count : {0U, 1U, 7U, 8U, 31U, 32U, 33U, 63U, 64U, 65U,
                                 127U, 128U, 129U, 255U, 256U}) {
            checked(cudaMemset(visits.pointer, 0, words * sizeof(unsigned)));
            checked(cudaMemset(errors.pointer, 0, sizeof(unsigned)));
            if (dynamic) {
              ownership_probe<true><<<blocks, threads>>>(count, mode, visits.pointer, errors.pointer);
            } else {
              ownership_probe<false><<<blocks, threads>>>(count, mode, visits.pointer, errors.pointer);
            }
            checked(cudaGetLastError());
            checked(cudaDeviceSynchronize());
            checked(cudaMemcpy(actual.data(), visits.pointer, words * sizeof(unsigned),
                               cudaMemcpyDeviceToHost));
            unsigned failures = 0;
            checked(cudaMemcpy(&failures, errors.pointer, sizeof(unsigned), cudaMemcpyDeviceToHost));
            if (failures != 0U) throw std::runtime_error("cross-warp task ownership failure");
            for (unsigned index = 0U; index < words; ++index) {
              if (actual[index] != (index % capacity < count ? 1U : 0U)) {
                throw std::runtime_error("missing, duplicated or out-of-range task");
              }
            }
            ++launches;
          }
        }
      }
    }
    std::printf("PASS: %u actual CUDA launches, three packet lifetimes each\n", launches);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
