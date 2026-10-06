#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "scf/direct_warp_queue.hpp"

namespace generativeqc::scf::cuda_execution {

struct DirectWarpQueueAtomics {
  __device__ __forceinline__ std::uint32_t load(std::uint32_t* address) const {
    return atomicAdd(address, 0U);
  }
  __device__ __forceinline__ std::uint32_t exchange(std::uint32_t* address,
                                                    std::uint32_t value) const {
    return atomicExch(address, value);
  }
  __device__ __forceinline__ std::uint32_t compare_exchange(std::uint32_t* address,
                                                            std::uint32_t expected,
                                                            std::uint32_t desired) const {
    return atomicCAS(address, expected, desired);
  }
};

template <bool Enabled, unsigned Classes, unsigned Capacity>
__device__ __forceinline__ void reset_direct_warp_queue(
    detail::BoundedClassWarpStorage<Enabled, Classes, Capacity>& state) {
  if constexpr (Enabled) {
    for (unsigned bucket = threadIdx.x; bucket < state.bucket_count; bucket += blockDim.x) {
      state.heads[bucket] = state.empty;
    }
  }
  // Caller supplies the existing publication-phase CTA barrier.
}

template <bool Enabled, unsigned Classes, unsigned Capacity>
__device__ __forceinline__ void publish_direct_warp_task(
    detail::BoundedClassWarpStorage<Enabled, Classes, Capacity>& state, unsigned shell_class,
    std::uint32_t slot) {
  if constexpr (Enabled) {
    DirectWarpQueueAtomics atomics;
    state.publish(shell_class, slot, atomics);
  }
}

template <unsigned Classes, unsigned Capacity>
__device__ __forceinline__ std::uint32_t pull_direct_warp_task(
    detail::BoundedClassWarpStorage<true, Classes, Capacity>& state, unsigned& preferred) {
  constexpr unsigned mask = 0xffffffffU;
  // All 32 lanes call this at a warp-uniform boundary, including lanes whose
  // previous AO predicate returned early from the mathematical consumer. Do
  // not use __activemask() inside the divergent recurrence to choose a leader.
  __syncwarp(mask);
  std::uint32_t slot = state.empty;
  if ((threadIdx.x & 31U) == 0U) {
    DirectWarpQueueAtomics atomics;
    slot = state.take_any(preferred, atomics);
  }
  // The shuffle broadcasts an index, not memory publication. Descriptor and
  // next[] visibility comes from the preceding CTA publication barrier; tile
  // mutations keep the original per-tile __syncwarp barriers in the consumer.
  return __shfl_sync(mask, slot, 0);
}

template <bool Enabled, unsigned Classes, unsigned Capacity>
__device__ __forceinline__ std::uint32_t first_direct_warp_task(
    detail::BoundedClassWarpStorage<Enabled, Classes, Capacity>& state, unsigned warp,
    unsigned& preferred) {
  if constexpr (Enabled) {
    return pull_direct_warp_task(state, preferred);
  } else {
    return warp;
  }
}

template <bool Enabled, unsigned Classes, unsigned Capacity>
__device__ __forceinline__ std::uint32_t next_direct_warp_task(
    detail::BoundedClassWarpStorage<Enabled, Classes, Capacity>& state, std::uint32_t slot,
    unsigned warps, unsigned& preferred) {
  if constexpr (Enabled) {
    return pull_direct_warp_task(state, preferred);
  } else {
    return slot + warps;
  }
}

}  // namespace generativeqc::scf::cuda_execution
