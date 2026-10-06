#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define GENERATIVEQC_WARP_QUEUE_DEVICE __device__
#else
#define GENERATIVEQC_WARP_QUEUE_DEVICE
#endif

namespace generativeqc::scf::cuda_execution {

/** A diagnostic schedule, not an automatically qualified production policy. */
struct BoundedForceWarpSchedule {
  unsigned threads;
  bool dynamic;
};

/** Keep the incumbent for absent/invalid settings and nonstandard launch shapes. */
constexpr BoundedForceWarpSchedule bounded_force_warp_schedule(const char* setting, unsigned x,
                                                               unsigned y, unsigned z) {
  if (setting == nullptr || x != 256U || y != 1U || z != 1U) return {x, false};
  const std::string_view name{setting};
  if (name == "static128") return {128U, false};
  if (name == "warp128") return {128U, true};
  if (name == "warp256") return {256U, true};
  return {x, false};
}

/**
 * Stable, bounded class buckets over an ALREADY admitted packet.
 *
 * Producers write classes[slot] while filling the original task queue. After a
 * producer barrier, one thread prepares the integer index; a second barrier
 * publishes it. Consumers then modify only cursors, using an atomic claim.
 * Task identities, slots and boundaries remain immutable until all consumers have joined
 * the packet-retirement barrier. No raw-domain claim or scientific work lives
 * here. Source identity is uniform for the enclosing force launch.
 */
template <unsigned Capacity, unsigned ClassCount>
struct BoundedClassWarpQueue {
  static_assert(Capacity > 0U && Capacity <= 1024U);
  static_assert(ClassCount > 0U && ClassCount <= 1024U);
  static constexpr std::uint32_t exhausted = Capacity;
  std::uint32_t classes[Capacity];
  std::uint32_t slots[Capacity];
  std::uint32_t boundaries[ClassCount + 1U];
  std::uint32_t cursors[ClassCount];
  bool ready;

  /** Stable counting sort of handles only; no task/scientific-data copy. */
  GENERATIVEQC_WARP_QUEUE_DEVICE constexpr bool prepare(std::uint32_t count) {
    ready = false;
    if (count > Capacity) return false;
    for (unsigned c = 0U; c <= ClassCount; ++c) boundaries[c] = 0U;
    for (std::uint32_t slot = 0U; slot < count; ++slot) {
      if (classes[slot] >= ClassCount) return false;
      ++boundaries[classes[slot] + 1U];
    }
    for (unsigned c = 0U; c < ClassCount; ++c) {
      boundaries[c + 1U] += boundaries[c];
      cursors[c] = boundaries[c];
    }
    for (std::uint32_t slot = 0U; slot < count; ++slot) {
      slots[cursors[classes[slot]]++] = slot;
    }
    for (unsigned c = 0U; c < ClassCount; ++c) cursors[c] = boundaries[c];
    ready = true;
    return true;
  }

  /**
   * Called by one leader per warp. Claim must return the old cursor atomically.
   * Empty buckets need no atomic. Each worker overclaims a nonempty bucket at
   * most once, so packet capacity plus the CTA's warp count bounds each cursor.
   * The caller owns class_cursor; no CTA rendezvous is needed between claims.
   */
  template <typename Claim>
  GENERATIVEQC_WARP_QUEUE_DEVICE constexpr std::uint32_t pop(unsigned& class_cursor, Claim claim) {
    if (!ready) return exhausted;
    while (class_cursor < ClassCount) {
      const auto end = boundaries[class_cursor + 1U];
      if (boundaries[class_cursor] < end) {
        const auto position = claim(cursors + class_cursor);
        if (position < end) return slots[position];
      }
      ++class_cursor;
    }
    return exhausted;
  }
};

/** Disabled kernel instantiations have no class-index arrays. */
struct DisabledClassWarpQueue {};

template <bool Dynamic, unsigned Capacity, unsigned ClassCount>
using BoundedWarpQueueStorage =
    std::conditional_t<Dynamic, BoundedClassWarpQueue<Capacity, ClassCount>,
                       DisabledClassWarpQueue>;

#if defined(__CUDACC__)
struct SharedWarpQueueClaim {
  __device__ __forceinline__ std::uint32_t operator()(std::uint32_t* cursor) const {
    return atomicAdd(cursor, 1U);
  }
};

/** Full-warp cursor; every non-exited lane must call next with the same mask. */
template <bool Dynamic, unsigned Capacity, unsigned ClassCount>
struct BoundedWarpTaskCursor {
  unsigned class_cursor = 0U;
  std::uint32_t static_slot;
  unsigned stride;

  __device__ BoundedWarpTaskCursor(unsigned warp, unsigned warps)
      : static_slot(warp), stride(warps) {}

  __device__ __forceinline__ std::uint32_t next(
      BoundedWarpQueueStorage<Dynamic, Capacity, ClassCount>& storage, unsigned lane,
      std::uint32_t count) {
    if constexpr (Dynamic) {
      if (storage.ready) {
        std::uint32_t slot = Capacity;
        if (lane == 0U) slot = storage.pop(class_cursor, SharedWarpQueueClaim{});
        // This broadcasts ownership, NOT shared-memory publication. The caller
        // must publish the prepared index with a CTA barrier before this loop.
        return __shfl_sync(0xffffffffU, slot, 0);
      }
    }
    // An invalid index never drops tasks: retain the original strided drain.
    const auto slot = static_slot;
    static_slot += stride;
    return slot < count ? slot : count;
  }
};
#endif

}  // namespace generativeqc::scf::cuda_execution

#undef GENERATIVEQC_WARP_QUEUE_DEVICE
