#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define GENERATIVEQC_WARP_QUEUE_INLINE __device__ __forceinline__
#else
#define GENERATIVEQC_WARP_QUEUE_INLINE inline
#endif

namespace generativeqc::scf::detail {

/** Bounded, batch-published class queues; no scientific admission happens here.
 *
 * Publication and consumption are separate phases, separated by a CTA barrier
 * on CUDA (or a host happens-before edge in the host test). During consumption,
 * next[] is immutable, heads only advance, and descriptors are never reinserted.
 * These conditions exclude ABA and make pop-only CAS traversal safe. Reset may
 * begin only after every consumer has finished using its descriptor.
 *
 * AtomicOps supplies load, exchange and compare_exchange on uint32_t words;
 * compare_exchange returns the previous value, like CUDA atomicCAS. The same
 * queue algorithm can therefore be exercised by a genuinely concurrent CPU test.
 * No floating-point operation, screening predicate or force formula lives here.
 */
template <unsigned ClassCount, unsigned Capacity>
struct BoundedClassWarpQueue {
  static_assert(ClassCount > 0 && ClassCount < 0xffffffffU);
  static_assert(Capacity > 0 && Capacity < 0xffffffffU);
  static constexpr std::uint32_t empty = 0xffffffffU;
  // Valid through-f classes remain exact. A defensive final bin keeps an
  // unexpected class from indexing outside metadata; scientific support is
  // still the original consumer's responsibility, not this queue's decision.
  static constexpr unsigned bucket_count = ClassCount + 1U;
  std::uint32_t heads[bucket_count];
  std::uint32_t next[Capacity];

  /** Each unique admitted slot must be < Capacity and published exactly once.
   * The exchanged head is not publication to consumers: the phase barrier is.
   */
  template <class AtomicOps>
  GENERATIVEQC_WARP_QUEUE_INLINE void publish(unsigned shell_class, std::uint32_t slot,
                                              AtomicOps& atomics) {
    const unsigned bucket = shell_class < ClassCount ? shell_class : ClassCount;
    next[slot] = atomics.exchange(heads + bucket, slot);
  }

  template <class AtomicOps>
  GENERATIVEQC_WARP_QUEUE_INLINE std::uint32_t take(unsigned bucket, AtomicOps& atomics) {
    std::uint32_t slot = atomics.load(heads + bucket);
    while (slot != empty) {
      const std::uint32_t observed = atomics.compare_exchange(heads + bucket, slot, next[slot]);
      if (observed == slot) return slot;
      slot = observed;
    }
    return empty;
  }

  /** Stay on the current class while possible, then look for another class.
   * A worker never waits on another worker to finish a task. An empty queue is
   * final for this batch because consumers cannot concurrently publish work.
   * Lists are LIFO by admission linearization, not a stable/FIFO reduction order.
   */
  template <class AtomicOps>
  GENERATIVEQC_WARP_QUEUE_INLINE std::uint32_t take_any(unsigned& preferred, AtomicOps& atomics) {
    unsigned bucket = preferred < bucket_count ? preferred : 0U;
    for (unsigned visited = 0; visited < bucket_count; ++visited) {
      const std::uint32_t slot = take(bucket, atomics);
      if (slot != empty) {
        preferred = bucket;
        return slot;
      }
      if (++bucket == bucket_count) bucket = 0U;
    }
    return empty;
  }
};

// The ordinary specialization has no queue arrays or queue operations. Final
// linked resources must still be compared; source isolation is not proof of
// identical compiler output or module-loading cost.
template <bool Enabled, unsigned ClassCount, unsigned Capacity>
struct BoundedClassWarpStorage {};

template <unsigned ClassCount, unsigned Capacity>
struct BoundedClassWarpStorage<true, ClassCount, Capacity>
    : BoundedClassWarpQueue<ClassCount, Capacity> {};

}  // namespace generativeqc::scf::detail

#undef GENERATIVEQC_WARP_QUEUE_INLINE
