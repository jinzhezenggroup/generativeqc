#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "generated_scf_diis_cuda.cuh"
#include "tensor/ring_gram.hpp"

namespace generativeqc::tensor {

/** Refresh a new row/column in a physical-slot Gram cache with one CUDA warp.
 * The caller first inserts the new vector at slot and supplies the live ring
 * window. Other live vectors must be immutable until their slots are replaced.
 * Cache entries remain unnormalized and never alias the destructive solve.
 * The first insertion computes no dots. On the second, initialize_previous
 * establishes the prior diagonal too; subsequent insertions only refresh the
 * new row. A reset therefore needs no cache clear, and a one-iteration warm
 * endpoint does not pay for an unused norm. Retirement narrows the valid window.
 */
__device__ inline void refresh_ring_gram(const double* history, std::size_t vector_size,
                                         std::uint32_t capacity, std::uint32_t slot,
                                         std::uint32_t first, std::uint32_t count,
                                         bool initialize_previous, double* cache,
                                         RingGramWork* work) {
  if (initialize_previous && threadIdx.x == 0) {
    const auto* previous = history + static_cast<std::size_t>(first) * vector_size;
    cache[static_cast<std::size_t>(first) * capacity + first] =
        generated::ordered_history_dot(previous, previous, vector_size);
  }
  const auto* current = history + static_cast<std::size_t>(slot) * vector_size;
  for (std::uint32_t column = threadIdx.x; column < count; column += 32U) {
    const auto other = (first + column) % capacity;
    const auto* previous = history + static_cast<std::size_t>(other) * vector_size;
    const double value = generated::ordered_history_dot(current, previous, vector_size);
    cache[static_cast<std::size_t>(slot) * capacity + other] = value;
    if (other != slot) cache[static_cast<std::size_t>(other) * capacity + slot] = value;
  }
  __syncwarp();
  if (work && threadIdx.x == 0) {
    const auto dots = static_cast<unsigned long long>(count) + initialize_previous;
    work->dots += dots;
    work->vector_elements += dots * vector_size;
  }
}

}  // namespace generativeqc::tensor
