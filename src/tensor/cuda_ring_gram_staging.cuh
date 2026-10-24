#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "generated_scf_diis_cuda.cuh"
#include "tensor/ring_gram.hpp"

namespace generativeqc::tensor {

/** Stage contiguous operands while one lane continues the generated ordered
 * dot. Each CTA owns one new physical Gram entry; the existing live history is
 * immutable until the following same-stream insertion/update kernel. Shared
 * storage changes the load schedule only: no partial sums are reassociated.
 *
 * A singleton still defers its unused norm. On insertion two the extra block
 * initializes the old diagonal, including after an uncleared count/head reset.
 * Invalid or inactive rings are left for the existing update/reset policy. */
template <unsigned Tile = 256>
__global__ void pending_ordered_ring_gram(std::size_t vector_size, std::uint32_t capacity,
                                          const double* residual, const double* history,
                                          const std::uint8_t* active, const std::uint32_t* counts,
                                          const std::uint32_t* heads, bool normalized,
                                          double* cache, RingGramWork* work) {
  const auto system = blockIdx.x, column = blockIdx.y;
  const auto old_count = counts[system], slot = heads[system];
  if (!active[system] || old_count == 0 || old_count > capacity || slot >= capacity) return;
  const auto count = old_count < capacity ? old_count + 1 : capacity;
  const bool previous_diagonal = column == capacity;
  if (previous_diagonal ? old_count != 1 : column >= count) return;
  const auto first = normalized ? (slot + 1 + capacity - count) % capacity : 0;
  const auto other = previous_diagonal ? first : (first + column) % capacity;
  const auto base = static_cast<std::size_t>(system) * vector_size;
  const auto history_base = base * capacity;
  const auto* current = residual + base;
  const auto* right = other == slot ? current : history + history_base + other * vector_size;
  const auto* left = previous_diagonal ? right : current;
  __shared__ double left_tile[Tile], right_tile[Tile];
  double value = 0.0;
  for (std::size_t begin = 0; begin < vector_size; begin += Tile) {
    const auto remaining = vector_size - begin;
    const auto width = remaining < Tile ? remaining : Tile;
    if (threadIdx.x < width) {
      left_tile[threadIdx.x] = left[begin + threadIdx.x];
      right_tile[threadIdx.x] = right[begin + threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0)
      value = generated::ordered_history_dot(left_tile, right_tile, width, value);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const auto cache_base = static_cast<std::size_t>(system) * capacity * capacity;
    if (previous_diagonal) {
      // An unusual but in-range unnormalized ring can name the new slot as
      // its first entry. Its ordinary diagonal block is the sole writer.
      if (other != slot) cache[cache_base + other * capacity + other] = value;
    } else {
      cache[cache_base + slot * capacity + other] = value;
      if (other != slot) cache[cache_base + other * capacity + slot] = value;
    }
    if (work) {
      atomicAdd(&work[system].dots, 1ULL);
      atomicAdd(&work[system].vector_elements, static_cast<unsigned long long>(vector_size));
    }
  }
}

}  // namespace generativeqc::tensor
