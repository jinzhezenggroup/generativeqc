#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "tensor/ring_gram.hpp"

namespace generativeqc::scf::cuda_execution {

namespace detail {

/** Host preflight shared by the ordered CUDA launcher and host contract tests.
 * Pointers are borrowed device addresses: inspect only nullness/exact aliasing,
 * never their contents. Allocation extents, partial overlap, live-ring identity
 * and stream ordering remain caller preconditions of the public launcher.
 */
inline void validate_ordered_gram_launch(dim3 grid, dim3 block, std::int32_t batch_size,
                                         std::int32_t nbf, std::int32_t matrices_per_system,
                                         std::uint32_t history_capacity, const double* fock,
                                         const double* residual, const std::uint8_t* active,
                                         double* fock_history, double* residual_history,
                                         double* linear_system, double* coefficients,
                                         std::uint32_t* history_count, std::uint32_t* history_head,
                                         double* effective_fock, double* gram_cache) {
  if (batch_size <= 0 || nbf <= 0 || (matrices_per_system != 1 && matrices_per_system != 2) ||
      history_capacity > 64 || !fock || !active || !effective_fock || block.x != 32 ||
      block.y != 1 || block.z != 1 || grid.y != 1 || grid.z != 1 ||
      grid.x < static_cast<unsigned>(batch_size))
    throw std::invalid_argument("invalid ordered Gram shape or launch geometry");
  if (history_capacity >= 2 &&
      (!residual || !fock_history || !residual_history || !linear_system || !coefficients ||
       !history_count || !history_head || !gram_cache || gram_cache == linear_system))
    throw std::invalid_argument("cached Gram requires independent history and solve storage");
  const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(double);
  const auto history_slots = history_capacity >= 2 ? history_capacity : 1U;
  const auto per_system = maximum / static_cast<std::size_t>(batch_size);
  if (static_cast<std::size_t>(nbf) >
          per_system / history_slots / static_cast<std::size_t>(nbf) / matrices_per_system ||
      (history_capacity >= 2 && history_capacity + 1U > per_system / (history_capacity + 1U)))
    throw std::invalid_argument("ordered Gram history size exceeds addressable storage");
}

}  // namespace detail

/** Form deterministic block partials for the history including the current
 * residual, before its circular slot is overwritten. The caller lends
 * batch*history^2*parts doubles; parts=ceil(nbf^2*spins/4096). The kernel
 * computes each symmetric history pair once and mirrors its identical partial
 * into the full-square layout consumed by the update kernel. Only active
 * systems and populated slots are read. No atomics or history mutation occur.
 */
void launch_diis_dot_partials(cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                              std::int32_t spins, std::uint32_t history, const double* residual,
                              const double* residual_history, const std::uint8_t* active,
                              const std::uint32_t* counts, const std::uint32_t* heads,
                              std::size_t parts, double* partials);

/** Update a physical raw Gram cache for the not-yet-stored residual. The cache
 * is disjoint from the augmented-system solve scratch. Only live slots are read. */
cudaError_t launch_diis_pending_gram(cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                     std::int32_t spins, std::uint32_t history,
                                     const double* residual, const double* residual_history,
                                     const std::uint8_t* active, const std::uint32_t* counts,
                                     const std::uint32_t* heads, double* raw_gram);

/** Preserve launch geometry, stream and per-item state routing.
 * cooperative_dots uses one complete 32-lane warp per system, as submitted
 * by compact DF SCF. It computes only unique symmetric Gram entries, mirrors
 * them into the dense solve, and otherwise changes only reduction order while
 * keeping the normalized metric, pivot gate, chronological history retirement
 * and Fock proposal unchanged.
 */
void launch_update_diis_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t nbf, std::int32_t matrices_per_system, std::uint32_t history_capacity,
    const double* fock, const double* residual, const std::uint8_t* active, double* fock_history,
    double* residual_history, double* linear_system, double* coefficients,
    std::uint32_t* history_count, std::uint32_t* history_head, double* effective_fock,
    bool normalize_metric = false, bool cooperative_dots = false,
    const double* dot_partials = nullptr, std::size_t parts = 0, const double* raw_gram = nullptr);

/** Explicit ordered Gram alternative using caller-owned physical-slot storage.
 * HF/KS selects this entry only for incremental admission plus an explicit
 * ordered reducer; unset/cooperative keeps the separate pending-row path.
 * It borrows the same charged raw Gram allocation. Do not switch reduction
 * routes on a live cache: reset count=head=0 before using this ordered owner.
 * Residuals in retained live slots must remain immutable; all calls for a ring
 * must use the same dimensions and stream ordering, with disjoint history,
 * solve and cache allocations. Lend batch*history_capacity^2 doubles. Reset
 * count=head=0 together as on the legacy route; the cache needs no clearing.
 * One 32-lane warp owns each system. Optional work records must be initialized
 * by the caller and measure newly computed dots. The first insertion does no
 * dots; the second also establishes the prior diagonal once after a reset.
 * Normalization, chronological retirement, solve and Fock mixing are unchanged.
 * Disabled/single-vector histories retain the original copy-only behavior and
 * admit a null cache because they neither insert vectors nor compute dots.
 */
void launch_update_diis_cached_gram(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t nbf, std::int32_t matrices_per_system, std::uint32_t history_capacity,
    const double* fock, const double* residual, const std::uint8_t* active, double* fock_history,
    double* residual_history, double* linear_system, double* coefficients,
    std::uint32_t* history_count, std::uint32_t* history_head, double* effective_fock,
    double* gram_cache, bool normalize_metric = false, tensor::RingGramWork* work = nullptr);

}  // namespace generativeqc::scf::cuda_execution
