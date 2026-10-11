#pragma once

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_queue_index.cuh"
#include "scf/direct_block_domain.hpp"
#include "scf/direct_task_layout.hpp"

namespace generativeqc::scf::cuda_execution {

/** One bounded claim borrows the incumbent's geometry-sorted pair inventory.
 * Indexed claims select one fixed page; unindexed claims retain a complete
 * block product. Consumers keep their own CTA-width candidate packets. */
struct BoundedDirectQueuePage {
  std::int32_t system;
  std::size_t first_block, second_block;
  std::size_t first_ordered_begin, second_ordered_begin, second_count;
  std::size_t candidate_begin, candidate_end;
  bool same_block;
};

/** Decode only a block claim. Candidate geometry is intentionally unfinished,
 * so a consumer can reject inactive/screened blocks before loading pair ranges.
 * Complete the page with bounded_direct_queue_candidate_page before use. */
template <class Batch>
__device__ inline bool bounded_direct_queue_block_page(const Batch& batch,
                                                       detail::BoundedDirectBlockDomain domain,
                                                       std::uint64_t ordinal,
                                                       BoundedDirectQueuePage& page) {
  constexpr auto block_candidates =
      detail::kBoundedDirectShellPairBlockSize * detail::kBoundedDirectShellPairBlockSize;
  constexpr auto indexed_candidates =
      block_candidates / detail::kBoundedDirectIndexedCandidatePages;
  static_assert(block_candidates % detail::kBoundedDirectIndexedCandidatePages == 0);
  static_assert(indexed_candidates <= detail::kBoundedDirectQueueCapacity);
  const std::size_t pages = domain.prefix ? detail::kBoundedDirectIndexedCandidatePages : 1U;
  const std::size_t products =
      domain.prefix ? domain.quartet_count : batch.total_shell_pair_block_quartets;
  if (ordinal >= products * pages) return false;
  const auto packed_product = static_cast<std::size_t>(ordinal) / pages;
  if (domain.prefix) {
    page.first_block = bounded_direct_block_row(domain.prefix, domain.row_count, packed_product);
    page.system = static_cast<std::int32_t>(bounded_direct_block_row(
        batch.system_shell_pair_block_offsets, batch.batch_size, page.first_block));
    page.second_block =
        static_cast<std::size_t>(batch.system_shell_pair_block_offsets[page.system]) +
        packed_product - domain.prefix[page.first_block];
  } else {
    page.system = shell_pair_block_quartet_system(batch, packed_product);
    const auto local =
        packed_product -
        static_cast<std::size_t>(batch.system_shell_pair_block_quartet_offsets[page.system]);
    decode_lower_triangle(local, page.first_block, page.second_block);
    const auto first = static_cast<std::size_t>(batch.system_shell_pair_block_offsets[page.system]);
    page.first_block += first;
    page.second_block += first;
  }
  page.same_block = page.first_block == page.second_block;
  page.candidate_begin = domain.prefix ? (ordinal % pages) * indexed_candidates : 0U;
  return true;
}

/** Finish a surviving claim without changing its page or physical ownership. */
template <class Batch>
__device__ inline void bounded_direct_queue_candidate_page(const Batch& batch, bool indexed,
                                                           BoundedDirectQueuePage& page) {
  constexpr auto indexed_candidates = detail::kBoundedDirectShellPairBlockSize *
                                      detail::kBoundedDirectShellPairBlockSize /
                                      detail::kBoundedDirectIndexedCandidatePages;
  const auto system_block_begin =
      static_cast<std::size_t>(batch.system_shell_pair_block_offsets[page.system]);
  const auto system_pair_begin =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[page.system]);
  const auto system_pair_end =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[page.system + 1]);
  page.first_ordered_begin = system_pair_begin + (page.first_block - system_block_begin) *
                                                     detail::kBoundedDirectShellPairBlockSize;
  page.second_ordered_begin = system_pair_begin + (page.second_block - system_block_begin) *
                                                      detail::kBoundedDirectShellPairBlockSize;
  const auto first_count =
      min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - page.first_ordered_begin);
  page.second_count =
      min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - page.second_ordered_begin);
  const auto candidates =
      page.same_block ? first_count * (first_count + 1U) / 2U : first_count * page.second_count;
  page.candidate_end =
      indexed ? min(candidates, page.candidate_begin + indexed_candidates) : candidates;
}

/** One-step adapter for existing experimental class-domain consumers. */
template <class Batch>
__device__ inline bool bounded_direct_queue_page(const Batch& batch,
                                                 detail::BoundedDirectBlockDomain domain,
                                                 std::uint64_t ordinal,
                                                 BoundedDirectQueuePage& page) {
  if (!bounded_direct_queue_block_page(batch, domain, ordinal, page)) return false;
  bounded_direct_queue_candidate_page(batch, domain.prefix != nullptr, page);
  return true;
}

}  // namespace generativeqc::scf::cuda_execution
