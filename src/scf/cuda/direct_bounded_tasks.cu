#include <cmath>

#include "scf/cuda/direct_bounded_tasks.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_queue_profile.cuh"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/direct_task_encoding.cuh"

namespace generativeqc::scf::cuda_execution {

/**
 * Materialize every enabled exact class in one hierarchical scan.
 *
 * Each class owns a fixed slice whose setup-time weight comes from shell-pair
 * angular histograms. Overflow is recorded per class so a later exact-class
 * page stream can recover only that class without discarding unrelated
 * generated routes or repeating a whole-topology integral evaluation.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose, bool Materialize,
          bool RetainPage = false>
__global__ void compact_bounded_generated_tasks_kernel(
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint8_t* active, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, std::uint64_t excluded_mask, const std::uint32_t* selected_classes,
    const std::uint32_t* selected_any, unsigned long long* global_cursor, GeneratedShellTask* tasks,
    std::uint32_t* task_counts, const std::uint32_t* task_offsets, std::uint32_t* overflow,
    detail::BoundedDirectBlockDomain domain = {}, std::size_t page_begin = 0,
    std::size_t page_end = 0, BoundedForcePage page = {}) {
  __shared__ unsigned long long block_quartet;
  if (selected_any != nullptr && *selected_any == 0U) return;
  const std::size_t total = RetainPage
                                ? page_end - page_begin
                                : static_cast<std::size_t>(batch.total_shell_pair_block_quartets);
  while (true) {
    // Complete every previous claim read, including inactive/screened skips,
    // before the leader publishes another block quartet.
    __syncthreads();
    if (threadIdx.x == 0) block_quartet = atomicAdd(global_cursor, 1ULL);
    __syncthreads();
    if (block_quartet >= total) return;

    const std::size_t packed_block_quartet =
        static_cast<std::size_t>(block_quartet) + (RetainPage ? page_begin : 0U);
    std::int32_t system;
    std::size_t first_block, second_block;
    if (RetainPage && domain.prefix) {
      first_block = bounded_direct_block_row(domain.prefix, domain.row_count, packed_block_quartet);
      system = static_cast<std::int32_t>(bounded_direct_block_row(
          batch.system_shell_pair_block_offsets, batch.batch_size, first_block));
      second_block = static_cast<std::size_t>(batch.system_shell_pair_block_offsets[system]) +
                     packed_block_quartet - domain.prefix[first_block];
    } else {
      system = shell_pair_block_quartet_system(batch, packed_block_quartet);
      const auto local =
          packed_block_quartet -
          static_cast<std::size_t>(batch.system_shell_pair_block_quartet_offsets[system]);
      decode_lower_triangle(local, first_block, second_block);
      const auto begin = static_cast<std::size_t>(batch.system_shell_pair_block_offsets[system]);
      first_block += begin;
      second_block += begin;
    }
    if (active != nullptr && active[system] == 0) continue;
    const std::size_t system_block_begin =
        static_cast<std::size_t>(batch.system_shell_pair_block_offsets[system]);
    const auto first_block_local = first_block - system_block_begin;
    const auto second_block_local = second_block - system_block_begin;
    if (!bounded_direct_block_pair_survives_screening<Purpose>(
            first_block, second_block, system, screening_tolerance, shell_pair_block_bounds,
            system_density_bounds)) {
      continue;
    }

    const std::size_t system_pair_begin =
        static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
    const std::size_t system_pair_end =
        static_cast<std::size_t>(batch.system_shell_pair_offsets[system + 1]);
    const std::size_t first_ordered_begin =
        system_pair_begin + first_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t second_ordered_begin =
        system_pair_begin + second_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t first_count =
        min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - first_ordered_begin);
    const std::size_t second_count =
        min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - second_ordered_begin);
    const bool same_block = first_block == second_block;
    const std::size_t candidate_count =
        same_block ? first_count * (first_count + 1) / 2 : first_count * second_count;
    for (std::size_t candidate = threadIdx.x; candidate < candidate_count;
         candidate += blockDim.x) {
      std::size_t first_local = 0;
      std::size_t second_local = 0;
      if (same_block) {
        decode_lower_triangle(candidate, first_local, second_local);
      } else {
        first_local = candidate / second_count;
        second_local = candidate % second_count;
      }
      const auto first_candidate = shell_pair_order[first_ordered_begin + first_local];
      const auto second_candidate = shell_pair_order[second_ordered_begin + second_local];
      const std::size_t first_pair =
          RetainPage ? max(first_candidate, second_candidate) : first_candidate;
      const std::size_t second_pair =
          RetainPage ? min(first_candidate, second_candidate) : second_candidate;
      if (!direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
              batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds)) {
        continue;
      }
      const std::int32_t first_shell = batch.shell_pair_first[first_pair];
      const std::int32_t second_shell = batch.shell_pair_second[first_pair];
      const std::int32_t third_shell = batch.shell_pair_first[second_pair];
      const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
      const unsigned shell_class = direct_quartet_shell_class_device(
          batch.shell_angular[first_shell], batch.shell_angular[second_shell],
          batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
      if (!bounded_generated_class_enabled(shell_class, enabled_mask_pointer, enabled_mask)) {
        continue;
      }
      if ((excluded_mask & (std::uint64_t{1} << shell_class)) != 0U) {
        continue;
      }
      if (selected_classes != nullptr && selected_classes[shell_class] == 0U) {
        continue;
      }
      if constexpr (RetainPage) {
        // Candidate slots are injective within the page. Retaining the raw
        // pair and its class lets prefix/scatter reuse screening, not repeat it.
        constexpr auto stride =
            detail::kBoundedDirectShellPairBlockSize * detail::kBoundedDirectShellPairBlockSize;
        const auto slot = static_cast<std::size_t>(block_quartet) * stride + candidate;
        page.input[slot] = {static_cast<std::uint32_t>(first_pair),
                            static_cast<std::uint32_t>(second_pair), 0U};
        page.classes[slot] = static_cast<std::uint8_t>(shell_class);
        atomicAdd(page.counts + shell_class, 1U);
        profile_bounded_direct_shell_quartet(batch, page.input[slot], page.profile);
      } else if constexpr (Materialize) {
        const std::uint32_t class_slot = atomicAdd(task_counts + shell_class, 1U);
        const std::uint32_t class_capacity =
            task_offsets[shell_class + 1U] - task_offsets[shell_class];
        if (class_slot >= class_capacity) {
          atomicExch(overflow + shell_class, 1U);
          continue;
        }
        const std::uint32_t slot = task_offsets[shell_class] + class_slot;
        const ActiveShellQuartetTile tile{static_cast<std::uint32_t>(first_pair),
                                          static_cast<std::uint32_t>(second_pair), 0U};
        populate_generated_shell_task(batch, tile, tasks[slot]);
      } else {
        atomicAdd(task_counts + shell_class, 1U);
      }
    }
    __syncthreads();
  }
}

void launch_classify_bounded_force_page(bool unrestricted, unsigned workers, cudaStream_t stream,
                                        DeviceBatch batch, double screening_tolerance,
                                        const double* shell_pair_bounds,
                                        const ShellPairDensityBounds* density_bounds,
                                        const std::uint32_t* pair_order, const double* block_bounds,
                                        const double* system_bounds, const std::uint8_t* active,
                                        detail::BoundedDirectBlockDomain domain, std::size_t begin,
                                        std::size_t end, unsigned long long* cursor,
                                        BoundedForcePage page) {
#define GENERATIVEQC_CLASSIFY_FORCE_PAGE(spin)                                                     \
  compact_bounded_generated_tasks_kernel<spin, DirectScreeningPurpose::Force, true, true>          \
      <<<workers, kBoundedDirectThreads, 0, stream>>>(                                             \
          batch, screening_tolerance, shell_pair_bounds, density_bounds, pair_order, block_bounds, \
          system_bounds, active, nullptr, ~std::uint64_t{0}, 0, nullptr, nullptr, cursor, nullptr, \
          nullptr, nullptr, nullptr, domain, begin, end, page)
  if (unrestricted) {
    GENERATIVEQC_CLASSIFY_FORCE_PAGE(true);
  } else {
    GENERATIVEQC_CLASSIFY_FORCE_PAGE(false);
  }
#undef GENERATIVEQC_CLASSIFY_FORCE_PAGE
}

void launch_compact_bounded_generated_tasks_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint8_t* active,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    std::uint64_t excluded_mask, const std::uint32_t* selected_classes,
    const std::uint32_t* selected_any, unsigned long long* global_cursor, GeneratedShellTask* tasks,
    std::uint32_t* task_counts, const std::uint32_t* task_offsets, std::uint32_t* overflow) {
  if (unrestricted == true) {
    if (purpose == DirectScreeningPurpose::Fock) {
      compact_bounded_generated_tasks_kernel<true, DirectScreeningPurpose::Fock, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds, active,
              enabled_mask_pointer, enabled_mask, excluded_mask, selected_classes, selected_any,
              global_cursor, tasks, task_counts, task_offsets, overflow);
    } else {
      compact_bounded_generated_tasks_kernel<true, DirectScreeningPurpose::Force, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds, active,
              enabled_mask_pointer, enabled_mask, excluded_mask, selected_classes, selected_any,
              global_cursor, tasks, task_counts, task_offsets, overflow);
    }
  } else {
    if (purpose == DirectScreeningPurpose::Fock) {
      compact_bounded_generated_tasks_kernel<false, DirectScreeningPurpose::Fock, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds, active,
              enabled_mask_pointer, enabled_mask, excluded_mask, selected_classes, selected_any,
              global_cursor, tasks, task_counts, task_offsets, overflow);
    } else {
      compact_bounded_generated_tasks_kernel<false, DirectScreeningPurpose::Force, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds, active,
              enabled_mask_pointer, enabled_mask, excluded_mask, selected_classes, selected_any,
              global_cursor, tasks, task_counts, task_offsets, overflow);
    }
  }
}

}  // namespace generativeqc::scf::cuda_execution
