#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "generated_derivative_cuda_shell_aot.cuh"
#include "scf/cuda/direct_angular_force.hpp"
#include "scf/cuda/direct_bounded_contraction.cuh"
#include "scf/cuda/direct_bounded_fallback.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_fock_order2.cuh"
#include "scf/cuda/direct_fock_quartet.cuh"
#include "scf/cuda/direct_force_execution.cuh"
#include "scf/cuda/direct_force_order4_sources.cuh"
#include "scf/cuda/direct_force_order5_sources.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/direct_queue_profile.cuh"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/direct_task_encoding.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

namespace {
/** Qualification needs a resident cache and the retained derivative algebra.
 * Optional reachable/convolution schedules keep their existing force owner. */
__host__ __device__ bool materialized_pair_derivative_available(const DeviceBatch& batch) {
  return batch.direct_pair_materialized_derivatives && batch.shell_primitive_pairs &&
         batch.shell_pair_primitive_offsets && (batch.direct_coulomb_reachable & 2U) == 0U &&
         (batch.direct_hermite_convolution & 2U) == 0U;
}

__device__ bool materialized_pair_derivative_task(const DeviceBatch& batch,
                                                  const ActiveShellQuartetTile& task) {
  return batch.shell_angular[batch.shell_pair_first[task.first_pair]] == 2U &&
         batch.shell_angular[batch.shell_pair_second[task.first_pair]] == 2U &&
         batch.shell_angular[batch.shell_pair_first[task.second_pair]] == 2U &&
         batch.shell_angular[batch.shell_pair_second[task.second_pair]] == 2U;
}
}  // namespace

/**
 * Enumerate, screen, queue, and drain shell pair-of-pairs hierarchically.
 *
 * A persistent CTA first claims one pair-block product. Conservative Schwarz
 * and density maxima reject the complete block without visiting its members;
 * surviving blocks are expanded in fixed 256-candidate chunks and retain the
 * exact shell-quartet predicate. This keeps storage bounded while replacing
 * the former unconditional O(N_shell^4) scan with a small O(N_shell^4/B^2)
 * outer domain plus exact work only in surviving blocks.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose, bool Force, int FixedAngularOrder = -1,
          int FixedRadialOperator = -1, bool PairDerivatives = false>
__global__ __launch_bounds__(kBoundedDirectThreads, 1) void bounded_direct_shell_quartet_kernel(
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile,
    double coulomb_coefficient, double exchange_coefficient,
    DirectRangeOperator runtime_radial_operator, double omega,
    double secondary_exchange_coefficient, bool coulomb_only, bool exchange_only,
    detail::BoundedDirectBlockDomain block_domain = {}) {
  static_assert(FixedAngularOrder < 0 || (Force && FixedAngularOrder <= 12));
  static_assert(!PairDerivatives || (Force && (FixedAngularOrder < 0 || FixedAngularOrder == 8)));
  extern __shared__ __align__(16) unsigned char materialized_pair_workspace[];
  const auto radial_operator = FixedRadialOperator < 0
                                   ? runtime_radial_operator
                                   : static_cast<DirectRangeOperator>(FixedRadialOperator);
  __shared__ ActiveShellQuartetTile queue[detail::kBoundedDirectQueueCapacity];
  __shared__ std::uint32_t queue_count;
  __shared__ unsigned long long block_quartet;
  const unsigned lane = threadIdx.x % detail::kDirectQuartetThreads;
  const unsigned warp = threadIdx.x / detail::kDirectQuartetThreads;
  constexpr auto full_block_candidates =
      detail::kBoundedDirectShellPairBlockSize * detail::kBoundedDirectShellPairBlockSize;
  constexpr auto indexed_page_candidates =
      full_block_candidates / detail::kBoundedDirectIndexedCandidatePages;
  static_assert(full_block_candidates % detail::kBoundedDirectIndexedCandidatePages == 0);
  static_assert(indexed_page_candidates <= detail::kBoundedDirectQueueCapacity);
  const std::size_t pages = block_domain.prefix ? detail::kBoundedDirectIndexedCandidatePages : 1U;
  const std::size_t products =
      block_domain.prefix ? block_domain.quartet_count
                          : static_cast<std::size_t>(batch.total_shell_pair_block_quartets);
  const std::size_t total = products * pages;

  while (true) {
    // Empty pages and screened/inactive claims bypass the candidate-loop barrier.
    // All warps must finish reading this claim before the leader overwrites it.
    __syncthreads();
    if (threadIdx.x == 0) {
      block_quartet = atomicAdd(global_cursor, 1ULL);
    }
    __syncthreads();
    if (block_quartet >= total) return;

    const std::size_t packed_block_quartet = static_cast<std::size_t>(block_quartet) / pages;
    std::int32_t system = 0;
    std::size_t first_block = 0, second_block = 0;
    if (block_domain.prefix) {
      first_block = bounded_direct_block_row(block_domain.prefix, block_domain.row_count,
                                             packed_block_quartet);
      system = static_cast<std::int32_t>(bounded_direct_block_row(
          batch.system_shell_pair_block_offsets, batch.batch_size, first_block));
      second_block = static_cast<std::size_t>(batch.system_shell_pair_block_offsets[system]) +
                     packed_block_quartet - block_domain.prefix[first_block];
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

    const std::size_t page_begin =
        block_domain.prefix ? (block_quartet % pages) * indexed_page_candidates : 0U;
    const std::size_t page_end = block_domain.prefix
                                     ? min(candidate_count, page_begin + indexed_page_candidates)
                                     : candidate_count;
    for (std::size_t candidate_begin = page_begin; candidate_begin < page_end;
         candidate_begin += detail::kBoundedDirectQueueCapacity) {
      if (threadIdx.x == 0) queue_count = 0;
      __syncthreads();
      const std::size_t candidate = candidate_begin + threadIdx.x;
      if (candidate < page_end) {
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
        // Sorting changes scheduling, never the canonical scientific task orientation.
        const std::size_t first_pair = max(first_candidate, second_candidate);
        const std::size_t second_pair = min(first_candidate, second_candidate);
        if (direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
                batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
                shell_pair_density_bounds, nullptr, exchange_only, coulomb_only)) {
          const std::int32_t first_shell = batch.shell_pair_first[first_pair];
          const std::int32_t second_shell = batch.shell_pair_second[first_pair];
          const std::int32_t third_shell = batch.shell_pair_first[second_pair];
          const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
          const unsigned shell_class = direct_quartet_shell_class_device(
              batch.shell_angular[first_shell], batch.shell_angular[second_shell],
              batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
          const bool generated_class =
              bounded_generated_overflow[shell_class] == 0U &&
              bounded_generated_class_enabled(shell_class, enabled_mask_pointer, enabled_mask);
          const unsigned candidate_order =
              batch.shell_angular[first_shell] + batch.shell_angular[second_shell] +
              batch.shell_angular[third_shell] + batch.shell_angular[fourth_shell];
          // An angular pass owns an exact disjoint subset of the original
          // queue. Keep all scientific predicates and physical orientations.
          if (!generated_class && (FixedAngularOrder < 0 || candidate_order == FixedAngularOrder)) {
            const std::uint32_t slot = atomicAdd(&queue_count, 1U);
            queue[slot] = {static_cast<std::uint32_t>(first_pair),
                           static_cast<std::uint32_t>(second_pair), 0U};
            if constexpr (Force) {
              profile_bounded_direct_shell_quartet(batch, queue[slot], profile);
            }
          }
        }
      }
      __syncthreads();

      // Retained low-order specialized tasks fit in one scalar lane. Drain
      // ssss/order2 Fock and order-zero-through-three force tasks concurrently before
      // assigning generic fallback classes one warp each. psss Fock is
      // compiler-owned; if that generated class is unavailable, order one
      // deliberately falls through to the generic full-warp oracle/fallback.
      if constexpr (PairDerivatives) {
        // A complete CTA shares recurrence publication for the admitted shell.
        // The warp fallback skips precisely these dddd tasks. Retire readers
        // before queue mutation, including empty and screened component domains.
        if (radial_operator == DirectRangeOperator::FullSources) {
          auto& workspace = *reinterpret_cast<MaterializedDirectPairDerivativeRecurrence*>(
              materialized_pair_workspace);
          for (std::uint32_t slot = 0; slot < queue_count; ++slot) {
            if (materialized_pair_derivative_task(batch, queue[slot]))
              contract_materialized_direct_pair_full_source_force<Unrestricted>(
                  batch, queue[slot], screening_tolerance, schwarz_bounds, density, active, output,
                  coulomb_coefficient, exchange_coefficient, workspace);
            __syncthreads();
          }
        }
      }
      if constexpr (FixedAngularOrder < 0 || FixedAngularOrder <= 5) {
        for (std::uint32_t slot = threadIdx.x; slot < queue_count; slot += blockDim.x) {
          const ActiveShellQuartetTile task = queue[slot];
          const std::int32_t first_shell = batch.shell_pair_first[task.first_pair];
          const std::int32_t second_shell = batch.shell_pair_second[task.first_pair];
          const std::int32_t third_shell = batch.shell_pair_first[task.second_pair];
          const std::int32_t fourth_shell = batch.shell_pair_second[task.second_pair];
          const unsigned angular_order =
              FixedAngularOrder >= 0
                  ? FixedAngularOrder
                  : batch.shell_angular[first_shell] + batch.shell_angular[second_shell] +
                        batch.shell_angular[third_shell] + batch.shell_angular[fourth_shell];
          if constexpr (Force) {
            if (radial_operator == DirectRangeOperator::FullSources && angular_order == 5U) {
              const unsigned shell_class = direct_quartet_shell_class_device(
                  batch.shell_angular[first_shell], batch.shell_angular[second_shell],
                  batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
              if (weighted_order5_source_class(shell_class)) {
                contract_two_electron_force_order5_sources<Unrestricted>(
                    shell_class, batch, task, screening_tolerance, schwarz_bounds, density, active,
                    output, coulomb_coefficient, exchange_coefficient);
              }
              // f-containing order-five classes still belong to the warp fallback.
              continue;
            }
            if (radial_operator == DirectRangeOperator::FullSources && angular_order == 4U) {
              const unsigned shell_class = direct_quartet_shell_class_device(
                  batch.shell_angular[first_shell], batch.shell_angular[second_shell],
                  batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
              if (weighted_order4_source_class(shell_class)) {
                contract_two_electron_force_order4_sources<Unrestricted>(
                    shell_class, batch, task, screening_tolerance, schwarz_bounds, density, active,
                    output, coulomb_coefficient, exchange_coefficient);
              }
              // Uncovered order-four classes are consumed by the warp fallback.
              continue;
            }
            if (radial_operator == DirectRangeOperator::Long && angular_order <= 3U) {
              const unsigned shell_class = direct_quartet_shell_class_device(
                  batch.shell_angular[first_shell], batch.shell_angular[second_shell],
                  batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
              contract_two_electron_force_low_order_sources<Unrestricted, true>(
                  shell_class, batch, task, screening_tolerance, schwarz_bounds, density, active,
                  output, 0.0, exchange_coefficient, omega);
              continue;
            }
            if (radial_operator != DirectRangeOperator::Full &&
                radial_operator != DirectRangeOperator::FullSources) {
              continue;
            }
            if (angular_order <= 3U) {
              if (radial_operator == DirectRangeOperator::FullSources)
                contract_direct_force_precontracted_task<Unrestricted,
                                                         DirectForceOutputMode::Separate>(
                    batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U,
                    coulomb_coefficient, exchange_coefficient);
              else
                contract_direct_force_precontracted_task<Unrestricted,
                                                         DirectForceOutputMode::Combined>(
                    batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U,
                    coulomb_coefficient, exchange_coefficient);
            }
          } else {
            // The scalar low-order Fock shortcuts are full-range identities.
            // SR/LR exchange falls through to the compiler-owned Cartesian
            // range recurrence in the warp contraction below.
            if (radial_operator == DirectRangeOperator::Full && angular_order == 0U) {
              contract_fock_direct_quartet_subtile<Unrestricted, 0U>(
                  batch, &queue_count, queue + slot, screening_tolerance, schwarz_bounds, density,
                  active, output, nullptr, 0U, 0U, coulomb_only, exchange_only);
            } else if (radial_operator == DirectRangeOperator::Full && angular_order == 2U) {
              contract_fock_direct_order2_task<Unrestricted>(
                  batch, task, screening_tolerance, schwarz_bounds, density, active, output,
                  nullptr, coulomb_only, exchange_only);
            }
          }
        }
      }
      __syncthreads();

      if constexpr (FixedAngularOrder < 0 || FixedAngularOrder >= 4) {
        for (std::uint32_t slot = warp; slot < queue_count;
             slot += kBoundedDirectThreads / detail::kDirectQuartetThreads) {
          const ActiveShellQuartetTile base = queue[slot];
          const std::int32_t first_shell = batch.shell_pair_first[base.first_pair];
          const std::int32_t second_shell = batch.shell_pair_second[base.first_pair];
          const std::int32_t third_shell = batch.shell_pair_first[base.second_pair];
          const std::int32_t fourth_shell = batch.shell_pair_second[base.second_pair];
          const unsigned angular_order =
              FixedAngularOrder >= 0
                  ? FixedAngularOrder
                  : batch.shell_angular[first_shell] + batch.shell_angular[second_shell] +
                        batch.shell_angular[third_shell] + batch.shell_angular[fourth_shell];
          const unsigned shell_class = direct_quartet_shell_class_device(
              batch.shell_angular[first_shell], batch.shell_angular[second_shell],
              batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
          if constexpr (Force) {
            if constexpr (PairDerivatives) {
              if (radial_operator == DirectRangeOperator::FullSources &&
                  materialized_pair_derivative_task(batch, base))
                continue;
            }
            if (radial_operator == DirectRangeOperator::FullSources &&
                (weighted_order4_source_class(shell_class) ||
                 weighted_order5_source_class(shell_class)))
              continue;
            // Full/LR order 0--3 was consumed once by the scalar shell workers.
            // Short range, fused RSH and higher orders retain their qualified
            // Cartesian/AOT recurrence and bounded queue traversal.
            if ((radial_operator == DirectRangeOperator::Full ||
                 radial_operator == DirectRangeOperator::FullSources ||
                 radial_operator == DirectRangeOperator::Long) &&
                angular_order <= 3U)
              continue;
          } else {
            // Fock order one has no psss-specific handwritten fallback anymore.
            // Full-range order zero/two were consumed above. Range exchange must
            // traverse every angular class because full-range value kernels
            // cannot substitute SR/LR mathematics.
            if (radial_operator == DirectRangeOperator::Full &&
                (angular_order == 0U || angular_order == 2U))
              continue;
          }
          const std::size_t first_ao_count = shell_ao_pair_count(batch, base.first_pair);
          const std::size_t second_ao_count = shell_ao_pair_count(batch, base.second_pair);
          const std::size_t ao_quartets = base.first_pair == base.second_pair
                                              ? first_ao_count * (first_ao_count + 1) / 2
                                              : first_ao_count * second_ao_count;
          const std::uint32_t tile_count = static_cast<std::uint32_t>(
              (ao_quartets + detail::kDirectQuartetTileSize - 1) / detail::kDirectQuartetTileSize);
          const std::size_t subtile_count = detail::direct_quartet_subtiles_per_tile(angular_order);
          for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
            if (lane == 0) queue[slot].tile = tile;
            __syncwarp();
            for (std::size_t subtile = 0; subtile < subtile_count; ++subtile) {
              if constexpr (Force) {
                if constexpr (FixedAngularOrder >= 4 &&
                              FixedRadialOperator ==
                                  static_cast<int>(DirectRangeOperator::FullSources)) {
                  contract_two_electron_force_quartet_subtile_scaled<Unrestricted,
                                                                     FixedAngularOrder, true>(
                      batch, &queue_count, queue + slot, screening_tolerance, schwarz_bounds,
                      density, active, output, 0U, coulomb_coefficient, exchange_coefficient,
                      subtile, lane);
                } else if constexpr (FixedAngularOrder >= 4 &&
                                     FixedRadialOperator ==
                                         static_cast<int>(DirectRangeOperator::Long)) {
                  contract_two_electron_force_quartet_subtile_range_scaled<Unrestricted,
                                                                           FixedAngularOrder>(
                      batch, &queue_count, queue + slot, screening_tolerance, schwarz_bounds,
                      density, active, output, exchange_coefficient,
                      generativeqc::integrals::CoulombRange::Long, omega, subtile, lane);
                } else {
                  if (radial_operator == DirectRangeOperator::Full) {
                    contract_bounded_direct_force_subtile_scaled<Unrestricted>(
                        batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                        schwarz_bounds, density, active, output, coulomb_coefficient,
                        exchange_coefficient, subtile, lane);
                  } else if (radial_operator == DirectRangeOperator::FullSources) {
                    contract_bounded_direct_force_subtile_scaled<Unrestricted, true>(
                        batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                        schwarz_bounds, density, active, output, coulomb_coefficient,
                        exchange_coefficient, subtile, lane);
                  } else if (radial_operator == DirectRangeOperator::RshSources) {
                    contract_bounded_direct_rsh_force_subtile<Unrestricted>(
                        batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                        schwarz_bounds, density, active, output, coulomb_coefficient,
                        exchange_coefficient, secondary_exchange_coefficient, omega, subtile, lane);
                  } else if (contract_packaged_derivative_shell_aot<Unrestricted>(
                                 shell_class, radial_operator, omega, batch, &queue_count,
                                 queue + slot, screening_tolerance, schwarz_bounds, density, active,
                                 output, exchange_coefficient, subtile, lane)) {
                    // Exact shell-class package consumed this SR/LR subtile.
                  } else if (omega == 0.3 && radial_operator == DirectRangeOperator::Long) {
                    contract_bounded_direct_force_subtile_range_aot_scaled<
                        Unrestricted, generativeqc::integrals::CoulombRange::Long, 300>(
                        batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                        schwarz_bounds, density, active, output, exchange_coefficient, subtile,
                        lane);
                  } else if (omega == 0.3 && radial_operator == DirectRangeOperator::Short) {
                    contract_bounded_direct_force_subtile_range_aot_scaled<
                        Unrestricted, generativeqc::integrals::CoulombRange::Short, 300>(
                        batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                        schwarz_bounds, density, active, output, exchange_coefficient, subtile,
                        lane);
                  } else {
                    const auto range = radial_operator == DirectRangeOperator::Long
                                           ? generativeqc::integrals::CoulombRange::Long
                                           : generativeqc::integrals::CoulombRange::Short;
                    contract_bounded_direct_force_subtile_range_scaled<Unrestricted>(
                        batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                        schwarz_bounds, density, active, output, exchange_coefficient, range, omega,
                        subtile, lane);
                  }
                }
              } else if (radial_operator == DirectRangeOperator::Full) {
                contract_bounded_direct_fock_subtile<Unrestricted>(
                    batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                    schwarz_bounds, density, active, output, subtile, lane, coulomb_only,
                    exchange_only);
              } else {
                const auto range = radial_operator == DirectRangeOperator::Long
                                       ? generativeqc::integrals::CoulombRange::Long
                                       : generativeqc::integrals::CoulombRange::Short;
                contract_bounded_direct_fock_subtile<Unrestricted>(
                    batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                    schwarz_bounds, density, active, output, subtile, lane, false, true, range,
                    omega);
              }
            }
            __syncwarp();
          }
        }
      }
      __syncthreads();
    }
  }
}

/**
 * Diagnostic angular partition of the retained molecular force consumer.
 *
 * Each pass has the same bounded queue and scientific gates. Enumeration is
 * repeated thirteen times, but admitted source evaluation is disjoint. This
 * deliberately exposes the work/occupancy tradeoff before a class-indexed
 * planner is justified; no register cap or different recurrence is introduced.
 * The cursor is reused sequentially on the caller's stream, so no new storage
 * or lifetime contract is needed.
 */
template <bool Unrestricted, DirectRangeOperator Range, unsigned Order = 0U>
cudaError_t launch_angular_force_passes(
    dim3 grid, dim3 block, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* pair_order, const double* block_bounds, const double* system_bounds,
    const std::uint32_t* class_state, const double* schwarz, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    double coulomb_coefficient, double exchange_coefficient, double omega,
    detail::BoundedDirectBlockDomain domain, DirectForceResidentBraSchedule resident) {
  auto launch_bounded = [&]() {
    auto error = cudaMemsetAsync(cursor, 0, sizeof(*cursor), stream);
    if (error != cudaSuccess) return error;
    auto launch = [&]<bool PairDerivatives>() {
      constexpr std::size_t shared_bytes =
          PairDerivatives ? sizeof(MaterializedDirectPairDerivativeRecurrence) : 0U;
      bounded_direct_shell_quartet_kernel<Unrestricted, DirectScreeningPurpose::Force, true, Order,
                                          static_cast<int>(Range), PairDerivatives>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds, pair_order,
              block_bounds, system_bounds, nullptr, 0U, class_state, schwarz, density, active,
              output, cursor, nullptr, coulomb_coefficient, exchange_coefficient, Range, omega, 0.0,
              false, Range == DirectRangeOperator::Long, domain);
    };
    if constexpr (Order == 8 && Range == DirectRangeOperator::FullSources) {
      if (materialized_pair_derivative_available(batch))
        launch.template operator()<true>();
      else
        launch.template operator()<false>();
    } else {
      launch.template operator()<false>();
    }
    return cudaGetLastError();
  };
  cudaError_t error = cudaSuccess;
  if constexpr (Order == 1U && Range == DirectRangeOperator::FullSources) {
    // The resident lease owns the complete psss class. Replace this pass,
    // rather than adding a second traversal or masking individual channels.
    error = direct_force_resident_bra_schedule_available(resident)
                ? launch_direct_force_resident_bra(DirectForceOutputMode::Separate, Unrestricted,
                                                   stream, batch, resident, screening_tolerance,
                                                   shell_pair_bounds, shell_pair_density_bounds,
                                                   true, schwarz, density, active, output, 0U,
                                                   coulomb_coefficient, exchange_coefficient)
                : launch_bounded();
  } else {
    error = launch_bounded();
  }
  if (error != cudaSuccess) return error;
  if constexpr (Order < 12U)
    return launch_angular_force_passes<Unrestricted, Range, Order + 1U>(
        grid, block, stream, batch, screening_tolerance, shell_pair_bounds,
        shell_pair_density_bounds, pair_order, block_bounds, system_bounds, class_state, schwarz,
        density, active, output, cursor, coulomb_coefficient, exchange_coefficient, omega, domain,
        resident);
  return cudaSuccess;
}

cudaError_t launch_bounded_direct_angular_force_kernel(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* block_bounds, const double* system_bounds, const std::uint32_t* class_state,
    const double* schwarz, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* cursor, DirectRangeOperator range, double omega, double coulomb_coefficient,
    double exchange_coefficient, detail::BoundedDirectBlockDomain domain,
    DirectForceResidentBraSchedule resident) {
  if (range != DirectRangeOperator::FullSources && range != DirectRangeOperator::Long)
    return cudaErrorInvalidValue;
#define GENERATIVEQC_ANGULAR_FORCE(U, R)                                                           \
  launch_angular_force_passes<U, R>(                                                               \
      worker_blocks, kBoundedDirectThreads, stream, batch, screening_tolerance, shell_pair_bounds, \
      shell_pair_density_bounds, pair_order, block_bounds, system_bounds, class_state, schwarz,    \
      density, active, output, cursor, coulomb_coefficient, exchange_coefficient, omega, domain,   \
      resident)
  if (range == DirectRangeOperator::Long)
    return unrestricted ? GENERATIVEQC_ANGULAR_FORCE(true, DirectRangeOperator::Long)
                        : GENERATIVEQC_ANGULAR_FORCE(false, DirectRangeOperator::Long);
  return unrestricted ? GENERATIVEQC_ANGULAR_FORCE(true, DirectRangeOperator::FullSources)
                      : GENERATIVEQC_ANGULAR_FORCE(false, DirectRangeOperator::FullSources);
#undef GENERATIVEQC_ANGULAR_FORCE
}

void launch_bounded_direct_shell_quartet_kernel_scaled(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile,
    double coulomb_coefficient, double exchange_coefficient, bool separate_sources,
    detail::BoundedDirectBlockDomain block_domain) {
  const auto radial_operator =
      separate_sources ? DirectRangeOperator::FullSources : DirectRangeOperator::Full;
  auto launch = [&]<bool Unrestricted, DirectScreeningPurpose Purpose, bool PairDerivatives>() {
    const auto workspace_bytes =
        PairDerivatives ? std::max(shared_bytes, sizeof(MaterializedDirectPairDerivativeRecurrence))
                        : shared_bytes;
    bounded_direct_shell_quartet_kernel<Unrestricted, Purpose, true, -1, -1, PairDerivatives>
        <<<grid, block, workspace_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, enabled_mask_pointer,
            enabled_mask, bounded_generated_overflow, schwarz_bounds, density, active, output,
            global_cursor, profile, coulomb_coefficient, exchange_coefficient, radial_operator, 0.0,
            0.0, false, false, block_domain);
  };
  auto select = [&]<bool Unrestricted, DirectScreeningPurpose Purpose>() {
    // Keep the default-off kernel's register and shared-memory footprint. The
    // candidate has its own specialization and borrows only per-CTA workspace.
    if (separate_sources && materialized_pair_derivative_available(batch))
      launch.template operator()<Unrestricted, Purpose, true>();
    else
      launch.template operator()<Unrestricted, Purpose, false>();
  };
  if (unrestricted) {
    if (purpose == DirectScreeningPurpose::Fock)
      select.template operator()<true, DirectScreeningPurpose::Fock>();
    else
      select.template operator()<true, DirectScreeningPurpose::Force>();
  } else {
    if (purpose == DirectScreeningPurpose::Fock)
      select.template operator()<false, DirectScreeningPurpose::Fock>();
    else
      select.template operator()<false, DirectScreeningPurpose::Force>();
  }
}

void launch_bounded_direct_range_exchange_force_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DirectRangeOperator radial_operator, double omega,
    double exchange_coefficient, detail::BoundedDirectBlockDomain block_domain) {
  if (radial_operator == DirectRangeOperator::Full) return;
  // This consumer publishes only range-separated K derivatives. Select its
  // raw-K linear bound and same-spin force-product bounds; the mixed J/K
  // defaults would retain distant Coulomb-only shell quartets unnecessarily.
  if (unrestricted) {
    bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Force, true>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, nullptr, 0U,
            bounded_generated_overflow, schwarz_bounds, density, active, output, global_cursor,
            nullptr, 0.0, exchange_coefficient, radial_operator, omega, 0.0, false, true,
            block_domain);
  } else {
    bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Force, true>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, nullptr, 0U,
            bounded_generated_overflow, schwarz_bounds, density, active, output, global_cursor,
            nullptr, 0.0, exchange_coefficient, radial_operator, omega, 0.0, false, true,
            block_domain);
  }
}

void launch_bounded_direct_range_exchange_fock_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DirectRangeOperator radial_operator, double omega) {
  if (radial_operator == DirectRangeOperator::Full) return;
  if (unrestricted) {
    bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Fock, false>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, nullptr, 0U,
            bounded_generated_overflow, schwarz_bounds, density, active, output, global_cursor,
            nullptr, 0.0, -1.0, radial_operator, omega, 0.0, false, true);
  } else {
    bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Fock, false>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, nullptr, 0U,
            bounded_generated_overflow, schwarz_bounds, density, active, output, global_cursor,
            nullptr, 0.0, -0.5, radial_operator, omega, 0.0, false, true);
  }
}

void launch_bounded_direct_rsh_force_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* source_forces,
    unsigned long long* global_cursor, double omega, double coulomb_coefficient,
    double short_exchange_coefficient, double long_exchange_coefficient) {
  if (unrestricted) {
    bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Force, true>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, nullptr, 0U,
            bounded_generated_overflow, schwarz_bounds, density, active, source_forces,
            global_cursor, nullptr, coulomb_coefficient, short_exchange_coefficient,
            DirectRangeOperator::RshSources, omega, long_exchange_coefficient, false, false);
  } else {
    bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Force, true>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, nullptr, 0U,
            bounded_generated_overflow, schwarz_bounds, density, active, source_forces,
            global_cursor, nullptr, coulomb_coefficient, short_exchange_coefficient,
            DirectRangeOperator::RshSources, omega, long_exchange_coefficient, false, false);
  }
}

void launch_bounded_direct_shell_quartet_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile) {
  launch_bounded_direct_shell_quartet_kernel_scaled(
      unrestricted, purpose, grid, block, shared_bytes, stream, batch, screening_tolerance,
      shell_pair_bounds, shell_pair_density_bounds, shell_pair_order, shell_pair_block_bounds,
      system_density_bounds, enabled_mask_pointer, enabled_mask, bounded_generated_overflow,
      schwarz_bounds, density, active, output, global_cursor, profile, 1.0,
      unrestricted ? -1.0 : -0.5);
}

void launch_bounded_direct_fock_shell_quartet_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock,
    unsigned long long* global_cursor) {
  if (unrestricted) {
    bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Fock, false>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, enabled_mask_pointer,
            enabled_mask, bounded_generated_overflow, schwarz_bounds, density, active, fock,
            global_cursor, nullptr, 1.0, unrestricted ? -1.0 : -0.5, DirectRangeOperator::Full, 0.0,
            0.0, false, false);
  } else {
    bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Fock, false>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, enabled_mask_pointer,
            enabled_mask, bounded_generated_overflow, schwarz_bounds, density, active, fock,
            global_cursor, nullptr, 1.0, unrestricted ? -1.0 : -0.5, DirectRangeOperator::Full, 0.0,
            0.0, false, false);
  }
}

void launch_bounded_direct_fock_source_shell_quartet_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    std::uint64_t covered_shell_class_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, bool coulomb_only, bool exchange_only) {
  if (unrestricted) {
    bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Fock, false>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, nullptr,
            covered_shell_class_mask, bounded_generated_overflow, schwarz_bounds, density, active,
            output, global_cursor, nullptr, 1.0, -1.0, DirectRangeOperator::Full, 0.0, 0.0,
            coulomb_only, exchange_only);
  } else {
    bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Fock, false>
        <<<grid, block, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, nullptr,
            covered_shell_class_mask, bounded_generated_overflow, schwarz_bounds, density, active,
            output, global_cursor, nullptr, 1.0, -0.5, DirectRangeOperator::Full, 0.0, 0.0,
            coulomb_only, exchange_only);
  }
}

}  // namespace generativeqc::scf::cuda_execution
