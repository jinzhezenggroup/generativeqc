#include <cuda_runtime.h>

#include <algorithm>

#include "scf/cuda/direct_bounded_queue_page.cuh"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_force_class_domains.hpp"
#include "scf/cuda/direct_force_class_pages.cuh"
#include "scf/cuda/direct_force_execution.cuh"
#include "scf/cuda/direct_force_order4_sources.cuh"
#include "scf/cuda/direct_force_order5_sources.cuh"
#include "scf/cuda/direct_force_quartet.cuh"
#include "scf/cuda/direct_queue_profile.cuh"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/direct_task_encoding.cuh"

namespace generativeqc::scf::cuda_execution {

/** Preserve the incumbent's scalar/warp ownership without instantiating its
 * unreachable order-seven-through-twelve private recurrence frames. Each warp
 * owns one order-six task at a time; its local tile view needs no shared writes. */
template <bool Unrestricted, DirectForceOutputMode Mode, bool SpatialReuse = false>
__device__ __forceinline__ void contract_bounded_spd_low_queue(
    const DeviceBatch& batch, const ActiveShellQuartetTile* queue, std::uint32_t queue_count,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, double coulomb_coefficient,
    double exchange_coefficient) {
  for (std::uint32_t slot = threadIdx.x; slot < queue_count; slot += blockDim.x) {
    const auto task = queue[slot];
    const auto shell_class = direct_force_task_shell_class(batch, task);
    const auto order = direct_shell_class_angular_order(shell_class);
    if (order <= 3U)
      contract_direct_force_precontracted_task<Unrestricted, Mode>(
          batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U,
          coulomb_coefficient, exchange_coefficient);
    else if (order == 4U)
      contract_two_electron_force_order4_sources<Unrestricted, Mode>(
          shell_class, batch, task, screening_tolerance, schwarz_bounds, density, active, output,
          coulomb_coefficient, exchange_coefficient);
    else if (order == 5U)
      contract_two_electron_force_order5_sources<Unrestricted, Mode>(
          shell_class, batch, task, screening_tolerance, schwarz_bounds, density, active, output,
          coulomb_coefficient, exchange_coefficient);
  }
  __syncthreads();
  const unsigned lane = threadIdx.x % detail::kDirectQuartetThreads;
  const unsigned warp = threadIdx.x / detail::kDirectQuartetThreads;
  for (std::uint32_t slot = warp; slot < queue_count;
       slot += blockDim.x / detail::kDirectQuartetThreads) {
    auto task = queue[slot];
    if (direct_shell_class_angular_order(direct_force_task_shell_class(batch, task)) != 6U)
      continue;
    const auto first_count = shell_ao_pair_count(batch, task.first_pair);
    const auto second_count = shell_ao_pair_count(batch, task.second_pair);
    const auto components = task.first_pair == task.second_pair
                                ? first_count * (first_count + 1U) / 2U
                                : first_count * second_count;
    const auto tiles =
        (components + detail::kDirectQuartetTileSize - 1U) / detail::kDirectQuartetTileSize;
    const std::uint32_t one_tile = 1U;
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
      task.tile = tile;
      for (std::size_t subtile = 0; subtile < detail::direct_quartet_subtiles_per_tile(6U);
           ++subtile)
        contract_two_electron_force_quartet_subtile_scaled<
            Unrestricted, 6U, Mode == DirectForceOutputMode::Separate, SpatialReuse>(
            batch, &one_tile, &task, screening_tolerance, schwarz_bounds, density, active, output,
            0U, coulomb_coefficient, exchange_coefficient, subtile, lane);
    }
  }
}

/** A single bounded inventory owns orders zero through six. Whole-block
 * rejection stays ahead of candidate geometry and exact quartet screening.
 * The incumbent's separate order-seven/eight consumers remain unchanged. */
template <bool Unrestricted, DirectForceOutputMode Mode>
__global__
__launch_bounds__(kBoundedDirectForceThreads, 1) void direct_force_spatial_spd_low_kernel(
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient, double exchange_coefficient,
    detail::BoundedDirectBlockDomain block_domain) {
  __shared__ ActiveShellQuartetTile queue[detail::kBoundedDirectQueueCapacity];
  __shared__ std::uint32_t queue_count;
  __shared__ BoundedDirectQueuePage page;
  __shared__ bool available;
  while (true) {
    __syncthreads();
    if (threadIdx.x == 0U)
      available =
          bounded_direct_queue_block_page(batch, block_domain, atomicAdd(cursor, 1ULL), page);
    __syncthreads();
    if (!available) return;
    if (active && active[page.system] == 0U) continue;
    if (!bounded_direct_block_pair_survives_screening<DirectScreeningPurpose::Force>(
            page.first_block, page.second_block, page.system, screening_tolerance,
            shell_pair_block_bounds, system_density_bounds))
      continue;
    if (threadIdx.x == 0U)
      bounded_direct_queue_candidate_page(batch, block_domain.prefix != nullptr, page);
    __syncthreads();
    for (auto packet = page.candidate_begin; packet < page.candidate_end; packet += blockDim.x) {
      if (threadIdx.x == 0U) queue_count = 0U;
      __syncthreads();
      const auto candidate = packet + threadIdx.x;
      if (candidate < page.candidate_end) {
        std::size_t first_local{}, second_local{};
        if (page.same_block)
          decode_lower_triangle(candidate, first_local, second_local);
        else {
          first_local = candidate / page.second_count;
          second_local = candidate % page.second_count;
        }
        const auto first_candidate = shell_pair_order[page.first_ordered_begin + first_local];
        const auto second_candidate = shell_pair_order[page.second_ordered_begin + second_local];
        const auto first_pair =
            first_candidate > second_candidate ? first_candidate : second_candidate;
        const auto second_pair =
            first_candidate < second_candidate ? first_candidate : second_candidate;
        const ActiveShellQuartetTile task{first_pair, second_pair, 0U};
        const auto shell_class = direct_force_task_shell_class(batch, task);
        if (direct_shell_class_angular_order(shell_class) <= 6U &&
            direct_shell_quartet_survives_screening<Unrestricted, DirectScreeningPurpose::Force>(
                batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
                shell_pair_density_bounds, nullptr, false, false) &&
            (generated_overflow[shell_class] != 0U ||
             !bounded_generated_class_enabled(shell_class, enabled_mask_pointer, enabled_mask))) {
          queue[atomicAdd(&queue_count, 1U)] = task;
          profile_bounded_direct_shell_quartet(batch, task, profile);
        }
      }
      __syncthreads();
      contract_bounded_spd_low_queue<Unrestricted, Mode, true>(
          batch, queue, queue_count, screening_tolerance, schwarz_bounds, density, active, output,
          coulomb_coefficient, exchange_coefficient);
      __syncthreads();
    }
  }
}

cudaError_t launch_direct_force_spatial_spd_low(
    bool unrestricted, dim3 grid, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient, double exchange_coefficient,
    bool separate_sources, detail::BoundedDirectBlockDomain block_domain) {
  if (batch.direct_maximum_shell_angular != 2U || !block_domain.prefix)
    return cudaErrorInvalidValue;
  auto launch = [&]<bool Unrestricted, DirectForceOutputMode Mode>() {
    direct_force_spatial_spd_low_kernel<Unrestricted, Mode>
        <<<grid, kBoundedDirectForceThreads, shared_bytes, stream>>>(
            batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            shell_pair_order, shell_pair_block_bounds, system_density_bounds, enabled_mask_pointer,
            enabled_mask, generated_overflow, schwarz_bounds, density, active, output, cursor,
            profile, coulomb_coefficient, exchange_coefficient, block_domain);
    return cudaPeekAtLastError();
  };
  if (unrestricted)
    return separate_sources ? launch.template operator()<true, DirectForceOutputMode::Separate>()
                            : launch.template operator()<true, DirectForceOutputMode::Combined>();
  return separate_sources ? launch.template operator()<false, DirectForceOutputMode::Separate>()
                          : launch.template operator()<false, DirectForceOutputMode::Combined>();
}

/** Each specialization instantiates only its existing scientific consumer,
 * never the private generic Coulomb/AD frame of an unrelated angular class. */
template <bool Unrestricted, DirectForceOutputMode Mode, DirectForceClassDomain Domain>
__global__ __launch_bounds__(kBoundedDirectThreads, 1) void direct_force_class_domain_kernel(
    DeviceBatch batch, const GeneratedShellPairStream* topology_pointer, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient,
    double exchange_coefficient) {
  extern __shared__ __align__(16) unsigned char recurrence_workspace[];
  __shared__ ActiveShellQuartetTile queue[direct_force_class_page_size(Domain)];
  __shared__ std::uint32_t queue_count;
  __shared__ DirectForceClassPage page;
  __shared__ bool available;
  DirectForceClassPagePosition position{};
  const auto& topology = *topology_pointer;
  while (true) {
    __syncthreads();
    if (threadIdx.x == 0U) {
      available = direct_force_class_pair_page<Domain>(
          topology.pair_class_offsets, batch.batch_size, atomicAdd(cursor, 1ULL), position, page);
      queue_count = 0U;
    }
    __syncthreads();
    if (!available) return;
    if (active && active[page.system] == 0U) continue;
    if (threadIdx.x < page.ket_end - page.ket_begin) {
      const auto bra = topology.pair_order[page.bra];
      const auto ket = topology.pair_order[page.ket_begin + threadIdx.x];
      const auto first_pair = bra > ket ? bra : ket;
      const auto second_pair = bra < ket ? bra : ket;
      if (direct_shell_quartet_survives_screening<Unrestricted, DirectScreeningPurpose::Force>(
              batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds, nullptr, false, false)) {
        const ActiveShellQuartetTile task{first_pair, second_pair, 0U};
        const auto shell_class = direct_force_task_shell_class(batch, task);
        if (generated_overflow[shell_class] != 0U ||
            !bounded_generated_class_enabled(shell_class, enabled_mask_pointer, enabled_mask)) {
          queue[atomicAdd(&queue_count, 1U)] = task;
          profile_bounded_direct_shell_quartet(batch, task, profile);
        }
      }
    }
    __syncthreads();
    if constexpr (Domain == DirectForceClassDomain::Cooperative ||
                  Domain == DirectForceClassDomain::Materialized) {
      for (std::uint32_t slot = 0; slot < queue_count; ++slot) {
        if constexpr (Domain == DirectForceClassDomain::Cooperative) {
          auto& workspace =
              *reinterpret_cast<CooperativeDirectPairDerivativeRecurrence*>(recurrence_workspace);
          contract_cooperative_direct_pair_force<Unrestricted, Mode>(
              batch, queue[slot], screening_tolerance, schwarz_bounds, density, active, output,
              coulomb_coefficient, exchange_coefficient, workspace);
        } else {
          auto& workspace =
              *reinterpret_cast<MaterializedDirectPairDerivativeRecurrence*>(recurrence_workspace);
          contract_materialized_direct_pair_force<Unrestricted, Mode>(
              batch, queue[slot], screening_tolerance, schwarz_bounds, density, active, output,
              coulomb_coefficient, exchange_coefficient, workspace);
        }
        __syncthreads();
      }
    } else {
      for (std::uint32_t slot = threadIdx.x; slot < queue_count; slot += blockDim.x) {
        const auto task = queue[slot];
        if constexpr (Domain == DirectForceClassDomain::LowOrder) {
          contract_direct_force_precontracted_task<Unrestricted, Mode>(
              batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U,
              coulomb_coefficient, exchange_coefficient);
        } else if constexpr (Domain == DirectForceClassDomain::WeightedFour) {
          contract_two_electron_force_order4_sources<Unrestricted, Mode>(
              direct_force_task_shell_class(batch, task), batch, task, screening_tolerance,
              schwarz_bounds, density, active, output, coulomb_coefficient, exchange_coefficient);
        } else {
          contract_two_electron_force_order5_sources<Unrestricted, Mode>(
              direct_force_task_shell_class(batch, task), batch, task, screening_tolerance,
              schwarz_bounds, density, active, output, coulomb_coefficient, exchange_coefficient);
        }
      }
    }
  }
}

cudaError_t launch_direct_force_class_domains(
    bool unrestricted, dim3 grid, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* topology, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient, double exchange_coefficient,
    bool separate_sources) {
  auto launch =
      [&]<bool Unrestricted, DirectForceOutputMode Mode, DirectForceClassDomain Domain>() {
        auto error = cudaMemsetAsync(cursor, 0, sizeof(*cursor), stream);
        if (error != cudaSuccess) return error;
        std::size_t workspace_bytes = shared_bytes;
        unsigned threads = kBoundedDirectForceThreads;
        if constexpr (Domain == DirectForceClassDomain::Cooperative) {
          workspace_bytes =
              std::max(workspace_bytes, sizeof(CooperativeDirectPairDerivativeRecurrence));
          threads = kBoundedDirectThreads;
        } else if constexpr (Domain == DirectForceClassDomain::Materialized) {
          workspace_bytes =
              std::max(workspace_bytes, sizeof(MaterializedDirectPairDerivativeRecurrence));
          threads = kBoundedDirectThreads;
        }
        direct_force_class_domain_kernel<Unrestricted, Mode, Domain>
            <<<grid, threads, workspace_bytes, stream>>>(
                batch, topology, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
                enabled_mask_pointer, enabled_mask, generated_overflow, schwarz_bounds, density,
                active, output, cursor, profile, coulomb_coefficient, exchange_coefficient);
        return cudaPeekAtLastError();
      };
  auto domains = [&]<bool Unrestricted, DirectForceOutputMode Mode>() {
    auto error = launch.template operator()<Unrestricted, Mode, DirectForceClassDomain::LowOrder>();
    if (error != cudaSuccess) return error;
    error = launch.template operator()<Unrestricted, Mode, DirectForceClassDomain::WeightedFour>();
    if (error != cudaSuccess) return error;
    error = launch.template operator()<Unrestricted, Mode, DirectForceClassDomain::WeightedFive>();
    if (error != cudaSuccess) return error;
    error = launch.template operator()<Unrestricted, Mode, DirectForceClassDomain::Cooperative>();
    if (error != cudaSuccess) return error;
    return launch.template operator()<Unrestricted, Mode, DirectForceClassDomain::Materialized>();
  };
  if (unrestricted)
    return separate_sources ? domains.template operator()<true, DirectForceOutputMode::Separate>()
                            : domains.template operator()<true, DirectForceOutputMode::Combined>();
  return separate_sources ? domains.template operator()<false, DirectForceOutputMode::Separate>()
                          : domains.template operator()<false, DirectForceOutputMode::Combined>();
}

}  // namespace generativeqc::scf::cuda_execution
