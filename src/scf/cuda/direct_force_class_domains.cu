#include <cuda_runtime.h>

#include <algorithm>

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
