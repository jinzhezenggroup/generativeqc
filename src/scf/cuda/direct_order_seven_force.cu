#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_force_quartet.cuh"
#include "scf/cuda/direct_order_seven_force.hpp"
#include "scf/cuda/direct_order_seven_pages.cuh"
#include "scf/cuda/direct_queue_profile.cuh"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/direct_task_encoding.cuh"

namespace generativeqc::scf::cuda_execution {

static_assert(direct_shell_pair_class_cuda(2U, 2U) == kDirectDdPairClass);
static_assert(direct_shell_pair_class_cuda(2U, 1U) == kDirectDpPairClass);

/** Stream only dd x dp pages, not another full shell-pair triangle. All lanes
 * cooperate on the unchanged generated derivative; reducing block width would
 * silently omit coefficients indexed by the generated 256-lane tile. */
template <bool Unrestricted, DirectForceOutputMode Mode>
__global__ __launch_bounds__(kBoundedDirectThreads, 1) void direct_order_seven_force_kernel(
    DeviceBatch batch, const GeneratedShellPairStream* topology_pointer, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient,
    double exchange_coefficient) {
  extern __shared__ __align__(16) unsigned char cooperative_workspace[];
  __shared__ ActiveShellQuartetTile queue[kDirectOrderSevenKetPageSize];
  __shared__ std::uint32_t queue_count;
  __shared__ DirectOrderSevenPairPage page;
  __shared__ bool page_available;
  DirectOrderSevenPagePosition position{};
  const auto& topology = *topology_pointer;
  auto& workspace =
      *reinterpret_cast<CooperativeDirectPairDerivativeRecurrence*>(cooperative_workspace);
  while (true) {
    // Inactive and empty claims also finish reading the old shared descriptor
    // before the leader publishes another page.
    __syncthreads();
    if (threadIdx.x == 0U) {
      const auto claim = atomicAdd(cursor, 1ULL);
      page_available = direct_order_seven_pair_page(topology.pair_class_offsets, batch.batch_size,
                                                    claim, position, page);
      queue_count = 0U;
    }
    __syncthreads();
    if (!page_available) return;
    if (active != nullptr && active[page.system] == 0U) continue;

    if (threadIdx.x < page.ket_end - page.ket_begin) {
      const auto bra = topology.pair_order[page.bra];
      const auto ket = topology.pair_order[page.ket_begin + threadIdx.x];
      const auto first_pair = max(bra, ket);
      const auto second_pair = min(bra, ket);
      if (direct_shell_quartet_survives_screening<Unrestricted, DirectScreeningPurpose::Force>(
              batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds, nullptr, false, false)) {
        const ActiveShellQuartetTile task{first_pair, second_pair, 0U};
        const auto shell_class = direct_quartet_shell_class_device(
            batch.shell_angular[batch.shell_pair_first[first_pair]],
            batch.shell_angular[batch.shell_pair_second[first_pair]],
            batch.shell_angular[batch.shell_pair_first[second_pair]],
            batch.shell_angular[batch.shell_pair_second[second_pair]]);
        const bool generated_class =
            generated_overflow[shell_class] == 0U &&
            bounded_generated_class_enabled(shell_class, enabled_mask_pointer, enabled_mask);
        if (!generated_class) {
          const auto slot = atomicAdd(&queue_count, 1U);
          queue[slot] = task;
          profile_bounded_direct_shell_quartet(batch, task, profile);
        }
      }
    }
    __syncthreads();
    for (std::uint32_t slot = 0; slot < queue_count; ++slot) {
      contract_cooperative_direct_pair_force<Unrestricted, Mode>(
          batch, queue[slot], screening_tolerance, schwarz_bounds, density, active, output,
          coulomb_coefficient, exchange_coefficient, workspace);
      __syncthreads();
    }
  }
}

cudaError_t launch_direct_order_seven_force(
    bool unrestricted, dim3 grid, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* topology, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient, double exchange_coefficient,
    bool separate_sources) {
  const auto workspace_bytes =
      std::max(shared_bytes, sizeof(CooperativeDirectPairDerivativeRecurrence));
  auto launch = [&]<bool Unrestricted, DirectForceOutputMode Mode>() {
    direct_order_seven_force_kernel<Unrestricted, Mode>
        <<<grid, kBoundedDirectThreads, workspace_bytes, stream>>>(
            batch, topology, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            enabled_mask_pointer, enabled_mask, generated_overflow, schwarz_bounds, density, active,
            output, cursor, profile, coulomb_coefficient, exchange_coefficient);
  };
  if (unrestricted) {
    if (separate_sources)
      launch.template operator()<true, DirectForceOutputMode::Separate>();
    else
      launch.template operator()<true, DirectForceOutputMode::Combined>();
  } else {
    if (separate_sources)
      launch.template operator()<false, DirectForceOutputMode::Separate>();
    else
      launch.template operator()<false, DirectForceOutputMode::Combined>();
  }
  return cudaPeekAtLastError();
}

}  // namespace generativeqc::scf::cuda_execution
