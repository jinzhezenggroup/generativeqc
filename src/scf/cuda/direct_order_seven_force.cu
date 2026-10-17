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
template <bool Unrestricted, DirectForceOutputMode Mode,
          unsigned PageSize = kDirectOrderSevenKetPageSize, bool DensityContraction = false>
__global__ __launch_bounds__(kBoundedDirectThreads, 1) void direct_order_seven_force_kernel(
    DeviceBatch batch, const GeneratedShellPairStream* topology_pointer, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient,
    double exchange_coefficient) {
  extern __shared__ __align__(16) unsigned char cooperative_workspace[];
  static_assert(PageSize == 8 || PageSize == 16 || PageSize == 32);
  __shared__ ActiveShellQuartetTile queue[PageSize];
  __shared__ std::uint32_t queue_count;
  __shared__ DirectOrderSevenPairPage page;
  __shared__ bool page_available;
  DirectOrderSevenPagePosition position{};
  const auto& topology = *topology_pointer;
  while (true) {
    // Inactive and empty claims also finish reading the old shared descriptor
    // before the leader publishes another page.
    __syncthreads();
    if (threadIdx.x == 0U) {
      const auto claim = atomicAdd(cursor, 1ULL);
      page_available = direct_order_seven_pair_page(topology.pair_class_offsets, batch.batch_size,
                                                    claim, position, page, PageSize);
      queue_count = 0U;
    }
    __syncthreads();
    if (!page_available) return;
    auto* traversal = batch.direct_force_work ? &batch.direct_force_work->traversal[1] : nullptr;
    if (traversal && threadIdx.x == 0U) atomicAdd(&traversal->pages, 1ULL);
    if (active != nullptr && active[page.system] == 0U) {
      if (traversal && threadIdx.x == 0U) atomicAdd(&traversal->inactive_pages, 1ULL);
      continue;
    }

    if (threadIdx.x < page.ket_end - page.ket_begin) {
      const auto bra = topology.pair_order[page.bra];
      const auto ket = topology.pair_order[page.ket_begin + threadIdx.x];
      // Some CUDA-compatible providers expose signed-only max/min overloads.
      // Preserve the full unsigned physical-ID range before canonicalization.
      const auto [first_pair, second_pair] = canonical_direct_shell_pairs(bra, ket);
      const ActiveShellQuartetTile task{first_pair, second_pair, 0U};
      auto* census = direct_force_task_work(batch, task);
      if (traversal) atomicAdd(&traversal->candidates, 1ULL);
      if (census) atomicAdd(&census->enumerated, 1ULL);
      if (direct_shell_quartet_survives_screening<Unrestricted, DirectScreeningPurpose::Force>(
              batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds, nullptr, false, exchange_coefficient == 0.0)) {
        const auto shell_class = direct_quartet_shell_class_device(
            batch.shell_angular[batch.shell_pair_first[first_pair]],
            batch.shell_angular[batch.shell_pair_second[first_pair]],
            batch.shell_angular[batch.shell_pair_first[second_pair]],
            batch.shell_angular[batch.shell_pair_second[second_pair]]);
        const bool generated_class =
            generated_overflow[shell_class] == 0U &&
            bounded_generated_class_enabled(shell_class, enabled_mask_pointer, enabled_mask);
        if (census) atomicAdd(generated_class ? &census->generated_owned : &census->admitted, 1ULL);
        if (!generated_class) {
          const auto slot = atomicAdd(&queue_count, 1U);
          queue[slot] = task;
          profile_bounded_direct_shell_quartet(batch, task, profile);
        }
      } else if (census) {
        atomicAdd(&census->shell_screened, 1ULL);
      }
    }
    __syncthreads();
    for (std::uint32_t slot = 0; slot < queue_count; ++slot) {
      contract_shared_direct_pair_force<Unrestricted, Mode, DensityContraction>(
          batch, queue[slot], screening_tolerance, schwarz_bounds, density, active, output,
          coulomb_coefficient, exchange_coefficient, cooperative_workspace);
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
    bool separate_sources, bool density_contraction) {
  density_contraction =
      density_contraction && exchange_coefficient == 0.0 && !batch.direct_force_seed;
  auto launch = [&]<bool Unrestricted, DirectForceOutputMode Mode, unsigned PageSize,
                    bool DensityContraction>() {
    const auto workspace_bytes = std::max(
        shared_bytes, DensityContraction ? sizeof(CooperativeDensityForceRecurrence)
                                         : sizeof(CooperativeDirectPairDerivativeRecurrence));
    direct_order_seven_force_kernel<Unrestricted, Mode, PageSize, DensityContraction>
        <<<grid, kBoundedDirectThreads, workspace_bytes, stream>>>(
            batch, topology, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            enabled_mask_pointer, enabled_mask, generated_overflow, schwarz_bounds, density, active,
            output, cursor, profile, coulomb_coefficient, exchange_coefficient);
  };
  auto pages = [&]<bool Unrestricted, DirectForceOutputMode Mode, bool DensityContraction>() {
    // Only acquisition width changes. Each queue entry keeps the existing
    // shared recurrence and uniform publish/retire barriers.
    if (batch.direct_order_seven_page_size == 8U)
      launch.template operator()<Unrestricted, Mode, 8U, DensityContraction>();
    else if (batch.direct_order_seven_page_size == 16U)
      launch.template operator()<Unrestricted, Mode, 16U, DensityContraction>();
    else
      launch.template operator()<Unrestricted, Mode, 32U, DensityContraction>();
  };
  auto contraction = [&]<bool Unrestricted, DirectForceOutputMode Mode>() {
    if (density_contraction)
      pages.template operator()<Unrestricted, Mode, true>();
    else
      pages.template operator()<Unrestricted, Mode, false>();
  };
  if (unrestricted) {
    if (separate_sources)
      contraction.template operator()<true, DirectForceOutputMode::Separate>();
    else
      contraction.template operator()<true, DirectForceOutputMode::Combined>();
  } else {
    if (separate_sources)
      contraction.template operator()<false, DirectForceOutputMode::Separate>();
    else
      contraction.template operator()<false, DirectForceOutputMode::Combined>();
  }
  return cudaPeekAtLastError();
}

}  // namespace generativeqc::scf::cuda_execution
