#pragma once

#include "scf/cuda/direct_force_low_order_sources.cuh"

namespace generativeqc::scf::cuda_execution {

/** Resolve shell ownership once for compact, paged and resident force queues.
 * Generated consumers own masked classes; task.tile remains the bounded AO
 * tiling identity and is checked by the precontracted task itself. */
__device__ inline unsigned direct_force_task_shell_class(const DeviceBatch& batch,
                                                       ActiveShellQuartetTile task) {
  return direct_quartet_shell_class_device(
      batch.shell_angular[batch.shell_pair_first[task.first_pair]],
      batch.shell_angular[batch.shell_pair_second[task.first_pair]],
      batch.shell_angular[batch.shell_pair_first[task.second_pair]],
      batch.shell_angular[batch.shell_pair_second[task.second_pair]]);
}

/** Execute a known class through the common coefficient/primitive consumer.
 * The resident view is optional and must contain the canonical bra in cache
 * order. Separate J'/K' outputs share geometry, never a second HF traversal. */
template <bool Unrestricted, DirectForceOutputMode Mode, unsigned ShellClass,
          bool ResidentBra = false>
__device__ inline void contract_direct_force_class_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask, double coulomb_coefficient,
    double exchange_coefficient, const PrimitivePairData* resident_first_pairs = nullptr,
    std::int64_t resident_first_pair_count = 0) {
  if ((generated_shell_class_mask & (std::uint64_t{1} << ShellClass)) != 0U ||
      direct_force_task_shell_class(batch, task) != ShellClass)
    return;
  contract_two_electron_force_low_order_sources_task<
      Unrestricted, ShellClass, false, LowOrderSourceRoots<ShellClass>, Mode, ResidentBra>(
      batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
      coulomb_coefficient, exchange_coefficient, 0.0, resident_first_pairs,
      resident_first_pair_count);
}

/** Shared precontracted execution for all qualified order-zero-through-three
 * classes. Queue policy belongs to the caller; source weights, cached-pair
 * traversal, Boys work and translational scatter have one execution owner. */
template <bool Unrestricted, DirectForceOutputMode Mode>
__device__ inline void contract_direct_force_precontracted_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask, double coulomb_coefficient,
    double exchange_coefficient) {
  const unsigned shell_class = direct_force_task_shell_class(batch, task);
  if (shell_class < 64U &&
      (generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U)
    return;
  contract_two_electron_force_low_order_sources<Unrestricted, false, Mode>(
      shell_class, batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
      coulomb_coefficient, exchange_coefficient);
}

}  // namespace generativeqc::scf::cuda_execution
