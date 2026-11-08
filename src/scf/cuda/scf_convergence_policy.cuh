#pragma once

#include <math_constants.h>

#include <cmath>
#include <cstddef>

#include "scf/convergence_limits.hpp"
#include "scf/cuda/scf_constants.hpp"
#include "scf/cuda/scf_incremental_policy.cuh"

namespace generativeqc::scf::cuda_execution {

/** Common FP64 HF energy comparison budget. Both quartet scatter and fitted
 * contractions/reductions lose a few representable values at stationarity.
 * This guard scales with energy, never with the requested density tolerance;
 * density and current physical-Fock stationarity must still pass separately.
 * An absent or nonfinite baseline cannot manufacture first-step convergence. */
__device__ inline double hf_energy_roundoff_guard(bool enabled, double energy,
                                                  double previous_energy) {
  if (!enabled || !isfinite(energy) || !isfinite(previous_energy)) return 0.0;
  const double scale = fmax(1.0, fmax(fabs(energy), fabs(previous_energy)));
  return kHfEnergyRoundoffFactor * kDoubleMachineEpsilon * scale;
}

/** Shared direct/DF stopping rule, independent of how a solver obtained its
 * baseline. The reported energy change remains the unmodified absolute delta.
 * A trusted warm baseline permits one step; a missing baseline requires a
 * second physical energy evaluation. NaNs/infinities fail every route closed. */
__device__ inline bool hf_iteration_converged(double energy, double previous_energy,
                                              double energy_tolerance, double density_tolerance,
                                              double density_rms, double physical_maximum,
                                              bool guard_roundoff = true,
                                              bool qualified_owner = false) {
  return isfinite(energy) && isfinite(previous_energy) && isfinite(density_rms) &&
         isfinite(physical_maximum) &&
         fabs(energy - previous_energy) <
             energy_tolerance + hf_energy_roundoff_guard(guard_roundoff, energy, previous_energy) &&
         density_rms < density_tolerance &&
         physical_maximum <=
             hf_stationarity_tolerance(energy_tolerance, density_tolerance, qualified_owner);
}

enum class HfEnergyRefinementTransition { none, retain_full_baseline, discard_delta_baseline };

/** Retire deltas within one decade of the physical gates, without changing
 * acceptance. Starting full-only refinement before stationarity lets the two
 * fresh full energies overlap the ordinary SCF tail rather than extend it.
 * This phase-selection margin never enters hf_iteration_converged.
 * A current full evaluation can establish the first full-energy baseline; a
 * delta evaluation cannot. After actual deltas, both transitions block current
 * convergence and require consecutive full energies. A stationary no-delta
 * solve keeps ordinary convergence but latches full-only if another step is
 * needed, avoiding a pointless delta/full transition on an energy-only tail.
 * Null state leaves ordinary callers alone. */
__device__ inline HfEnergyRefinementTransition begin_hf_full_energy_refinement(
    std::int32_t system, double energy, double energy_tolerance, double density_tolerance,
    double density_rms, double physical_maximum, bool physical_target, bool qualified_owner,
    std::uint32_t* delta_updates, const unsigned long long* delta_build_count) {
  constexpr double refinement_margin = 10.0;
  if (delta_updates == nullptr || delta_build_count == nullptr || !physical_target ||
      delta_updates[system] == kIncrementalDirectJkFullEnergyRefinement || !isfinite(energy) ||
      !isfinite(density_rms) || !isfinite(physical_maximum) ||
      density_rms / refinement_margin >= density_tolerance ||
      physical_maximum >
          refinement_margin *
              hf_stationarity_tolerance(energy_tolerance, density_tolerance, qualified_owner))
    return HfEnergyRefinementTransition::none;
  const bool current_full_build = delta_updates[system] == 0U;
  delta_updates[system] = kIncrementalDirectJkFullEnergyRefinement;
  if (delta_build_count[system] == 0ULL) return HfEnergyRefinementTransition::none;
  return current_full_build ? HfEnergyRefinementTransition::retain_full_baseline
                            : HfEnergyRefinementTransition::discard_delta_baseline;
}

/** One-warp maximum of the physical pre-DIIS FDS-SDF residual. Null is reserved
 * for compatibility callers without residual storage and coarse mixed stages;
 * production FP64 HF supplies the current residual. UHF passes both spins. */
__device__ inline double maximum_physical_residual(const double* values, std::size_t size) {
  if (values == nullptr) return 0.0;
  double maximum = 0.0;
  for (std::size_t element = threadIdx.x; element < size; element += blockDim.x)
    maximum = isfinite(values[element]) ? fmax(maximum, fabs(values[element])) : CUDART_INF;
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1)
    maximum = fmax(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
  return __shfl_sync(0xffffffffU, maximum, 0);
}

}  // namespace generativeqc::scf::cuda_execution
