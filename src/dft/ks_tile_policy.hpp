#pragma once

#include <cstddef>

#include "runtime/cuda_fp64_matrix_policy.hpp"

namespace generativeqc::dft {

/** Resolve the public automatic CUDA SCF tile without changing an explicit request.
 * Non-Direct-RKS semilocal XC retains the general 1024-point CUDA default.
 * Direct RKS stays at 256 without exact-workload performance qualification;
 * hardware generation alone cannot supply that qualification.
 * Numeric resource promises retain their quoted 256-point owner.
 * Nonlocal work retains its separately measured fitted/full-rate-FP64 policy;
 * CPU callers do not consult this CUDA policy.
 */
constexpr std::size_t automatic_ks_scf_tile(std::size_t requested, bool qualified_rks_fp64,
                                            bool unconstrained, std::size_t nao, std::size_t points,
                                            bool nonlocal, int compute_major, int fp32_fp64_ratio,
                                            bool direct_rks) noexcept {
  if (requested) return requested;
  if (direct_rks) return 256;
  if (unconstrained && !nonlocal) return 1024;
  if (!qualified_rks_fp64 || !unconstrained || nao < 512 || points < (1U << 19) ||
      !runtime::cuda_fp64_matrix_acceleration_supported(compute_major, fp32_fp64_ratio))
    return 256;
  return nonlocal ? 512 : 1024;
}

/** Force batching amortizes grid launches even without FP64 matrix hardware.
 * Keep the measured large, all-electron, through-d RKS domain separate from
 * SCF admission. Above 128 atoms the geometry scratch schedule changes; small
 * grids, old/unknown devices and resource promises retain the bounded owner.
 * The caller preserves explicit SCF inheritance before consulting this policy.
 */
constexpr std::size_t automatic_ks_force_tile(bool qualified_rks_fp64, bool unconstrained,
                                              bool all_electron, std::size_t maximum_angular,
                                              std::size_t atoms, std::size_t nao,
                                              std::size_t points, int compute_major) noexcept {
  return qualified_rks_fp64 && unconstrained && all_electron && maximum_angular <= 2 &&
                 atoms >= 48 && atoms <= 128 && nao >= 512 && points >= (1U << 19) &&
                 compute_major >= 8
             ? 1024
             : 256;
}

}  // namespace generativeqc::dft
