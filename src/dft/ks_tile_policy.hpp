#pragma once

#include <cstddef>

#include "runtime/cuda_fp64_matrix_policy.hpp"

namespace generativeqc::dft {

/** Prefer larger Direct batches without treating device identity as a cost model.
 * This is a portable scheduling preference, not a universal speedup claim.
 * Keep the established full-range, strict-FP64, all-electron RKS workload and
 * resource envelope; the two owners retain explicit controls and bounded retries.
 */
constexpr std::size_t automatic_direct_ks_tile(std::size_t requested, bool full_range_rks_fp64,
                                               bool strict_budget, bool unconstrained,
                                               bool all_electron, std::size_t maximum_angular,
                                               std::size_t atoms, std::size_t nao,
                                               std::size_t points) noexcept {
  if (requested) return requested;
  return full_range_rks_fp64 && strict_budget && unconstrained && all_electron &&
                 maximum_angular <= 2 && atoms >= 48 && atoms <= 128 && nao >= 512 &&
                 points >= (1U << 19)
             ? 1024
             : 256;
}

/** Resolve the public automatic SCF tile without changing an explicit request.
 * The larger tiles amortize launch and mapped FP64 contraction overhead for
 * qualified fitted RKS grids. Direct XC uses its independent portable policy;
 * nonlocal work has a different fitted optimum.
 * CPU, small, mixed, unrestricted, constrained, and unknown device domains
 * retain 256. Force tiling is independently resolved by the enclosing owner.
 */
constexpr std::size_t automatic_ks_scf_tile(std::size_t requested, bool qualified_rks_fp64,
                                            bool unconstrained, std::size_t nao, std::size_t points,
                                            bool nonlocal, int compute_major,
                                            int fp32_fp64_ratio) noexcept {
  if (requested) return requested;
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
