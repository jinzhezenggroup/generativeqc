#pragma once

#include <cstddef>

#include "runtime/cuda_fp64_matrix_policy.hpp"

namespace generativeqc::dft {

/** Resolve the public automatic SCF tile without changing an explicit request.
 * The larger tiles amortize launch and mapped FP64 contraction overhead for
 * large fitted RKS grids. Nonlocal work has a different measured optimum.
 * CPU, small, mixed, unrestricted, constrained, and unknown device domains
 * retain 256. Force tiling is independently resolved by the enclosing owner.
 */
constexpr std::size_t automatic_ks_scf_tile(std::size_t requested, bool fitted_rks_fp64,
                                            bool unconstrained, std::size_t nao, std::size_t points,
                                            bool nonlocal, int compute_major,
                                            int fp32_fp64_ratio) noexcept {
  if (requested) return requested;
  if (!fitted_rks_fp64 || !unconstrained || nao < 512 || points < (1U << 19) ||
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
constexpr std::size_t automatic_ks_force_tile(bool fitted_rks_fp64, bool unconstrained,
                                              bool all_electron, std::size_t maximum_angular,
                                              std::size_t atoms, std::size_t nao,
                                              std::size_t points, int compute_major) noexcept {
  return fitted_rks_fp64 && unconstrained && all_electron && maximum_angular <= 2 && atoms >= 48 &&
                 atoms <= 128 && nao >= 512 && points >= (1U << 19) && compute_major >= 8
             ? 1024
             : 256;
}

}  // namespace generativeqc::dft
