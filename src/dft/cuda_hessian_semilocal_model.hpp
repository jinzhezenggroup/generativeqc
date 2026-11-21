#pragma once

#include "dft/semilocal_family.hpp"
#include "r2scan_cuda_domain.hpp"

namespace generativeqc::dft {

/** Mathematical signature shared by native CUDA Hessian admission and the
 * scalar-ECP fitted-primary SCF provider. Backend, source and state are separate gates. */
constexpr bool cuda_hessian_canonical_semilocal(SemilocalFamily family, double exchange_fraction,
                                                double exchange_scale,
                                                double correlation_scale) noexcept {
  if (correlation_scale != 1) return false;
  if (exchange_fraction == 0)
    return exchange_scale == 1 &&
           (family == SemilocalFamily::Lda || family == SemilocalFamily::Pbe ||
            (kCudaR2scanVariational && family == SemilocalFamily::R2scan));
  const auto canonical_fraction = semilocal_family_cuda_global_hybrid_exact_exchange(family);
  if (exchange_fraction != canonical_fraction) return false;
  // PBE0 rescales the PBE exchange; B3LYP embeds its semilocal fractions.
  return (family == SemilocalFamily::Pbe && exchange_scale == 1 - canonical_fraction) ||
         (family == SemilocalFamily::B3lyp && exchange_scale == 1);
}

}  // namespace generativeqc::dft
