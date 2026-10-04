#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "dft/semilocal_family.hpp"
#include "generativeqc/generativeqc.h"
#include "runtime/execution_precision.hpp"

namespace generativeqc::dft {

namespace cuda_ks_precision_region {
inline constexpr std::string_view kCoulombJ = "dft.coulomb_j";
inline constexpr std::string_view kDensityContraction = "dft.density_contraction";
inline constexpr std::string_view kExactExchange = "dft.exact_exchange";
inline constexpr std::string_view kTau = "dft.tau";
inline constexpr std::string_view kXcPointAlgebra = "dft.xc_point_algebra";
inline constexpr std::string_view kNonlocalCorrelation = "dft.nonlocal_correlation";
inline constexpr std::string_view kFinalAudit = "dft.final_audit";
}  // namespace cuda_ks_precision_region

/** Resolve method policy into the compiler-common native precision contract.
 *
 * AUTO lowers only independently qualified components. Ordinary local KS may
 * also lower qualified LDA/PBE/r2SCAN density contractions; a nonlocal graph
 * initially lowers Direct J only. Exact exchange (including SR/LR K), tau,
 * XC point algebra/reductions, nonlocal correlation and final audits remain
 * strict FP64.
 */
inline runtime::ExecutionPrecisionSchedule resolve_cuda_ks_precision_schedule(
    std::optional<generativeqc_precision_mode> mode, std::uint32_t functional, bool fitted_coulomb,
    bool nonlocal_correlation) {
  const bool automatic = mode && *mode == GENERATIVEQC_PRECISION_AUTO;
  if (mode && *mode != GENERATIVEQC_PRECISION_FP64 && !automatic)
    throw std::invalid_argument("CUDA KS received an unknown precision mode");
  if (automatic && fitted_coulomb)
    throw std::invalid_argument("CUDA fitted KS requires strict FP64");
  const bool mixed_density =
      automatic && !nonlocal_correlation &&
      (functional == semilocal_family_code(SemilocalFamily::Lda) ||
                    functional == semilocal_family_code(SemilocalFamily::Pbe) ||
                    functional == semilocal_family_code(SemilocalFamily::R2scan));

  runtime::ExecutionPrecisionSchedule schedule;
  schedule.add_region(cuda_ks_precision_region::kCoulombJ,
                      automatic
                          ? runtime::fp32_compute_fp64_accumulation("dft.cuda.auto/coulomb-j-v1")
                          : runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kDensityContraction,
                      mixed_density ? runtime::fp32_compute_fp64_accumulation(
                                          "dft.cuda.auto/density-contraction-v1")
                                    : runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kExactExchange, runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kTau, runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kXcPointAlgebra, runtime::strict_fp64_precision());
  if (nonlocal_correlation)
    schedule.add_region(cuda_ks_precision_region::kNonlocalCorrelation,
                        runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kFinalAudit, runtime::strict_fp64_precision());
  return schedule;
}

}  // namespace generativeqc::dft
