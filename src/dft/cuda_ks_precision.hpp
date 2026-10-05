#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "dft/xc_capabilities.hpp"
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
 * also lower density contractions admitted by the point-program capabilities;
 * a nonlocal graph initially lowers Direct J only. Exact exchange (including
 * SR/LR K), tau, XC point algebra/reductions, nonlocal correlation and final
 * audits remain strict FP64.
 */
inline runtime::ExecutionPrecisionSchedule resolve_cuda_ks_precision_schedule(
    std::optional<generativeqc_precision_mode> mode,
    const CudaXcFastPathCapabilities& xc_fast_paths, bool fitted_coulomb,
    bool nonlocal_correlation) {
  const bool automatic = mode && *mode == GENERATIVEQC_PRECISION_AUTO;
  if (mode && *mode != GENERATIVEQC_PRECISION_FP64 && !automatic)
    throw std::invalid_argument("CUDA KS received an unknown precision mode");
  if (automatic && fitted_coulomb)
    throw std::invalid_argument("CUDA fitted KS requires strict FP64");
  const bool mixed_density = automatic && !nonlocal_correlation &&
                             cuda_xc_capability_qualified(xc_fast_paths.mixed_density_precision);

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

/** Resolve the arithmetic actually executable by one physical iteration.
 * Schedule policy says what may be lowered; the consuming layout says what can
 * execute that lowering. Keep the shared schedule, including qualification and
 * audit metadata, through capability intersection. Strict refinement restores
 * every region to FP64, including regions added by future compositions. */
inline runtime::ExecutionPrecisionSchedule resolve_cuda_ks_iteration_precision(
    const runtime::ExecutionPrecisionSchedule& schedule, bool strict_refinement,
    bool mixed_density_contraction_capability) {
  return schedule.filter_lower_precision([=](const runtime::PrecisionRegion& region) {
    return !strict_refinement && (region.name != cuda_ks_precision_region::kDensityContraction ||
                                  mixed_density_contraction_capability);
  });
}

}  // namespace generativeqc::dft
