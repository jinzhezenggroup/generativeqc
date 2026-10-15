#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "dft/xc_capabilities.hpp"
#include "generativeqc/generativeqc.h"
#include "runtime/execution_precision.hpp"
#include "scf/fock_precision.hpp"

namespace generativeqc::dft {

namespace cuda_ks_precision_region {
inline constexpr std::string_view kCoulombJ = "dft.coulomb_j";
inline constexpr std::string_view kDensityContraction = "dft.density_contraction";
inline constexpr std::string_view kExactExchange = "dft.exact_exchange";
inline constexpr std::string_view kRangeExchange = "dft.range_exchange";
inline constexpr std::string_view kTau = "dft.tau";
inline constexpr std::string_view kXcPointAlgebra = "dft.xc_point_algebra";
inline constexpr std::string_view kXcMatrixContractions = "dft.xc_matrix_contractions";
inline constexpr std::string_view kNonlocalCorrelation = "dft.nonlocal_correlation";
inline constexpr std::string_view kFinalAudit = "dft.final_audit";
}  // namespace cuda_ks_precision_region

/** Resolve method policy into the compiler-common native precision contract.
 *
 * AUTO lowers only independently qualified components. Ordinary local KS may
 * also lower density contractions admitted by the point-program capabilities;
 * a nonlocal graph initially lowers Direct J only. Exact exchange (including
 * SR/LR K), tau, XC point algebra/reductions, nonlocal correlation and final
 * audits remain strict FP64. Budget-Fock admission additionally permits full-range
 * K to share J's FP32 recurrence, with FP64 contraction and final physical Focks.
 * The separate application-budget XC warmup keeps
 * J/K strict and instead admits PBE correlation points, AO production and tiled
 * density/potential products with FP32 inner accumulation. Its strict phase
 * restores the entire schedule; it does not narrow analytic gradients.
 */
inline runtime::ExecutionPrecisionSchedule resolve_cuda_ks_precision_schedule(
    std::optional<generativeqc_precision_mode> mode,
    const CudaXcFastPathCapabilities& xc_fast_paths, bool fitted_coulomb, bool nonlocal_correlation,
    bool experimental_point = false, bool experimental_contractions = false,
    scf::CudaFockPrecisionCapabilities budget_fock = {}, bool range_exchange = false) {
  const bool automatic = mode && *mode == GENERATIVEQC_PRECISION_AUTO;
  if (mode && *mode != GENERATIVEQC_PRECISION_FP64 && !automatic)
    throw std::invalid_argument("CUDA KS received an unknown precision mode");
  if (automatic && fitted_coulomb)
    throw std::invalid_argument("CUDA fitted KS requires strict FP64");
  if (experimental_point && (!automatic || nonlocal_correlation))
    throw std::invalid_argument("experimental point precision requires local CUDA AUTO KS");
  if (experimental_contractions && !experimental_point)
    throw std::invalid_argument("mixed XC contractions require the admitted point warmup");
  const bool budget_requested = budget_fock.any_lower_precision();
  if (budget_requested &&
      (!automatic || fitted_coulomb || experimental_point || !budget_fock.coulomb))
    throw std::invalid_argument("budget Fock requires a supported CUDA AUTO Direct consumer");
  const bool mixed_density = automatic && !budget_requested && !experimental_point &&
                             !nonlocal_correlation &&
                             cuda_xc_capability_qualified(xc_fast_paths.mixed_density_precision);

  runtime::ExecutionPrecisionSchedule schedule;
  schedule.add_region(cuda_ks_precision_region::kCoulombJ,
                      automatic && !experimental_point
                          ? runtime::fp32_compute_fp64_accumulation("dft.cuda.auto/coulomb-j-v1")
                          : runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kDensityContraction,
                      mixed_density ? runtime::fp32_compute_fp64_accumulation(
                                          "dft.cuda.auto/density-contraction-v1")
                                    : runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kExactExchange,
                      budget_fock.exchange ? runtime::fp32_compute_fp64_accumulation(
                                                 "direct-fock-budget-v2/full-range-k")
                                           : runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kTau, runtime::strict_fp64_precision());
  if (range_exchange)
    schedule.add_region(cuda_ks_precision_region::kRangeExchange, runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kXcPointAlgebra,
                      experimental_point ? runtime::fp32_compute_fp64_accumulation(
                                               "experimental/pbe-correlation-point-v1")
                                         : runtime::strict_fp64_precision());
  schedule.add_region(cuda_ks_precision_region::kXcMatrixContractions,
                      experimental_contractions
                          ? runtime::PrecisionDirective{runtime::PrecisionDtype::Fp64,
                                                        runtime::PrecisionDtype::Fp32,
                                                        runtime::PrecisionDtype::Fp32,
                                                        "pbe-xc-warmup-v1/tiled-contractions"}
                          : runtime::strict_fp64_precision());
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
