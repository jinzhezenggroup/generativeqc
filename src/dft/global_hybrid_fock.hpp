#pragma once

#include "dft/global_hybrid_composition.hpp"
#include "scf/fock_build.hpp"

namespace generativeqc::dft {

/** Common exact full-range-exchange composition check, before DF selection. */
constexpr bool global_hybrid_coefficients_match(const scf::FockBuildSpec& spec,
                                                std::uint32_t functional, std::uint32_t spins,
                                                double semilocal_exchange_scale,
                                                double semilocal_correlation_scale) noexcept {
  const auto composition = global_hybrid_composition(functional);
  if (!composition.present || spins < 1U || spins > 2U ||
      spec.spin != (spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted) ||
      spec.derivative_order != 0 || !spec.coulomb.present || spec.coulomb.coefficient != 1.0 ||
      spec.coulomb.op != scf::FockOperator::FullRange || spec.coulomb.omega != 0.0 ||
      !spec.exchange.present || spec.exchange.op != scf::FockOperator::FullRange ||
      spec.exchange.omega != 0.0 ||
      semilocal_exchange_scale != composition.semilocal_exchange_scale ||
      semilocal_correlation_scale != composition.semilocal_correlation_scale)
    return false;
  return spec.exchange.coefficient == -composition.exact_exchange / (spins == 1 ? 2.0 : 1.0);
}

/** One J/K identity gate for curated GGA and generated meta-GGA hybrids.
 * Reject mixed Direct/DF, COSX, and split DF lacking explicit provider coverage.
 * The calling method owner independently checks range/nonlocal corrections.
 */
constexpr bool cuda_global_hybrid_jk_match(const scf::FockBuildSpec& spec, std::uint32_t functional,
                                           std::uint32_t spins, double semilocal_exchange_scale,
                                           double semilocal_correlation_scale) noexcept {
  if (!global_hybrid_coefficients_match(spec, functional, spins, semilocal_exchange_scale,
                                        semilocal_correlation_scale))
    return false;
  const auto j = spec.coulomb.approximation;
  const auto k = spec.exchange.approximation;
  return (j == scf::FockApproximation::Exact && k == scf::FockApproximation::Exact) ||
         (j == scf::FockApproximation::DensityFitted &&
          k == scf::FockApproximation::DensityFitted && cuda_global_hybrid_df_admitted(functional));
}

}  // namespace generativeqc::dft
