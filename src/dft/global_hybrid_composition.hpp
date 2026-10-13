#pragma once

#include <cstdint>

#include "dft/semilocal_family.hpp"
#include "scf/fock_build.hpp"

#if GENERATIVEQC_HAS_CUDA
#include "generated_split_hybrid_registry.cuh"
#endif

namespace generativeqc::dft {

/** The XC/EXX composition is independent of the installed J/K implementation.
 * Curated exact coefficients and split-global-hybrid coefficients are generated
 * from the pinned Libxc records. PBE's native scales also describe PBE0.
 */
struct GlobalHybridComposition {
  double exact_exchange{};
  double semilocal_exchange_scale{1.0};
  double semilocal_correlation_scale{1.0};
  bool fixed_semilocal{};
  bool present{};
};

constexpr GlobalHybridComposition global_hybrid_composition(std::uint32_t functional) noexcept {
  if (const auto* family = semilocal_family_metadata_from_code(functional)) {
    const double fraction = family->cuda_global_hybrid_exact_exchange;
    if (fraction > 0.0)
      return {fraction,
              family->component_coefficients_are_native_scales ? 1.0 - fraction : 1.0,
              1.0,
              !family->component_coefficients_are_native_scales,
              true};
  }
#if GENERATIVEQC_HAS_CUDA
  const auto split = generated::split_hybrid_composition(functional);
  if (split.matched && split.exact_exchange_denominator)
    return {static_cast<double>(split.exact_exchange_numerator) /
                static_cast<double>(split.exact_exchange_denominator),
            1.0, 1.0, true, true};
#endif
  return {};
}

/** A fitted J/K implementation and complete derivatives are not Libxc traits:
 * keep their independent runtime/manifest admission, even for valid Libxc XC.
 */
constexpr bool cuda_global_hybrid_df_admitted(std::uint32_t functional) noexcept {
  if (const auto* family = semilocal_family_metadata_from_code(functional))
    return family->cuda_ks && family->cuda_global_hybrid_exact_exchange > 0.0;
#if GENERATIVEQC_HAS_CUDA
  return generated::split_hybrid_cuda_df_energy_force(functional);
#else
  return false;
#endif
}

/** Common exact full-range-exchange composition check, before DF selection. */
constexpr bool global_hybrid_coefficients_match(
    const scf::FockBuildSpec& spec, std::uint32_t functional, std::uint32_t spins,
    double semilocal_exchange_scale, double semilocal_correlation_scale) noexcept {
  const auto composition = global_hybrid_composition(functional);
  if (!composition.present || spins < 1U || spins > 2U ||
      spec.spin != (spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted) ||
      spec.derivative_order != 0 || !spec.coulomb.present ||
      spec.coulomb.coefficient != 1.0 || spec.coulomb.op != scf::FockOperator::FullRange ||
      spec.coulomb.omega != 0.0 || !spec.exchange.present ||
      spec.exchange.op != scf::FockOperator::FullRange || spec.exchange.omega != 0.0 ||
      semilocal_exchange_scale != composition.semilocal_exchange_scale ||
      semilocal_correlation_scale != composition.semilocal_correlation_scale)
    return false;
  return spec.exchange.coefficient ==
         -composition.exact_exchange / (spins == 1 ? 2.0 : 1.0);
}

/** One J/K identity gate for curated GGA and generated meta-GGA hybrids.
 * Reject mixed Direct/DF, COSX, and split DF lacking explicit provider coverage.
 * The calling method owner independently checks range/nonlocal corrections.
 */
constexpr bool cuda_global_hybrid_jk_match(
    const scf::FockBuildSpec& spec, std::uint32_t functional, std::uint32_t spins,
    double semilocal_exchange_scale, double semilocal_correlation_scale) noexcept {
  if (!global_hybrid_coefficients_match(spec, functional, spins,
                                        semilocal_exchange_scale, semilocal_correlation_scale))
    return false;
  const auto j = spec.coulomb.approximation;
  const auto k = spec.exchange.approximation;
  return (j == scf::FockApproximation::Exact && k == scf::FockApproximation::Exact) ||
         (j == scf::FockApproximation::DensityFitted &&
          k == scf::FockApproximation::DensityFitted &&
          cuda_global_hybrid_df_admitted(functional));
}

}  // namespace generativeqc::dft
