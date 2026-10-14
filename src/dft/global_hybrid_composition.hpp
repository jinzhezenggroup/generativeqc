#pragma once

#include <cstdint>

#include "dft/semilocal_family.hpp"

#if GENERATIVEQC_HAS_CUDA
#include "generated_split_hybrid_registry.cuh"
#endif

namespace generativeqc::dft {

/** The XC/EXX composition is independent of the installed J/K implementation.
 * B3LYP and split-hybrid fractions originate in pinned Libxc method metadata.
 * The curated PBE/PBE0 fraction is a separate, explicit MethodIR contract.
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
      return {fraction, family->component_coefficients_are_native_scales ? 1.0 - fraction : 1.0,
              1.0, !family->component_coefficients_are_native_scales, true};
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

}  // namespace generativeqc::dft
