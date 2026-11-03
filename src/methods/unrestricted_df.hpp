#pragma once

#include <algorithm>
#include <cmath>
#include <optional>

#include "methods/method.hpp"
#include "posthf/capacity.hpp"
#include "scf/fock_build.hpp"

namespace generativeqc::methods::detail {

/** Owned auxiliary/reference policy. Correlation-only fitting never changes
 * the UHF stationary problem; full fitting uses the same metric branch in both. */
struct UnrestrictedFitting {
  std::optional<core::System> auxiliary;
  bool full_reference{};
  double threshold{1e-10};
  std::size_t budget{};

  /** Reserve fitted preparation, spectral conditioning and optional coordinate
   * tensors before constructing the reference provider, including one-electron
   * states for which an MP2 occupied/virtual capacity query is not applicable. */
  std::size_t reference_capacity(const core::System& system, unsigned history) const {
    auto result = posthf::uhf_reference_capacity(system, history, true);
    if (!full_reference) return result;
    const auto orbital = molecule::cartesian_ao_count(system);
    const auto auxiliary_count = molecule::cartesian_ao_count(*auxiliary);
    const auto metric = posthf::checked_mul(auxiliary_count, auxiliary_count);
    const auto raw = posthf::checked_mul(posthf::checked_mul(orbital, orbital), auxiliary_count);
    const auto coordinates = posthf::checked_add(1, posthf::checked_mul(3, system.atoms.size()));
    auto fitted = posthf::checked_mul(
        16 * sizeof(double), posthf::checked_mul(coordinates, posthf::checked_add(raw, metric)));
    fitted = posthf::checked_add(fitted, posthf::checked_mul(2, posthf::source_scratch_bytes));
    fitted = posthf::checked_add(fitted, posthf::source_capacity(*auxiliary));
    if (fitted > budget)
      throw std::length_error("unrestricted DF reference exceeds fitting memory budget");
    if (posthf::checked_mul(metric, auxiliary_count) > 100000000000ULL)
      throw std::length_error("unrestricted DF metric exceeds scalar-work bound");
    return posthf::checked_add(result, fitted);
  }

  scf::FockBuildSpec reference_spec() const {
    auto spec = scf::make_hf_fock_spec(scf::FockSpin::Unrestricted);
    if (full_reference) {
      spec.coulomb.approximation = scf::FockApproximation::DensityFitted;
      spec.exchange.approximation = scf::FockApproximation::DensityFitted;
    }
    return spec;
  }
};

inline UnrestrictedFitting unrestricted_fitting(const generativeqc_method_descriptor& descriptor,
                                                const core::System* auxiliary,
                                                std::size_t endpoint_budget) {
  UnrestrictedFitting result;
  const auto mode = descriptor.density_fitting_mode;
  if (mode == GENERATIVEQC_DENSITY_FITTING_NONE) {
    if (auxiliary || descriptor.density_fitting_auxiliary_basis ||
        descriptor.density_fitting_memory_budget_bytes)
      throw std::invalid_argument("unrestricted auxiliary controls require density fitting");
    return result;
  }
  if (!auxiliary) throw std::invalid_argument("unrestricted DF requires an auxiliary basis");
  if (mode != GENERATIVEQC_DENSITY_FITTING_CUDA && mode != GENERATIVEQC_DENSITY_FITTING_AUTO &&
      mode != GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE &&
      mode != GENERATIVEQC_DENSITY_FITTING_FULL_CUDA &&
      mode != GENERATIVEQC_DENSITY_FITTING_CORRELATION_CUDA)
    throw std::invalid_argument("unknown unrestricted density-fitting mode");
  result.auxiliary = *auxiliary;
  result.full_reference = mode == GENERATIVEQC_DENSITY_FITTING_FULL_CUDA;
  result.threshold = descriptor.density_fitting_relative_threshold
                         ? descriptor.density_fitting_relative_threshold
                         : 1e-10;
  if (!std::isfinite(result.threshold) || result.threshold <= 0 || result.threshold >= 1)
    throw std::invalid_argument("invalid unrestricted DF metric threshold");
  result.budget =
      descriptor.density_fitting_memory_budget_bytes
          ? std::min<std::size_t>(descriptor.density_fitting_memory_budget_bytes, endpoint_budget)
          : endpoint_budget;
  return result;
}
}  // namespace generativeqc::methods::detail
