#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "integrals/s_integrals.hpp"
#include "posthf/capacity.hpp"
#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_derivative_common.hpp"
#include "runtime/df_progress_trace.hpp"

namespace generativeqc::mp2 {

std::vector<double> conventional_derivative_cpu(const core::System& system,
                                                const hf::PhysicalReference& reference,
                                                const LagrangianWeights& weights) {
  runtime::df_progress::Scope trace("mp2_cpu_derivative");
  runtime::df_progress::Scope::number("packaged_available",
                                      integrals::has_packaged_cpu_eri_shell_derivatives());
  std::size_t packaged_quartets = 0, reference_quartets = 0, primitive_evaluations = 0;
  auto derivative = detail::conventional_derivative(
      system, reference, weights,
      [&](std::span<const double> overlap, std::span<const double> hcore) {
        return integrals::contract_weighted_one_electron_derivative(system, overlap, hcore, true);
      },
      [&](const std::array<std::size_t, 4>& shells, std::span<const double> local) {
        // Capability and normalization belong to the integral owner. Keep the
        // single-channel reference wrapper independent of production selection.
        const bool packaged =
            integrals::supports_packaged_cpu_eri_shell_derivatives(system, shells);
        const std::array<std::span<const double>, 1> channels{local};
        std::uint64_t evaluations = 0;
        const auto result = integrals::contract_weighted_eri_shell_derivatives(
                                system, shells, channels, &evaluations,
                                packaged ? integrals::WeightedEriDerivativeExecution::PackagedCpu
                                         : integrals::WeightedEriDerivativeExecution::Reference)
                                .front();
        if (trace.enabled()) {
          auto& quartets = packaged ? packaged_quartets : reference_quartets;
          quartets = posthf::checked_add(quartets, 1);
          primitive_evaluations = posthf::checked_add(primitive_evaluations, evaluations);
        }
        return result;
      });
  runtime::df_progress::Scope::number("packaged_quartets", packaged_quartets);
  runtime::df_progress::Scope::number("reference_quartets", reference_quartets);
  runtime::df_progress::Scope::number("primitive_evaluations", primitive_evaluations);
  return derivative;
}

std::vector<double> density_fitted_derivative_cpu(const core::System& orbital,
                                                  const core::System& auxiliary,
                                                  const DensityFittedLagrangianWeights& weights,
                                                  std::size_t stage_budget) {
  if (!stage_budget) throw std::invalid_argument("RI-MP2 derivative requires a memory budget");
  auto gradient = integrals::contract_weighted_one_electron_derivative(orbital, weights.overlap,
                                                                       weights.one_electron, true);
  auto density_fitting = integrals::contract_weighted_density_fitting_derivative(
      orbital, auxiliary, weights.three_center, weights.metric, stage_budget);
  if (gradient.size() != density_fitting.size())
    throw std::runtime_error("RI-MP2 derivative consumers returned inconsistent dimensions");
  for (std::size_t coordinate = 0; coordinate < gradient.size(); ++coordinate)
    gradient[coordinate] += density_fitting[coordinate];
  if (!std::all_of(gradient.begin(), gradient.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("RI-MP2 derivative contraction produced nonfinite values");
  return gradient;
}

}  // namespace generativeqc::mp2
