#pragma once

#include <algorithm>
#include <cstddef>

#include "core/types.hpp"
#include "dft/grid.hpp"
#include "dft/split_hybrid_cuda_domain.hpp"
#include "generated_split_hybrid_registry.cuh"
#include "scf/fock_build.hpp"
#include "scf/types.hpp"

namespace generativeqc::dft {

/** Map a public functional to its independently opted-in stationary program. */
inline constexpr std::uint32_t cuda_split_hybrid_variational_domain(
    std::uint32_t functional) noexcept {
  if (kCudaM062XVariational && functional == generated::kM062XFunctionalCode)
    return kCudaM062XVariationalDomain;
  if (kCudaMN15Variational && functional == generated::kMN15FunctionalCode)
    return kCudaMN15VariationalDomain;
  return 0;
}

/** Authenticate the physical point model and its exact-exchange composition.
 * This predicate also validates detached SCF identities before borrowing data. */
inline bool cuda_split_hybrid_variational_model(std::uint32_t functional,
                                                const scf::ResolvedFockBuild& fock,
                                                std::uint32_t spins, const GridSpec& grid,
                                                double exchange_scale,
                                                double correlation_scale) noexcept {
  if (!cuda_split_hybrid_variational_domain(functional)) return false;
  const auto& spec = fock.spec;
  const auto hybrid = generated::split_hybrid_composition(functional);
  const auto exact_exchange =
      static_cast<double>(hybrid.exact_exchange_numerator) / hybrid.exact_exchange_denominator;
  const auto approximation = spec.coulomb.approximation;
  const auto full_range = [approximation](const scf::FockTermSpec& term) {
    return term.present && term.op == scf::FockOperator::FullRange && term.omega == 0.0 &&
           term.approximation == approximation;
  };
  return fock.backend == scf::FockBackend::Cuda && fock.precision == scf::FockPrecision::Float64 &&
         grid.partition_iterations == 3 && exchange_scale == 1.0 && correlation_scale == 1.0 &&
         (spins == 1 || spins == 2) && spec.version == 1 && spec.derivative_order == 0 &&
         spec.spin == (spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted) &&
         (approximation == scf::FockApproximation::Exact ||
          (approximation == scf::FockApproximation::DensityFitted &&
           generated::split_hybrid_cuda_df_energy_force(functional))) &&
         full_range(spec.coulomb) && spec.coulomb.coefficient == 1.0 && full_range(spec.exchange) &&
         spec.exchange.coefficient == -exact_exchange / (spins == 1 ? 2.0 : 1.0);
}

/** Check the stationary ECP shape for an orbital or a separate fitting space. */
inline bool cuda_split_hybrid_ecp_angular_domain(const core::System& system,
                                                 unsigned spherical_limit) noexcept {
  const auto limit = system.basis_representation == core::kSphericalBasis ? spherical_limit : 1U;
  return std::all_of(system.shells.begin(), system.shells.end(),
                     [limit](const auto& shell) { return shell.angular_momentum <= limit; });
}

/** Bind SCF and native derivatives to one model, regardless of requested properties.
 * Lower precision, noncanonical composition, and unqualified basis shapes keep
 * their incumbent split-hybrid point; they cannot mint this Hessian identity.
 * The caller checks basis representation and admits each derivative separately. */
inline bool use_cuda_split_hybrid_variational(std::uint32_t functional,
                                              const scf::ScfOptions& options,
                                              const scf::ResolvedFockBuild& fock,
                                              std::uint32_t spins, const GridSpec& grid,
                                              std::size_t maximum_angular) noexcept {
  return maximum_angular <= 3U &&
         options.precision_mode.value_or(GENERATIVEQC_PRECISION_FP64) ==
             GENERATIVEQC_PRECISION_FP64 &&
         options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused &&
         std::all_of(options.application_precision_budget.begin(),
                     options.application_precision_budget.end(),
                     [](double budget) { return budget == 0.0; }) &&
         cuda_split_hybrid_variational_model(functional, fock, spins, grid,
                                             options.semilocal_exchange_scale,
                                             options.semilocal_correlation_scale);
}

}  // namespace generativeqc::dft
