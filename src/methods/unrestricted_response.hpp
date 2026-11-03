#pragma once
#include "hf/reference.hpp"
#include "molecule/nuclear_gradient.hpp"
#include "response/unrestricted.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/stationary_integral_gradient.hpp"

namespace generativeqc::methods::detail {
/** Variational UHF force on the same physical reference, without a new SCF
 * solve. Both spins carry unit occupations and their own energy-weighted D. */
inline std::vector<double> unrestricted_reference_force_cpu(
    const hf::UnrestrictedPhysicalReference& reference, const scf::PreparedFockPlan& provider,
    std::size_t budget) {
  const auto n = reference.nbf;
  std::array<std::vector<double>, 2> weighted{std::vector<double>(n * n),
                                              std::vector<double>(n * n)};
  for (unsigned spin = 0; spin < 2; ++spin)
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t column = 0; column < n; ++column)
        for (std::size_t occupied = 0; occupied < reference.nocc[spin]; ++occupied)
          weighted[spin][row * n + column] += reference.coefficients[spin][row * n + occupied] *
                                              reference.orbital_energies[spin][occupied] *
                                              reference.coefficients[spin][column * n + occupied];
  const auto integral = scf::cpu_stationary_integral_gradient(
      provider, reference.density[0], weighted[0], budget, reference.density[1], weighted[1]);
  const auto coordinates = 3 * provider.system().atoms.size();
  std::vector<double> gradient(coordinates);
  for (unsigned source = 0; source < 4; ++source)
    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate)
      gradient[coordinate] += integral.sources[source * coordinates + coordinate];
  molecule::add_nuclear_repulsion_gradient(provider.system(), gradient);
  for (auto& value : gradient) value = -value;
  return gradient;
}

/** Bind the shared spin response to the same explicitly selected UHF provider. */
inline response::SpinResponseProblem unrestricted_response_problem(
    const hf::UnrestrictedPhysicalReference& ref, const scf::PreparedFockPlan& provider,
    scf::FockBuildSpec expected = scf::make_hf_fock_spec(scf::FockSpin::Unrestricted)) {
  const auto validate = [&ref, &provider, expected]() mutable {
    expected.derivative_order = provider.strategy().spec.derivative_order;
    if (!ref.source_identity || !ref.source_identity->matches(provider.system()) ||
        ref.source_charge != provider.system().charge ||
        ref.source_electrons != provider.system().electron_count ||
        ref.source_multiplicity != provider.system().multiplicity ||
        provider.strategy().spec != expected ||
        provider.strategy().screening_tolerance != 0 || !std::isfinite(ref.commutator_residual) ||
        ref.commutator_residual > 1e-8)
      throw std::invalid_argument("stale or nonstationary unrestricted response owner");
  };
  return response::spin_orbital_problem(
      ref.electronic_reference(),
      [&provider](const std::array<std::vector<double>, 2>& density) {
        auto jk = provider.build(density[0], density[1]);
        const auto n2 = jk.coulomb.size();
        if (jk.exchange_alpha.size() != n2 || jk.exchange_beta.size() != n2)
          throw std::runtime_error("unrestricted J/K response shape mismatch");
        for (std::size_t k = 0; k < n2; ++k) {
          jk.exchange_alpha[k] = jk.coulomb[k] - jk.exchange_alpha[k];
          jk.exchange_beta[k] = jk.coulomb[k] - jk.exchange_beta[k];
        }
        return std::array<std::vector<double>, 2>{std::move(jk.exchange_alpha),
                                                  std::move(jk.exchange_beta)};
      },
      validate);
}
}  // namespace generativeqc::methods::detail
