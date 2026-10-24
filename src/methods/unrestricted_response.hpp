#pragma once
#include "hf/reference.hpp"
#include "response/unrestricted.hpp"
#include "scf/fock_prepared.hpp"

namespace generativeqc::methods::detail {
/** Bind the shared spin response to the same live exact native UHF provider. */
inline response::SpinResponseProblem unrestricted_response_problem(
    const hf::UnrestrictedPhysicalReference& ref, const scf::PreparedFockPlan& provider) {
  const auto validate = [&ref, &provider] {
    auto expected = scf::make_hf_fock_spec(scf::FockSpin::Unrestricted);
    expected.derivative_order = provider.strategy().spec.derivative_order;
    if (!ref.source_identity || !ref.source_identity->matches(provider.system()) ||
        ref.source_charge != provider.system().charge ||
        ref.source_electrons != provider.system().electron_count ||
        ref.source_multiplicity != provider.system().multiplicity ||
        provider.strategy().spec != expected ||
        provider.strategy().backend != scf::FockBackend::Cpu ||
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
