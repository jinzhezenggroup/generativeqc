#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "scf/solver/proposal_control.hpp"

int main() {
  using namespace generativeqc::scf;
  try {
    // The density is built analytically in a rotated, nonidentity AO metric.
    // Its occupations are exactly {1, 1/2, 1/2}, with trace two. No native
    // eigensolver or validation product constructs this oracle.
    const double a = 1 / std::sqrt(2.0);
    const reference::Matrix q{a, -a, 0, a, a, 0, 0, 0, 1};
    const double spectrum[]{1, 2, 3};
    ScfSnapshot state;
    state.nbf = 3;
    state.electrons = {2};
    state.occupation_weight = 1;
    state.overlap.resize(9);
    state.density.resize(9);
    for (unsigned i = 0; i < 3; ++i)
      for (unsigned j = 0; j < 3; ++j)
        for (unsigned k = 0; k < 3; ++k)
          state.overlap[i * 3 + j] += q[i * 3 + k] * spectrum[k] * q[j * 3 + k];
    const auto density = [&](double o0, double o1, double o2) {
      const double occupations[]{o0, o1, o2};
      reference::Matrix result(9);
      for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
          for (unsigned k = 0; k < 3; ++k)
            result[i * 3 + j] += q[i * 3 + k] * occupations[k] / spectrum[k] * q[j * 3 + k];
      return result;
    };
    const auto check = [&](reference::Matrix candidate, const char* expected,
                           ProposalRepresentation kind = ProposalRepresentation::ensemble_density) {
      const auto original = candidate;
      const ScfProposal proposal{kind, 0, 0, candidate};
      const auto result = solver::invalid_proposal(state, proposal);
      if (result != expected || candidate != original)
        throw std::runtime_error(std::string("proposal admission: ") + result + " expected " +
                                 expected);
    };
    check(density(1, .5, .5), "");
    check(density(1, .5, .5), "nonidempotent", ProposalRepresentation::determinant_density);
    check(density(1, 1, 0), "", ProposalRepresentation::determinant_density);
    check(density(1.01, .49, .5), "occupation_bounds");
    check(density(1, 1.01, -.01), "occupation_bounds");
    check(density(1, .5, .49), "electron_count");
    auto asymmetric = density(1, .5, .5);
    asymmetric[1] += 2e-7;
    check(asymmetric, "symmetry");
    auto singular = state.overlap;
    state.overlap.assign(9, 0);
    check(density(1, .5, .5), "singular_metric");
    state.overlap = singular;
    const auto seed = density(1, .5, .5);
    solver::validate_seed(state.overlap, seed, 3, {2}, 1);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
