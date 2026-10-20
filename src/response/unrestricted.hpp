#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/electronic_reference.hpp"
#include "response/solve.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::response {

/** Shared real UHF Jacobian: symmetric spin OV densities, J[Pa+Pb]-K[Ps].
 * The prepared native mean-field owner supplies the physical two-electron
 * action. No independent integral formula or SCF solve is introduced here.
 * Both borrowed owners must outlive the problem and every synchronous action.
 */
using SpinFockAction =
    std::function<std::array<std::vector<double>, 2>(const std::array<std::vector<double>, 2>&)>;

class SpinResponseProblem : public LinearResponseProblem {
 public:
  SpinResponseProblem(LinearResponseProblem problem, core::ElectronicReferenceView ref)
      : LinearResponseProblem(std::move(problem)), reference_(ref) {}
  bool matches(core::ElectronicReferenceView ref) const noexcept {
    const auto same = [](auto a, auto b) { return a.size() == b.size() && a.data() == b.data(); };
    if (ref.basis_functions != reference_.basis_functions || ref.spin_channels != 2 ||
        ref.energy != reference_.energy || !same(ref.overlap, reference_.overlap) ||
        !same(ref.hcore, reference_.hcore))
      return false;
    for (unsigned s = 0; s < 2; ++s) {
      const auto& a = ref.channels[s];
      const auto& b = reference_.channels[s];
      if (a.occupied != b.occupied || !same(a.coefficients, b.coefficients) ||
          !same(a.orbital_energies, b.orbital_energies) || !same(a.density, b.density) ||
          !same(a.fock, b.fock))
        return false;
    }
    return true;
  }

 private:
  core::ElectronicReferenceView reference_;
};

inline SpinResponseProblem spin_orbital_problem(core::ElectronicReferenceView ref,
                                                SpinFockAction induced_fock,
                                                std::function<void()> validate_current) {
  const auto n = ref.basis_functions;
  const auto n2 = detail::response_checked_multiply(n, n);
  const auto validate = [ref, validate_current, induced_fock] {
    if (!validate_current || !induced_fock)
      throw std::invalid_argument("spin response requires a live owner and physical Fock action");
    validate_current();
    core::validate_electronic_reference_shape(ref);
    if (ref.spin_channels != 2) throw std::invalid_argument("spin response requires two channels");
    for (const auto& spin : ref.channels)
      if (!std::all_of(spin.orbital_energies.begin(), spin.orbital_energies.end(),
                       [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("invalid spin response frame");
  };
  validate();
  std::size_t dimension = 0;
  for (auto channel : ref.channels)
    dimension = detail::response_checked_add(
        dimension, detail::response_checked_multiply(channel.occupied, n - channel.occupied));
  LinearResponseProblem problem{
      dimension,
      [ref, induced_fock, n, n2, dimension, validate](std::span<const double> x,
                                                      std::span<double> out) {
        validate();
        if (x.size() != dimension || out.size() != dimension)
          throw std::invalid_argument("unrestricted response vector shape mismatch");
        std::array<std::vector<double>, 2> density;
        std::vector<double> mo(n2), scratch(n2);
        std::size_t offset = 0;
        for (unsigned spin = 0; spin < 2; ++spin) {
          std::fill(mo.begin(), mo.end(), 0.0);
          const auto no = ref.channels[spin].occupied, nv = n - no;
          for (std::size_t i = 0; i < no; ++i)
            for (std::size_t a = 0; a < nv; ++a)
              mo[i * n + no + a] = mo[(no + a) * n + i] = x[offset + i * nv + a];
          density[spin].resize(n2);
          tensor::cpu_congruence('N', n, ref.channels[spin].coefficients.data(), mo.data(),
                                 density[spin].data(), scratch.data());
          offset += no * nv;
        }
        const auto fock = induced_fock(density);
        offset = 0;
        for (unsigned spin = 0; spin < 2; ++spin) {
          if (fock[spin].size() != n2)
            throw std::runtime_error("spin Fock response shape mismatch");
          std::copy(fock[spin].begin(), fock[spin].end(), scratch.begin());
          tensor::cpu_congruence('T', n, ref.channels[spin].coefficients.data(), scratch.data(),
                                 mo.data(), density[spin].data());
          const auto no = ref.channels[spin].occupied, nv = n - no;
          for (std::size_t i = 0; i < no; ++i)
            for (std::size_t a = 0; a < nv; ++a)
              out[offset + i * nv + a] =
                  mo[(no + a) * n + i] + (ref.channels[spin].orbital_energies[no + a] -
                                          ref.channels[spin].orbital_energies[i]) *
                                             x[offset + i * nv + a];
          offset += no * nv;
        }
      },
      LinearResponseSymmetry::Symmetric};
  return {std::move(problem), ref};
}
}  // namespace generativeqc::response
