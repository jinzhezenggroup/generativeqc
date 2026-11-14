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
using SpinFockBatchAction = std::function<std::vector<SpinFockAction::result_type>(
    std::span<const SpinFockAction::result_type>)>;

namespace detail {
/** Canonical symmetric OV density map, shared by scalar and batched schedules.
 * The transform panels are reused sequentially; batching only retains densities
 * needed by the physical same-frame source, never a second orbital equation.
 */
inline SpinFockAction::result_type spin_orbital_density(core::ElectronicReferenceView ref,
                                                        std::span<const double> input,
                                                        std::span<double> mo,
                                                        std::span<double> scratch) {
  const auto basis = ref.basis_functions;
  SpinFockAction::result_type density;
  std::size_t offset = 0;
  for (unsigned spin = 0; spin < 2; ++spin) {
    std::fill(mo.begin(), mo.end(), 0.0);
    const auto occupied = ref.channels[spin].occupied, virtuals = basis - occupied;
    for (std::size_t occupied_index = 0; occupied_index < occupied; ++occupied_index)
      for (std::size_t virtual_index = 0; virtual_index < virtuals; ++virtual_index)
        mo[occupied_index * basis + occupied + virtual_index] =
            mo[(occupied + virtual_index) * basis + occupied_index] =
                input[offset + occupied_index * virtuals + virtual_index];
    density[spin].resize(mo.size());
    tensor::cpu_congruence('N', basis, ref.channels[spin].coefficients.data(), mo.data(),
                           density[spin].data(), scratch.data());
    offset += occupied * virtuals;
  }
  return density;
}

/** Canonical Fock projection/gap action. The now-consumed density panels serve
 * as transform scratch, retaining the ordinary scalar reduction/BLAS schedule.
 */
inline void spin_orbital_projection(core::ElectronicReferenceView ref,
                                    std::span<const double> input,
                                    const SpinFockAction::result_type& fock,
                                    SpinFockAction::result_type& density, std::span<double> mo,
                                    std::span<double> scratch, std::span<double> output) {
  const auto basis = ref.basis_functions;
  std::size_t offset = 0;
  for (unsigned spin = 0; spin < 2; ++spin) {
    if (fock[spin].size() != mo.size())
      throw std::runtime_error("spin Fock response shape mismatch");
    std::copy(fock[spin].begin(), fock[spin].end(), scratch.begin());
    tensor::cpu_congruence('T', basis, ref.channels[spin].coefficients.data(), scratch.data(),
                           mo.data(), density[spin].data());
    const auto occupied = ref.channels[spin].occupied, virtuals = basis - occupied;
    for (std::size_t occupied_index = 0; occupied_index < occupied; ++occupied_index)
      for (std::size_t virtual_index = 0; virtual_index < virtuals; ++virtual_index)
        output[offset + occupied_index * virtuals + virtual_index] =
            mo[(occupied + virtual_index) * basis + occupied_index] +
            (ref.channels[spin].orbital_energies[occupied + virtual_index] -
             ref.channels[spin].orbital_energies[occupied_index]) *
                input[offset + occupied_index * virtuals + virtual_index];
    offset += occupied * virtuals;
  }
}
}  // namespace detail

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
                                                std::function<void()> validate_current,
                                                SpinFockBatchAction induced_batch = {}) {
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
  BatchLinearOperator batch;
  if (induced_batch) {
    batch = [ref, n2, validate,
             induced_batch = std::move(induced_batch)](std::span<const LinearAction> actions) {
      validate();
      std::vector<double> mo(n2), scratch(n2);
      std::vector<SpinFockAction::result_type> density;
      density.reserve(actions.size());
      for (const auto& action : actions)
        density.push_back(detail::spin_orbital_density(ref, action.input, mo, scratch));
      const auto fock = induced_batch(density);
      if (fock.size() != actions.size())
        throw std::runtime_error("spin Fock response batch shape mismatch");
      for (std::size_t index = 0; index < actions.size(); ++index)
        detail::spin_orbital_projection(ref, actions[index].input, fock[index], density[index], mo,
                                        scratch, actions[index].output);
    };
  }
  LinearResponseProblem problem{
      dimension,
      [ref, induced_fock, n2, dimension, validate](std::span<const double> x,
                                                   std::span<double> out) {
        validate();
        if (x.size() != dimension || out.size() != dimension)
          throw std::invalid_argument("unrestricted response vector shape mismatch");
        std::vector<double> mo(n2), scratch(n2);
        auto density = detail::spin_orbital_density(ref, x, mo, scratch);
        const auto fock = induced_fock(density);
        detail::spin_orbital_projection(ref, x, fock, density, mo, scratch, out);
      },
      LinearResponseSymmetry::Symmetric,
      {},
      std::move(batch)};
  return {std::move(problem), ref};
}
}  // namespace generativeqc::response
