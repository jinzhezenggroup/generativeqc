#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "hf/reference.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "scf/mean_field.hpp"
#include "scf/warm_state.hpp"

namespace generativeqc::methods::warm_reference {

inline std::vector<double> coordinates(const core::System& system) {
  std::vector<double> result;
  result.reserve(3 * system.atoms.size());
  for (const auto& atom : system.atoms)
    result.insert(result.end(), atom.position.begin(), atom.position.end());
  return result;
}

inline bool valid_coordinates(const std::vector<double>& values, const core::System& system) {
  return values.size() == 3 * system.atoms.size() &&
         std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

inline void set_coordinates(core::System& system, const std::vector<double>& values) {
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    std::copy_n(values.begin() + 3 * atom, 3, system.atoms[atom].position.begin());
}

inline std::size_t payload_bytes(std::size_t density_count, std::size_t coordinate_count) {
  return checked_mul(sizeof(double), checked_add(density_count, coordinate_count));
}

inline std::size_t reservation_bytes(const core::System& system,
                                     const scf::HfWarmState* initial_state, bool retain_candidate) {
  std::size_t result = 0;
  if (initial_state)
    result = payload_bytes(initial_state->density.size(), initial_state->coordinates.size());
  if (retain_candidate) {
    const auto n = molecule::ao_count(system);
    result =
        checked_add(result, payload_bytes(checked_mul(n, n), checked_mul(3, system.atoms.size())));
  }
  return result;
}

inline scf::HfWarmState capture(const core::System& system, std::vector<double> density,
                                double energy, double energy_change, double density_rms,
                                int iterations) {
  scf::HfWarmState state;
  state.density = std::move(density);
  state.coordinates = coordinates(system);
  state.energy = energy;
  state.energy_change = energy_change;
  state.density_rms = density_rms;
  state.iterations = iterations;
  return state;
}

inline scf::HfWarmState capture(const core::System& system, const hf::PhysicalReference& reference,
                                double energy_change, double density_rms, int iterations) {
  return capture(system, reference.density, reference.energy, energy_change, density_rms,
                 iterations);
}

inline void validate_checkpoint(const core::System& template_system, const scf::HfWarmState& state,
                                std::string_view method_name) {
  const auto n = molecule::ao_count(template_system);
  const auto expected_density = checked_mul(n, n);
  if (state.density.size() != expected_density ||
      !valid_coordinates(state.coordinates, template_system) || state.iterations < 0 ||
      !std::isfinite(state.energy) || !std::isfinite(state.energy_change) ||
      !std::isfinite(state.density_rms) || state.density_rms < 0)
    throw std::invalid_argument("invalid " + std::string(method_name) +
                                " checkpoint state dimensions or diagnostics");
  auto source = template_system;
  set_coordinates(source, state.coordinates);
  scf::validate_hf_warm_density(source, GENERATIVEQC_METHOD_RHF, state.density);
}

}  // namespace generativeqc::methods::warm_reference
