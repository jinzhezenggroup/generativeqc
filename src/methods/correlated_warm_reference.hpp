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
#include "generativeqc/generativeqc.h"
#include "hf/reference.hpp"
#include "methods/execution_failure.hpp"
#include "methods/geometry.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "scf/mean_field.hpp"
#include "scf/warm_state.hpp"

namespace generativeqc::methods::warm_reference {

using geometry::coordinates;
using geometry::set_coordinates;
using geometry::valid_coordinates;

/** Bind the same physical-reference acceptance and bounded seed policy to any
 * existing CPU/CUDA reference provider. Retire an unsuccessful proposal before
 * allocating its single cold replacement; never retry a resource/driver error. */
template <class Operation>
scf::ScfResult solve_reference(const std::vector<double>* seed, scf::FockSpin spin, bool* fallback,
                               Operation&& operation, std::string_view failure_message) {
  if (fallback) *fallback = false;
  const auto valid = [spin](const scf::ScfResult& result) {
    return result.converged &&
           (spin == scf::FockSpin::Restricted ? static_cast<bool>(result.reference)
                                              : static_cast<bool>(result.unrestricted_reference));
  };
  scf::ScfResult result;
  bool retry = false;
  try {
    result = operation(seed);
  } catch (...) {
    const auto failure = std::current_exception();
    if (!seed || !retryable_seed_failure(item_exception_status(failure), failure)) throw;
    retry = true;
  }
  if (seed && (retry || !valid(result))) {
    result = {};
    if (fallback) *fallback = true;
    result = operation(nullptr);
  }
  if (!valid(result))
    throw MethodError(GENERATIVEQC_STATUS_NOT_CONVERGED, std::string(failure_message));
  return result;
}

inline std::size_t payload_bytes(std::size_t density_count, std::size_t coordinate_count) {
  return posthf::checked_mul(sizeof(double), posthf::checked_add(density_count, coordinate_count));
}

inline std::size_t reservation_bytes(const core::System& system,
                                     const scf::HfWarmState* initial_state, bool retain_candidate,
                                     std::size_t spin_channels = 1) {
  std::size_t result = 0;
  if (initial_state)
    result = payload_bytes(initial_state->density.size(), initial_state->coordinates.size());
  if (retain_candidate) {
    const auto n = molecule::ao_count(system);
    result = posthf::checked_add(
        result, payload_bytes(posthf::checked_mul(spin_channels, posthf::checked_mul(n, n)),
                              posthf::checked_mul(3, system.atoms.size())));
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
                                std::string_view method_name,
                                generativeqc_method hf_method = GENERATIVEQC_METHOD_RHF) {
  const auto n = molecule::ao_count(template_system);
  const auto spins = hf_method == GENERATIVEQC_METHOD_UHF ? 2U : 1U;
  const auto expected_density = posthf::checked_mul(spins, posthf::checked_mul(n, n));
  if (state.density.size() != expected_density ||
      !valid_coordinates(state.coordinates, template_system) || state.iterations < 0 ||
      !std::isfinite(state.energy) || !std::isfinite(state.energy_change) ||
      !std::isfinite(state.density_rms) || state.density_rms < 0)
    throw std::invalid_argument("invalid " + std::string(method_name) +
                                " checkpoint state dimensions or diagnostics");
  auto source = template_system;
  set_coordinates(source, state.coordinates);
  scf::validate_hf_warm_density(source, hf_method, state.density);
}

}  // namespace generativeqc::methods::warm_reference
