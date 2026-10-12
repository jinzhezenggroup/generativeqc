#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

#include "methods/method.hpp"
#include "scf/incremental_direct_jk_control.hpp"
#include "scf/preliminary_guess.hpp"

namespace generativeqc::methods::detail {

// Shared descriptor-to-SCF defaults. Method families retain their distinct
// numerical, backend, density-fitting, and feature admission policies.
inline scf::ScfOptions common_scf_options(const generativeqc_method_descriptor& descriptor) {
  scf::ScfOptions options;
  options.preliminary_guess = scf::initial_guess::preliminary_options(descriptor.initial_guess);
  options.max_iterations = descriptor.max_iterations == 0 ? 100 : descriptor.max_iterations;
  options.diis_history = descriptor.diis_history == 0 ? 8 : descriptor.diis_history;
  options.energy_tolerance =
      descriptor.energy_tolerance > 0.0 ? descriptor.energy_tolerance : 1.0e-10;
  options.density_tolerance =
      descriptor.density_tolerance > 0.0 ? descriptor.density_tolerance : 1.0e-8;
  options.screening_tolerance =
      descriptor.screening_tolerance > 0.0 ? descriptor.screening_tolerance : 1.0e-12;
  return options;
}

inline generativeqc_density_fitting_mode checked_density_fitting_mode(
    const generativeqc_method_descriptor& descriptor) {
  const auto mode = descriptor.density_fitting_mode;
  if (mode != GENERATIVEQC_DENSITY_FITTING_NONE &&
      mode != GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE &&
      mode != GENERATIVEQC_DENSITY_FITTING_CUDA && mode != GENERATIVEQC_DENSITY_FITTING_AUTO) {
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "unknown density-fitting execution mode");
  }
  return mode;
}

inline generativeqc_precision_mode checked_precision_mode(
    const generativeqc_method_descriptor& descriptor) {
  const auto mode = descriptor.precision_mode;
  if (mode != GENERATIVEQC_PRECISION_FP64 && mode != GENERATIVEQC_PRECISION_AUTO) {
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "unknown floating-point precision mode");
  }
  return mode;
}

// The selectors have one parser, but each family still performs its own
// capability/admission checks before allowing incremental Direct-J/K.
inline bool incremental_direct_jk_benchmark_requested(const char* name) {
  const auto control = scf::classify_incremental_direct_jk_control(std::getenv(name));
  if (control == scf::IncrementalDirectJkControl::Invalid)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      std::string(name) + " must be 0/off or 1/on");
  return control == scf::IncrementalDirectJkControl::Enabled;
}

inline std::optional<unsigned> parse_incremental_direct_jk_rebuild_interval(const char* name,
                                                                            const char* value) {
  if (value == nullptr) return std::nullopt;
  const char* end = value + std::strlen(value);
  unsigned parsed = 0U;
  const auto result = std::from_chars(value, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      std::string(name) + " must be an unsigned integer");
  return parsed;
}

inline std::optional<unsigned> incremental_direct_jk_benchmark_rebuild_interval(const char* name) {
  return parse_incremental_direct_jk_rebuild_interval(name, std::getenv(name));
}

}  // namespace generativeqc::methods::detail
