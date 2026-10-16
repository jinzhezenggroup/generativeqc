#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace generativeqc::scf::effort {

enum class Action : std::uint32_t { strict, strict_fallback, hold, relax, tighten };
enum class Reason : std::uint32_t {
  reproducible,
  missing_force,
  near_stationary,
  calibration_domain,
  energy_forecast,
  force_max_forecast,
  force_rms_forecast,
  no_admissible_level,
  relaxation_wait,
  relaxation_margin,
  tighter_allowance,
  current_allowance
};

/** Typed selection controls; these never change the physical SCF model. */
struct Policy {
  double optimizer_force_tolerance{}, intermediate_energy_error{};
  double intermediate_force_fraction{}, max_intermediate_force_error{};
  double near_stationary_factor{5.0};
  std::uint64_t relax_after{2};
  bool strict_reproducible{};
};

/** Fitted empirical envelope, not a rigorous scientific error bound.
 * Calibration provenance and joint method/basis eligibility stay with the
 * caller; the native decision never promotes an unknown domain to admission.
 */
struct Calibration {
  double energy_scale{}, force_max_scale{}, force_rms_scale{};
  double diagnostic_floor{1e-14};
};

struct State {
  std::size_t level_index{};
  std::uint64_t safe_streak{};
};

struct Estimate {
  double energy_abs{}, force_max_abs{}, force_rms{}, diagnostic_value{};
};

/** Scalar publication only; history strings/identities are presentation data.
 * Every strict path resets the safe streak, even without a level transition.
 * An estimate is published only for a calibrated hold/relax/tighten decision.
 */
struct Decision {
  Action action{};
  Reason reason{};
  State state{};
  double allowed_energy_error{}, allowed_force_error{};
  std::optional<Estimate> estimate;
};

inline bool nonnegative_finite(double value) { return std::isfinite(value) && value >= 0.0; }

inline void validate(const Policy& policy, const Calibration& calibration,
                     std::span<const double> density_tolerances, const State& state) {
  if (density_tolerances.empty() || state.level_index >= density_tolerances.size())
    throw std::invalid_argument("SCF effort state requires a nonempty matching level table");
  for (const auto value : {policy.optimizer_force_tolerance, policy.intermediate_energy_error,
                           policy.intermediate_force_fraction, policy.max_intermediate_force_error,
                           policy.near_stationary_factor, calibration.diagnostic_floor})
    if (!std::isfinite(value) || value <= 0.0)
      throw std::invalid_argument("SCF effort controls must be positive and finite");
  if (policy.intermediate_force_fraction >= 1.0 || policy.near_stationary_factor < 1.0 ||
      !policy.relax_after)
    throw std::invalid_argument("invalid SCF effort fraction, stationarity factor or hysteresis");
  for (const auto value :
       {calibration.energy_scale, calibration.force_max_scale, calibration.force_rms_scale})
    if (!nonnegative_finite(value))
      throw std::invalid_argument("SCF effort calibration scales must be finite and nonnegative");
  double previous = density_tolerances.front();
  for (const auto tolerance : density_tolerances) {
    if (!std::isfinite(tolerance) || tolerance <= 0.0 || tolerance > previous)
      throw std::invalid_argument("SCF effort diagnostics must be positive, finite and tightening");
    previous = tolerance;
  }
}

/** Select the next effort and hysteresis state without allocating level arrays.
 * The last level is strict. Missing force/reproducibility/stationarity gates
 * precede empirical forecasting. All level forecasts are checked, even after
 * finding an admissible loose level: a later invalid forecast must still force
 * strict fallback, matching the public calibration policy's fail-closed rule.
 */
inline Decision decide(const Policy& policy, const Calibration& calibration,
                       std::span<const double> density_tolerances, const State& state,
                       std::optional<double> current_force_max, bool calibrated_domain) {
  validate(policy, calibration, density_tolerances, state);
  const auto strict_index = density_tolerances.size() - 1;
  const auto strict = [&](Action action, Reason reason, double allowance) {
    return Decision{action,    reason,      {strict_index, 0}, policy.intermediate_energy_error,
                    allowance, std::nullopt};
  };
  if (policy.strict_reproducible) return strict(Action::strict, Reason::reproducible, 0.0);
  if (!current_force_max) return strict(Action::strict, Reason::missing_force, 0.0);
  if (!nonnegative_finite(*current_force_max))
    throw std::invalid_argument("current SCF effort force must be finite and nonnegative");
  const auto allowed_force =
      std::min(policy.max_intermediate_force_error,
               policy.intermediate_force_fraction *
                   std::max(*current_force_max, policy.optimizer_force_tolerance));
  if (*current_force_max <= policy.near_stationary_factor * policy.optimizer_force_tolerance)
    return strict(Action::strict, Reason::near_stationary, allowed_force);
  if (!calibrated_domain)
    return strict(Action::strict_fallback, Reason::calibration_domain, allowed_force);

  std::optional<std::size_t> target;
  Estimate selected{}, current{};
  for (std::size_t index = 0; index < density_tolerances.size(); ++index) {
    const auto diagnostic = density_tolerances[index];
    const auto effective = std::max(diagnostic, calibration.diagnostic_floor);
    const Estimate estimate{calibration.energy_scale * effective,
                            calibration.force_max_scale * effective,
                            calibration.force_rms_scale * effective, diagnostic};
    if (!nonnegative_finite(estimate.energy_abs))
      return strict(Action::strict_fallback, Reason::energy_forecast, allowed_force);
    if (!nonnegative_finite(estimate.force_max_abs))
      return strict(Action::strict_fallback, Reason::force_max_forecast, allowed_force);
    if (!nonnegative_finite(estimate.force_rms))
      return strict(Action::strict_fallback, Reason::force_rms_forecast, allowed_force);
    if (index == state.level_index) current = estimate;
    if (!target && estimate.energy_abs <= policy.intermediate_energy_error &&
        estimate.force_max_abs <= allowed_force) {
      target = index;
      selected = estimate;
    }
  }
  if (!target) return strict(Action::strict, Reason::no_admissible_level, allowed_force);
  if (*target < state.level_index) {
    if (state.safe_streak < policy.relax_after - 1)
      return {Action::hold,
              Reason::relaxation_wait,
              {state.level_index, state.safe_streak + 1},
              policy.intermediate_energy_error,
              allowed_force,
              current};
    return {Action::relax, Reason::relaxation_margin,
            {*target, 0},  policy.intermediate_energy_error,
            allowed_force, selected};
  }
  if (*target > state.level_index)
    return {Action::tighten, Reason::tighter_allowance,
            {*target, 0},    policy.intermediate_energy_error,
            allowed_force,   selected};
  return {Action::hold,  Reason::current_allowance,
          {*target, 0},  policy.intermediate_energy_error,
          allowed_force, selected};
}

/** Final observable gates remain distinct from the empirical forecast policy.
 * Identity/level equality are caller-supplied metadata comparisons, not native
 * authentication of a physical model. An identity mismatch is unverified even
 * when other measured gates pass; no reason can be dropped by that precedence.
 */
struct FinalInput {
  bool model_matches{}, strict_level{}, converged{};
  double energy{}, energy_change{}, density_rms{}, force_max_abs{};
  double energy_tolerance{}, density_tolerance{}, optimizer_force_tolerance{};
};

struct FinalVerification {
  std::uint32_t status{}, unmet_gates{};
};

inline FinalVerification verify(const FinalInput& input) {
  if (!std::isfinite(input.energy)) throw std::invalid_argument("final energy must be finite");
  for (const auto value : {input.energy_change, input.density_rms, input.force_max_abs})
    if (!nonnegative_finite(value))
      throw std::invalid_argument("final SCF diagnostics must be finite and nonnegative");
  for (const auto value :
       {input.energy_tolerance, input.density_tolerance, input.optimizer_force_tolerance})
    if (!std::isfinite(value) || value <= 0.0)
      throw std::invalid_argument("final SCF thresholds must be finite and positive");
  std::uint32_t unmet = 0;
  if (!input.model_matches) unmet |= 1u;
  if (!input.strict_level) unmet |= 2u;
  if (!input.converged) unmet |= 4u;
  if (input.energy_change > input.energy_tolerance) unmet |= 8u;
  if (input.density_rms > input.density_tolerance) unmet |= 16u;
  if (input.force_max_abs > input.optimizer_force_tolerance) unmet |= 32u;
  return {unmet & 1u ? 2u : unmet ? 1u : 0u, unmet};
}

}  // namespace generativeqc::scf::effort
