#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "solver/cpu/symmetric_eigen.hpp"

namespace generativeqc::geometry::saddle {

/** Internal local saddle corrector, not molecular task admission.
 * Callbacks supply gradients (not forces) and raw row-major Hessians at the
 * exact atom-major Bohr coordinates. Energies are Hartree. This local algorithm
 * requires a negative target curvature; it does not escape a minimum, project
 * rigid motions, choose chemistry, or establish provider/state provenance.
 */
struct Evaluation {
  double energy{};
  std::vector<double> gradient;
};

struct Options {
  std::size_t max_attempts{100}, max_coordinates{300};
  double energy_change{1e-6};
  double gradient_rms{3e-4}, gradient_max{4.5e-4};
  double displacement_rms_angstrom{1.2e-3}, displacement_max_angstrom{1.8e-3};
  double initial_radius{0.1}, minimum_radius{1e-8}, maximum_radius{0.5};
  double curvature_tolerance{1e-7}, raw_symmetry_tolerance{1e-10};
  double minimum_mode_overlap{0.5}, acceptance_ratio{0.1};
  std::vector<double> initial_mode;
};

enum class Status {
  first_order_saddle,
  stationary_other_index,
  stationary_ambiguous_curvature,
  attempt_limit,
  target_not_negative,
  mode_tracking_lost,
  trust_radius_exhausted,
  evaluation_failed,
  hessian_failed,
  cancelled
};

struct Attempt {
  std::size_t number{};
  bool accepted{};
  std::vector<double> coordinates;
  double radius{}, ratio{}, target_overlap{};
};

struct Result {
  Status status{Status::attempt_limit};
  bool has_valid_state{}, stationary_verified{};
  std::vector<double> coordinates, gradient, target_mode, curvatures;
  double energy{std::numeric_limits<double>::quiet_NaN()};
  double energy_change{}, gradient_rms{}, gradient_max{};
  double displacement_rms_angstrom{}, displacement_max_angstrom{};
  double trust_radius{}, target_curvature{}, target_overlap{};
  std::size_t attempts{}, accepted_steps{}, energy_gradient_calls{}, hessian_calls{};
  std::size_t negative_curvatures{}, near_zero_curvatures{};
  std::string failure;
  std::vector<Attempt> history;
};

using Evaluator = std::function<Evaluation(std::span<const double>)>;
using Hessian = std::function<std::vector<double>(std::span<const double>)>;
using Cancellation = std::function<bool()>;

namespace detail {
inline bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

inline double dot(std::span<const double> left, std::span<const double> right) {
  return std::inner_product(left.begin(), left.end(), right.begin(), 0.0);
}

inline double maximum(std::span<const double> values) {
  double output = 0;
  for (double value : values) output = std::max(output, std::abs(value));
  return output;
}

inline double rms(std::span<const double> values) {
  return std::sqrt(dot(values, values) / static_cast<double>(values.size()));
}

inline void validate(std::span<const double> initial, const Options& options) {
  const std::vector<double> positive{options.energy_change,
                                     options.gradient_rms,
                                     options.gradient_max,
                                     options.displacement_rms_angstrom,
                                     options.displacement_max_angstrom,
                                     options.initial_radius,
                                     options.minimum_radius,
                                     options.maximum_radius,
                                     options.curvature_tolerance,
                                     options.raw_symmetry_tolerance,
                                     options.minimum_mode_overlap,
                                     options.acceptance_ratio};
  if (initial.empty() || initial.size() % 3 || initial.size() > options.max_coordinates ||
      initial.size() > std::numeric_limits<std::size_t>::max() / initial.size() || !finite(initial))
    throw std::invalid_argument(
        "saddle coordinates require bounded finite atom-major Bohr triples");
  if (!options.max_attempts || options.max_attempts > 100 || !finite(positive) ||
      std::any_of(positive.begin(), positive.end(), [](double value) { return value <= 0; }) ||
      options.minimum_radius > options.initial_radius ||
      options.initial_radius > options.maximum_radius || options.minimum_mode_overlap > 1 ||
      options.acceptance_ratio >= 1)
    throw std::invalid_argument("invalid saddle controls or attempt cap (maximum 100)");
  if (!options.initial_mode.empty() &&
      (options.initial_mode.size() != initial.size() || !finite(options.initial_mode) ||
       !(dot(options.initial_mode, options.initial_mode) > 0) ||
       !std::isfinite(dot(options.initial_mode, options.initial_mode))))
    throw std::invalid_argument("initial saddle mode must be finite, nonzero and coordinate-sized");
}
}  // namespace detail

/** Rational eigenvector-following with exact curvature refresh after each
 * accepted step. A total Cartesian 2-norm trust radius bounds each attempt.
 * Acceptance compares actual/predicted reduction of ||gradient||^2/2, not
 * energy descent (a saddle direction can increase energy). The target is
 * tracked by absolute eigenvector overlap; ambiguous loss fails explicitly.
 * Every rejected trial consumes the same <=100 attempt budget as an accepted
 * trial. Evaluator/Hessian failures return the last fully accepted state.
 * Five convergence tests use component RMS (3N denominator), max component,
 * and unaligned Cartesian displacements converted to Angstrom. Final success
 * requires fresh energy/gradient AND raw curvature at the same coordinates.
 */
inline Result correct(std::span<const double> initial, const Options& options,
                      const Evaluator& evaluate, const Hessian& hessian,
                      const Cancellation& cancelled = {}) {
  detail::validate(initial, options);
  if (!evaluate || !hessian) throw std::invalid_argument("missing native saddle callbacks");
  Result result;
  result.coordinates.assign(initial.begin(), initial.end());
  result.trust_radius = options.initial_radius;
  const std::size_t count = initial.size();
  constexpr double bohr_to_angstrom = 0.529177210903;
  auto stop = [&] {
    if (cancelled && cancelled()) {
      result.status = Status::cancelled;
      return true;
    }
    return false;
  };
  auto energy_gradient = [&](std::span<const double> coordinates, Evaluation& output) {
    if (stop()) return false;
    ++result.energy_gradient_calls;
    try {
      output = evaluate(coordinates);
      if (!std::isfinite(output.energy) || output.gradient.size() != count ||
          !detail::finite(output.gradient) ||
          !std::isfinite(detail::dot(output.gradient, output.gradient)))
        throw std::runtime_error("invalid saddle energy/gradient");
      return true;
    } catch (const std::exception& error) {
      result.status = Status::evaluation_failed;
      result.failure = error.what();
      return false;
    }
  };
  auto curvature = [&](std::span<const double> coordinates, std::vector<double>& matrix,
                       tensor::CpuSymmetricEigenResult& spectrum) {
    if (stop()) return false;
    ++result.hessian_calls;
    try {
      matrix = hessian(coordinates);
      if (matrix.size() != count * count || !detail::finite(matrix))
        throw std::runtime_error("invalid raw saddle Hessian");
      for (std::size_t row = 0; row < count; ++row)
        for (std::size_t column = 0; column < row; ++column)
          if (std::abs(matrix[row * count + column] - matrix[column * count + row]) >
              options.raw_symmetry_tolerance)
            throw std::runtime_error("raw saddle Hessian exceeds symmetry tolerance");
      const auto prepared = solver::cpu::prepare_owned_symmetric_eigen(
          count, solver::cpu::SymmetricEigenFamily::scalar_owned_row_major);
      spectrum = solver::cpu::execute_symmetric_eigen(prepared, matrix);
      return true;
    } catch (const std::exception& error) {
      result.status = Status::hessian_failed;
      result.failure = error.what();
      return false;
    }
  };
  auto select = [&](const tensor::CpuSymmetricEigenResult& spectrum,
                    std::span<const double> previous, std::size_t& selected,
                    std::vector<double>& mode, double& overlap) {
    selected = 0;
    overlap = previous.empty() ? 1.0 : 0.0;
    double signed_overlap = 1;
    if (!previous.empty()) {
      const double norm = std::sqrt(detail::dot(previous, previous));
      for (std::size_t column = 0; column < count; ++column) {
        double candidate = 0;
        for (std::size_t row = 0; row < count; ++row)
          candidate += spectrum.vectors[row * count + column] * previous[row] / norm;
        if (std::abs(candidate) > overlap) {
          overlap = std::abs(candidate);
          signed_overlap = candidate;
          selected = column;
        }
      }
    }
    if (overlap < options.minimum_mode_overlap) {
      result.status = Status::mode_tracking_lost;
      return false;
    }
    mode.resize(count);
    for (std::size_t row = 0; row < count; ++row)
      mode[row] = std::copysign(1.0, signed_overlap) * spectrum.vectors[row * count + selected];
    return true;
  };
  auto publish = [&](const Evaluation& state, const tensor::CpuSymmetricEigenResult& spectrum,
                     const std::vector<double>& mode, std::size_t selected, double overlap) {
    result.has_valid_state = true;
    result.energy = state.energy;
    result.gradient = state.gradient;
    result.gradient_rms = detail::rms(state.gradient);
    result.gradient_max = detail::maximum(state.gradient);
    result.curvatures = spectrum.values;
    result.target_mode = mode;
    result.target_curvature = spectrum.values[selected];
    result.target_overlap = overlap;
    result.negative_curvatures =
        std::count_if(spectrum.values.begin(), spectrum.values.end(),
                      [&](double value) { return value < -options.curvature_tolerance; });
    result.near_zero_curvatures =
        std::count_if(spectrum.values.begin(), spectrum.values.end(),
                      [&](double value) { return std::abs(value) <= options.curvature_tolerance; });
  };
  auto five_tests = [&] {
    return result.energy_change <= options.energy_change &&
           result.gradient_rms <= options.gradient_rms &&
           result.gradient_max <= options.gradient_max &&
           result.displacement_rms_angstrom <= options.displacement_rms_angstrom &&
           result.displacement_max_angstrom <= options.displacement_max_angstrom;
  };
  Evaluation current;
  std::vector<double> matrix, mode;
  tensor::CpuSymmetricEigenResult spectrum;
  std::size_t selected{};
  double overlap{};
  if (!energy_gradient(result.coordinates, current) ||
      !curvature(result.coordinates, matrix, spectrum) ||
      !select(spectrum, options.initial_mode, selected, mode, overlap))
    return result;
  publish(current, spectrum, mode, selected, overlap);
  for (;;) {
    if (stop()) return result;
    if (five_tests()) {
      Evaluation final_state;
      std::vector<double> final_matrix, final_mode;
      tensor::CpuSymmetricEigenResult final_spectrum;
      std::size_t final_selected{};
      double final_overlap{};
      if (!energy_gradient(result.coordinates, final_state) ||
          !curvature(result.coordinates, final_matrix, final_spectrum) ||
          !select(final_spectrum, mode, final_selected, final_mode, final_overlap))
        return result;
      const double verified_energy_change =
          std::max(result.energy_change, std::abs(final_state.energy - current.energy));
      if (verified_energy_change > options.energy_change ||
          detail::rms(final_state.gradient) > options.gradient_rms ||
          detail::maximum(final_state.gradient) > options.gradient_max) {
        result.status = Status::evaluation_failed;
        result.failure = "fresh final verification changed the stationary state";
        return result;
      }
      result.energy_change = verified_energy_change;
      publish(final_state, final_spectrum, final_mode, final_selected, final_overlap);
      result.stationary_verified = true;
      result.status = result.near_zero_curvatures ? Status::stationary_ambiguous_curvature
                      : result.negative_curvatures == 1 &&
                              result.target_curvature < -options.curvature_tolerance
                          ? Status::first_order_saddle
                          : Status::stationary_other_index;
      return result;
    }
    if (spectrum.values[selected] >= -options.curvature_tolerance) {
      result.status = Status::target_not_negative;
      return result;
    }
    if (result.attempts == options.max_attempts) {
      result.status = Status::attempt_limit;
      return result;
    }
    std::vector<double> step(count, 0);
    for (std::size_t column = 0; column < count; ++column) {
      double component = 0;
      for (std::size_t row = 0; row < count; ++row)
        component += spectrum.vectors[row * count + column] * current.gradient[row];
      const double stiffness =
          std::max(std::abs(spectrum.values[column]), options.curvature_tolerance);
      const double amplitude =
          (column == selected ? 1 : -1) *
          (component / (0.5 * stiffness + 0.5 * std::hypot(stiffness, 2 * component)));
      for (std::size_t row = 0; row < count; ++row)
        step[row] += spectrum.vectors[row * count + column] * amplitude;
    }
    const double norm = std::sqrt(detail::dot(step, step));
    if (norm > result.trust_radius)
      for (double& component : step) component *= result.trust_radius / norm;
    std::vector<double> trial_coordinates(count), predicted_gradient = current.gradient;
    for (std::size_t row = 0; row < count; ++row) {
      trial_coordinates[row] = result.coordinates[row] + step[row];
      for (std::size_t column = 0; column < count; ++column)
        predicted_gradient[row] += matrix[row * count + column] * step[column];
    }
    ++result.attempts;
    result.history.push_back(
        {result.attempts, false, trial_coordinates, result.trust_radius, 0, overlap});
    Evaluation trial;
    if (!energy_gradient(trial_coordinates, trial)) return result;
    const double merit = 0.5 * detail::dot(current.gradient, current.gradient);
    const double predicted_reduction =
        merit - 0.5 * detail::dot(predicted_gradient, predicted_gradient);
    const double actual_reduction = merit - 0.5 * detail::dot(trial.gradient, trial.gradient);
    const bool small_stationary_step =
        detail::maximum(trial.gradient) <= options.gradient_max &&
        detail::rms(trial.gradient) <= options.gradient_rms &&
        detail::maximum(step) * bohr_to_angstrom <= options.displacement_max_angstrom &&
        detail::rms(step) * bohr_to_angstrom <= options.displacement_rms_angstrom &&
        std::abs(trial.energy - current.energy) <= options.energy_change &&
        actual_reduction >= -1e-24;
    const double ratio = predicted_reduction > 0 ? actual_reduction / predicted_reduction : 0;
    result.history.back().ratio = ratio;
    if (!(predicted_reduction > 0 && ratio >= options.acceptance_ratio) && !small_stationary_step) {
      result.trust_radius *= 0.5;
      if (result.trust_radius < options.minimum_radius) {
        result.status = Status::trust_radius_exhausted;
        return result;
      }
      continue;
    }
    std::vector<double> trial_matrix, trial_mode;
    tensor::CpuSymmetricEigenResult trial_spectrum;
    std::size_t trial_selected{};
    double trial_overlap{};
    if (!curvature(trial_coordinates, trial_matrix, trial_spectrum) ||
        !select(trial_spectrum, mode, trial_selected, trial_mode, trial_overlap))
      return result;
    result.energy_change = std::abs(trial.energy - current.energy);
    result.displacement_rms_angstrom = detail::rms(step) * bohr_to_angstrom;
    result.displacement_max_angstrom = detail::maximum(step) * bohr_to_angstrom;
    result.coordinates = std::move(trial_coordinates);
    current = std::move(trial);
    matrix = std::move(trial_matrix);
    spectrum = std::move(trial_spectrum);
    mode = std::move(trial_mode);
    selected = trial_selected;
    overlap = trial_overlap;
    ++result.accepted_steps;
    result.history.back().accepted = true;
    result.history.back().target_overlap = overlap;
    publish(current, spectrum, mode, selected, overlap);
    if (ratio > 0.75 && norm >= 0.8 * result.trust_radius)
      result.trust_radius = std::min(options.maximum_radius, 2 * result.trust_radius);
    else if (ratio < 0.25)
      result.trust_radius = std::max(options.minimum_radius, 0.5 * result.trust_radius);
  }
}
}  // namespace generativeqc::geometry::saddle
