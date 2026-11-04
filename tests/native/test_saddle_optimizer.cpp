#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "geometry/saddle_optimizer.hpp"

namespace saddle = generativeqc::geometry::saddle;

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

static saddle::Evaluation polynomial(std::span<const double> coordinates) {
  const double reaction = coordinates[0], stable = coordinates[1], stiff = coordinates[2];
  return {-reaction * reaction + 0.25 * std::pow(reaction, 4) + stable * stable + 2 * stiff * stiff,
          {-2 * reaction + std::pow(reaction, 3), 2 * stable, 4 * stiff}};
}

static std::vector<double> polynomial_hessian(std::span<const double> coordinates) {
  return {-2 + 3 * coordinates[0] * coordinates[0], 0, 0, 0, 2, 0, 0, 0, 4};
}

int main() {
  saddle::Options controls;
  const std::vector<double> initial{0.4, -0.3, 0.2};
  auto result = saddle::correct(initial, controls, polynomial, polynomial_hessian);
  require(result.status == saddle::Status::first_order_saddle, "known saddle did not converge");
  require(result.stationary_verified && result.negative_curvatures == 1,
          "missing saddle classification");
  require(
      result.gradient_rms <= controls.gradient_rms && result.gradient_max <= controls.gradient_max,
      "gradient criteria failed");
  require(result.energy_change <= controls.energy_change &&
              result.displacement_rms_angstrom <= controls.displacement_rms_angstrom &&
              result.displacement_max_angstrom <= controls.displacement_max_angstrom,
          "energy/displacement criteria failed");
  require(std::abs(result.coordinates[0]) < 1e-3 && std::abs(result.target_mode[0]) > 0.99,
          "incorrect stationary point or reaction mode");
  require(result.energy_gradient_calls == result.attempts + 2 &&
              result.hessian_calls == result.accepted_steps + 2,
          "fresh initial/final evaluations not separately counted");

  auto minimum = [](std::span<const double> coordinates) {
    return saddle::Evaluation{
        0.5 * (coordinates[0] * coordinates[0] + coordinates[1] * coordinates[1] +
               coordinates[2] * coordinates[2]),
        {coordinates[0], coordinates[1], coordinates[2]}};
  };
  auto positive = [](std::span<const double>) {
    return std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1};
  };
  result = saddle::correct(std::vector<double>{0, 0, 0}, controls, minimum, positive);
  require(result.stationary_verified && result.status == saddle::Status::stationary_other_index &&
              result.negative_curvatures == 0,
          "minimum mislabeled as TS");
  result = saddle::correct(initial, controls, minimum, positive);
  require(result.status == saddle::Status::target_not_negative && result.attempts == 0,
          "local corrector silently claimed minimum escape");
  controls.initial_mode = {0, 1, 0};
  result = saddle::correct(initial, controls, polynomial, polynomial_hessian);
  require(result.status == saddle::Status::target_not_negative, "wrong target mode not rejected");
  controls.initial_mode.clear();

  result = saddle::correct(
      std::vector<double>{0, 0, 0}, controls, polynomial,
      [](std::span<const double>) { return std::vector<double>{-2, 0, 0, 0, 1e-9, 0, 0, 0, 2}; });
  require(result.status == saddle::Status::stationary_ambiguous_curvature &&
              result.near_zero_curvatures == 1,
          "soft curvature silently deleted to claim a first-order saddle");

  controls.minimum_radius = 1e-100;
  auto no_progress = [](std::span<const double>) { return saddle::Evaluation{0, {1, 0, 0}}; };
  auto negative = [](std::span<const double>) {
    return std::vector<double>{-2, 0, 0, 0, 2, 0, 0, 0, 4};
  };
  result = saddle::correct(initial, controls, no_progress, negative);
  require(result.status == saddle::Status::attempt_limit && result.attempts == 100 &&
              result.accepted_steps == 0 && result.energy_gradient_calls == 101 &&
              result.coordinates == initial && result.history.size() == 100,
          "rejected attempts did not consume exactly the cap");
  for (const auto& attempt : result.history)
    require(!attempt.accepted, "rejected trial published as accepted");
  controls = {};

  result = saddle::correct(initial, controls, no_progress, negative);
  require(result.status == saddle::Status::trust_radius_exhausted && result.attempts < 100 &&
              result.coordinates == initial,
          "exhausted trust radius reported success");

  std::size_t evaluations = 0;
  result = saddle::correct(
      initial, controls,
      [&](std::span<const double> coordinates) {
        if (++evaluations == 2) throw std::runtime_error("injected SCF failure");
        return polynomial(coordinates);
      },
      polynomial_hessian);
  require(result.status == saddle::Status::evaluation_failed && result.has_valid_state &&
              result.coordinates == initial && !result.stationary_verified &&
              result.failure == "injected SCF failure",
          "failed trial lost last accepted state");
  std::size_t hessians = 0;
  result = saddle::correct(initial, controls, polynomial, [&](std::span<const double> coordinates) {
    if (++hessians == 2) throw std::runtime_error("injected response failure");
    return polynomial_hessian(coordinates);
  });
  require(result.status == saddle::Status::hessian_failed && result.coordinates == initial &&
              !result.history.back().accepted,
          "failed curvature committed a trial geometry");

  result = saddle::correct(initial, controls, polynomial, [](std::span<const double>) {
    return std::vector<double>{-2, 1, 0, 0, 2, 0, 0, 0, 4};
  });
  require(result.status == saddle::Status::hessian_failed && !result.has_valid_state,
          "asymmetric raw Hessian accepted");
  result = saddle::correct(initial, controls, polynomial, polynomial_hessian, [] { return true; });
  require(result.status == saddle::Status::cancelled && result.energy_gradient_calls == 0,
          "cancelled request evaluated electronic work");

  hessians = 0;
  result = saddle::correct(std::vector<double>{0, 0, 0}, controls, polynomial,
                           [&](std::span<const double> coordinates) {
                             if (++hessians == 2) throw std::runtime_error("final Hessian failure");
                             return polynomial_hessian(coordinates);
                           });
  require(result.status == saddle::Status::hessian_failed && !result.stationary_verified,
          "final verification failure reported success");

  evaluations = 0;
  result = saddle::correct(
      std::vector<double>{0, 0, 0}, controls,
      [&](std::span<const double> coordinates) {
        auto state = polynomial(coordinates);
        if (++evaluations == 2) state.energy = 1;
        return state;
      },
      polynomial_hessian);
  require(result.status == saddle::Status::evaluation_failed && !result.stationary_verified &&
              result.energy == 0,
          "fresh final energy drift was ignored");

  hessians = 0;
  controls.minimum_mode_overlap = 0.9;
  result = saddle::correct(initial, controls, polynomial, [&](std::span<const double> coordinates) {
    if (++hessians == 1) return polynomial_hessian(coordinates);
    return std::vector<double>{0, -2, 0, -2, 0, 0, 0, 0, 4};
  });
  require(result.status == saddle::Status::mode_tracking_lost && result.coordinates == initial,
          "lost target overlap silently switched reaction modes");
  controls = {};

  auto two_negative = [](std::span<const double> coordinates) {
    return saddle::Evaluation{-coordinates[0] * coordinates[0] - coordinates[1] * coordinates[1] +
                                  coordinates[2] * coordinates[2],
                              {-2 * coordinates[0], -2 * coordinates[1], 2 * coordinates[2]}};
  };
  auto close_hessian = [](std::span<const double>) {
    return std::vector<double>{-2, 0, 0, 0, -2.000001, 0, 0, 0, 2};
  };
  controls.initial_mode = {1, 0, 0};
  result = saddle::correct(std::vector<double>{0, 0, 0}, controls, two_negative, close_hessian);
  require(result.status == saddle::Status::stationary_other_index &&
              result.negative_curvatures == 2 && std::abs(result.target_mode[0]) > 0.99,
          "near-degenerate explicit mode was replaced by lowest eigenvalue");

  hessians = 0;
  result = saddle::correct(
      std::vector<double>{0, 0, 0}, controls, two_negative, [&](std::span<const double>) {
        ++hessians;
        return hessians == 1 ? std::vector<double>{-2, 0, 0, 0, -2.000001, 0, 0, 0, 2}
                             : std::vector<double>{-2.000001, 0, 0, 0, -2, 0, 0, 0, 2};
      });
  require(result.status == saddle::Status::stationary_other_index && result.target_overlap > 0.99 &&
              std::abs(result.target_mode[0]) > 0.99,
          "near-degenerate eigenvalue crossing lost the previous target");

  controls = {};
  controls.max_attempts = 101;
  bool invalid = false;
  try {
    saddle::correct(initial, controls, polynomial, polynomial_hessian);
  } catch (const std::invalid_argument&) {
    invalid = true;
  }
  require(invalid, "unbounded attempt cap admitted");
  std::cout << "native local saddle: algorithm, five gates, classification, cap, failures and "
               "tracking pass\n";
}
