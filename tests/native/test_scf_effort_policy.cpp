#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "scf/effort_policy.hpp"

namespace {
using namespace generativeqc::scf::effort;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void run() {
  const std::array<double, 3> levels{{1e-4, 1e-6, 1e-8}};
  const Policy policy{1e-4, 1e-5, 0.2, 5e-3, 5.0, 2, false};
  const Calibration calibration{0.03, 3.0, 2.0, 1e-14};
  const auto choose = [&](State state, std::optional<double> force, bool domain = true) {
    return decide(policy, calibration, levels, state, force, domain);
  };
  const auto missing = choose({0, 7}, std::nullopt, false);
  require(missing.action == Action::strict && missing.reason == Reason::missing_force &&
              missing.state.level_index == 2 && missing.state.safe_streak == 0 &&
              missing.allowed_force_error == 0.0 && !missing.estimate,
          "missing force did not select strict before calibration");
  const auto near = choose({0, 7}, 5e-4, false);
  require(near.reason == Reason::near_stationary && near.action == Action::strict,
          "inclusive near-stationary gate did not precede calibration");
  const auto outside = choose({0, 7}, 0.1, false);
  require(outside.action == Action::strict_fallback && outside.state.level_index == 2,
          "uncalibrated domain was admitted");
  const auto waiting = choose({2, 0}, 0.1);
  require(waiting.action == Action::hold && waiting.reason == Reason::relaxation_wait &&
              waiting.state.level_index == 2 && waiting.state.safe_streak == 1 &&
              waiting.estimate && waiting.estimate->diagnostic_value == levels[2],
          "first safe step bypassed hysteresis or published a loose estimate");
  const auto relaxed = choose(waiting.state, 0.1);
  require(relaxed.action == Action::relax && relaxed.state.level_index == 0 &&
              relaxed.state.safe_streak == 0 && relaxed.estimate &&
              std::abs(relaxed.estimate->energy_abs - 3e-6) < 1e-20 &&
              relaxed.allowed_force_error == 5e-3,
          "sustained margin did not select the first admissible level");
  const auto tighter = choose({0, 9}, 0.001);
  require(tighter.action == Action::tighten && tighter.state.level_index == 1 &&
              tighter.state.safe_streak == 0,
          "tightening incorrectly waited for hysteresis");
  const auto unchanged = choose({1, 9}, 0.001);
  require(unchanged.action == Action::hold && unchanged.reason == Reason::current_allowance &&
              unchanged.state.safe_streak == 0,
          "unchanged effort failed to clear the safe streak");
  auto reproducible = policy;
  reproducible.strict_reproducible = true;
  const auto strict = decide(reproducible, calibration, levels, {0, 9},
                             std::numeric_limits<double>::quiet_NaN(), false);
  require(strict.reason == Reason::reproducible && !strict.estimate,
          "reproducible mode inspected an unused force/calibration");
  const auto maximal = choose({2, std::numeric_limits<std::uint64_t>::max()}, 0.1);
  require(maximal.action == Action::relax && maximal.state.safe_streak == 0,
          "safe-streak publication overflowed instead of relaxing");
  auto impossible = calibration;
  impossible.energy_scale = 1e12;
  require(
      decide(policy, impossible, levels, {0, 0}, 0.1, true).reason == Reason::no_admissible_level,
      "unmet energy allowance was admitted");
  auto overflowing = calibration;
  overflowing.force_rms_scale = std::numeric_limits<double>::max();
  const std::array<double, 2> oversized{{2.0, 1.0}};
  require(decide(policy, overflowing, oversized, {0, 0}, 0.1, true).reason ==
              Reason::force_rms_forecast,
          "nonfinite empirical RMS forecast did not fail closed");
  bool rejected = false;
  try {
    (void)choose({3, 0}, 0.1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "out-of-range policy state was accepted");
}
}  // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
