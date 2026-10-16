#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "generativeqc/generativeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Descriptor>
Descriptor versioned() {
  Descriptor result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  return result;
}

void run() {
  const std::array<double, 3> levels{{1e-4, 1e-6, 1e-8}};
  auto descriptor = versioned<generativeqc_scf_effort_descriptor_v1>();
  descriptor.flags = 2;
  descriptor.level_count = levels.size();
  descriptor.level_index = 2;
  descriptor.relax_after = 2;
  descriptor.density_tolerances = levels.data();
  descriptor.optimizer_force_tolerance = 1e-4;
  descriptor.intermediate_energy_error = 1e-5;
  descriptor.intermediate_force_fraction = 0.2;
  descriptor.max_intermediate_force_error = 5e-3;
  descriptor.near_stationary_factor = 5.0;
  descriptor.energy_scale = 0.03;
  descriptor.force_max_scale = 3.0;
  descriptor.force_rms_scale = 2.0;
  descriptor.diagnostic_floor = 1e-14;
  descriptor.current_force_max = 0.1;
  auto output = versioned<generativeqc_scf_effort_decision_v1>();
  require(generativeqc_scf_effort_decide_v1(&descriptor, &output) == GENERATIVEQC_STATUS_SUCCESS,
          "installed policy call failed");
  require(output.action == GENERATIVEQC_SCF_EFFORT_HOLD &&
              output.reason == GENERATIVEQC_SCF_EFFORT_RELAXATION_WAIT && output.level_index == 2 &&
              output.safe_streak == 1 && output.has_estimate == 1 &&
              output.diagnostic_value == levels[2],
          "installed policy bypassed strict hysteresis or published a wrong estimate");
  descriptor.safe_streak = output.safe_streak;
  require(generativeqc_scf_effort_decide_v1(&descriptor, &output) == GENERATIVEQC_STATUS_SUCCESS,
          "second policy call failed");
  require(output.action == GENERATIVEQC_SCF_EFFORT_RELAX && output.level_index == 0 &&
              output.safe_streak == 0 && output.has_estimate == 1 &&
              std::abs(output.energy_abs - 3e-6) < 1e-20,
          "installed policy failed independent scalar allowance gate");
  const auto detached = output;

  const auto invalid = [&](const auto& malformed, generativeqc_status expected) {
    auto sentinel = versioned<generativeqc_scf_effort_decision_v1>();
    sentinel.level_index = 919;
    sentinel.energy_abs = -77.0;
    std::array<unsigned char, sizeof(sentinel)> previous{};
    std::memcpy(previous.data(), &sentinel, sizeof(sentinel));
    require(generativeqc_scf_effort_decide_v1(&malformed, &sentinel) == expected,
            "invalid installed policy input returned wrong status");
    require(std::memcmp(previous.data(), &sentinel, sizeof(sentinel)) == 0,
            "refused policy input modified its output");
  };
  auto malformed = descriptor;
  malformed.flags = 4;
  invalid(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.calibration_domain = 3;
  invalid(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.level_count = std::numeric_limits<std::uint64_t>::max();
  invalid(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.density_tolerances = nullptr;
  invalid(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.level_index = malformed.level_count;
  invalid(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.relax_after = 0;
  invalid(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.current_force_max = std::numeric_limits<double>::quiet_NaN();
  invalid(malformed, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  malformed = descriptor;
  malformed.abi_version += 1;
  invalid(malformed, GENERATIVEQC_STATUS_ABI_MISMATCH);
  malformed = descriptor;
  malformed.struct_size -= 1;
  invalid(malformed, GENERATIVEQC_STATUS_ABI_MISMATCH);
  require(
      generativeqc_scf_effort_decide_v1(nullptr, &output) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
          generativeqc_scf_effort_decide_v1(&descriptor, nullptr) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
      "null policy descriptors were accepted");

  descriptor.calibration_domain = 2;
  require(generativeqc_scf_effort_decide_v1(&descriptor, &output) == GENERATIVEQC_STATUS_SUCCESS &&
              output.action == GENERATIVEQC_SCF_EFFORT_STRICT_FALLBACK && output.level_index == 2 &&
              output.safe_streak == 0 && !output.has_estimate && output.energy_abs == 0.0,
          "unknown joint calibration domain did not fall back explicitly");
  descriptor.flags = 3;
  descriptor.current_force_max = std::numeric_limits<double>::quiet_NaN();
  require(generativeqc_scf_effort_decide_v1(&descriptor, &output) == GENERATIVEQC_STATUS_SUCCESS &&
              output.reason == GENERATIVEQC_SCF_EFFORT_REPRODUCIBLE &&
              output.allowed_force_error == 0.0,
          "strict reproducibility inspected an unused force");
  require(detached.level_index == 0 && detached.diagnostic_value == levels[0],
          "later calls invalidated detached policy publication");

  auto final = versioned<generativeqc_scf_effort_final_descriptor_v1>();
  final.energy = -1.0;
  final.energy_tolerance = 1e-10;
  final.density_tolerance = 1e-8;
  final.optimizer_force_tolerance = 1e-4;
  for (std::uint32_t mask = 0; mask < 64; ++mask) {
    final.flags = (~mask) & 7u;
    final.energy_change =
        mask & 8u ? std::nextafter(final.energy_tolerance, INFINITY) : final.energy_tolerance;
    final.density_rms =
        mask & 16u ? std::nextafter(final.density_tolerance, INFINITY) : final.density_tolerance;
    final.force_max_abs = mask & 32u ? std::nextafter(final.optimizer_force_tolerance, INFINITY)
                                     : final.optimizer_force_tolerance;
    auto verified = versioned<generativeqc_scf_effort_verification_v1>();
    require(generativeqc_scf_effort_verify_v1(&final, &verified) == GENERATIVEQC_STATUS_SUCCESS &&
                verified.unmet_gates == mask &&
                verified.status == (mask & 1u ? 2u
                                    : mask    ? 1u
                                              : 0u),
            "final strict verification dropped a gate or changed inclusive status precedence");
  }
  auto verified = versioned<generativeqc_scf_effort_verification_v1>();
  verified.status = 919;
  std::array<unsigned char, sizeof(verified)> previous{};
  std::memcpy(previous.data(), &verified, sizeof(verified));
  final.energy = std::numeric_limits<double>::quiet_NaN();
  require(generativeqc_scf_effort_verify_v1(&final, &verified) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              std::memcmp(previous.data(), &verified, sizeof(verified)) == 0,
          "invalid final metadata changed verification publication");
  final.energy = -1.0;
  final.flags = 8;
  require(
      generativeqc_scf_effort_verify_v1(&final, &verified) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
      "unknown final flags were accepted");
  final.flags = 7;
  verified.abi_version += 1;
  std::memcpy(previous.data(), &verified, sizeof(verified));
  require(
      generativeqc_scf_effort_verify_v1(&final, &verified) == GENERATIVEQC_STATUS_ABI_MISMATCH &&
          std::memcmp(previous.data(), &verified, sizeof(verified)) == 0,
      "invalid final output header was published");
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
