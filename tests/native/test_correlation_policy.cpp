#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "generativeqc/generativeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc_correlation_limits_v1 query(double energy, double force, double rms) {
  generativeqc_correlation_budget_v1 input{sizeof(input), GENERATIVEQC_ABI_VERSION, energy, force,
                                           rms};
  generativeqc_correlation_limits_v1 output{sizeof(output), GENERATIVEQC_ABI_VERSION};
  require(
      generativeqc_correlation_convergence_query_v1(&input, &output) == GENERATIVEQC_STATUS_SUCCESS,
      "valid policy query failed");
  require(output.struct_size == sizeof(output) && output.abi_version == GENERATIVEQC_ABI_VERSION,
          "invalid output header");
  return output;
}

void run() {
  const auto strict = query(0, 0, 0);
  require(strict.scale == 0 && strict.energy_tolerance == 1e-11 &&
              strict.density_tolerance == 1e-11 && strict.ccsd_energy_tolerance == 1e-8 &&
              strict.ccsd_residual_tolerance == 1e-9 && strict.lambda_residual_tolerance == 1e-9 &&
              strict.response_absolute_tolerance == 1e-12 &&
              strict.orbital_residual_tolerance == 1e-10 && strict.stationarity_tolerance == 1e-8,
          "strict solver contract changed");
  const auto relaxed = query(1e-5, 1e-4, 3e-5);
  require(relaxed.scale == 1 && relaxed.energy_tolerance == 1e-8 &&
              relaxed.density_tolerance == 1e-9 && relaxed.ccsd_energy_tolerance == 1e-8 &&
              relaxed.ccsd_residual_tolerance == 1e-6 &&
              relaxed.lambda_residual_tolerance == 1e-7 &&
              relaxed.response_absolute_tolerance == 1e-8 &&
              relaxed.orbital_residual_tolerance == 1e-7 && relaxed.stationarity_tolerance == 1e-6,
          "relaxed solver contract changed");
  require(query(1e-5, 1e-4, 0).scale == 1, "omitted RMS became a zero-error request");
  require(query(1e-5, 1e-4, 1.5e-5).scale == 0.5, "RMS did not constrain the policy");
  require(query(1e-5, 5e-5, 3e-5).scale == 0.5, "force did not constrain the policy");
  require(query(5e-6, 1e-4, 3e-5).scale == 0.5, "energy did not constrain the policy");
  require(query(1e-8, 1e-4, 0).scale == 1e-3, "exact bracket was rejected");
  require(query(std::nextafter(1e-8, 0.0), 1e-4, 0).scale == 0,
          "below-bracket allowance relaxed the solver");
  require(
      query(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0).scale == 1,
      "large finite allowances exceeded the cap");

  for (int test = 0; test < 15; ++test) {
    generativeqc_correlation_budget_v1 input{sizeof(input), GENERATIVEQC_ABI_VERSION, 1e-5, 1e-4,
                                             3e-5};
    auto output = relaxed;
    if (test < 9) {
      const double invalid[] = {-1, std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN()};
      if (test / 3 == 0) input.energy_abs = invalid[test % 3];
      if (test / 3 == 1) input.force_max_abs = invalid[test % 3];
      if (test / 3 == 2) input.force_rms = invalid[test % 3];
    }
    if (test == 9) input.energy_abs = 0;
    if (test == 10) input.force_max_abs = 0;
    if (test == 11) input.struct_size -= 1;
    if (test == 12) input.abi_version += 1;
    if (test == 13) output.struct_size -= 1;
    if (test == 14) output.abi_version += 1;
    const auto original = output;
    require(
        generativeqc_correlation_convergence_query_v1(&input, &output) ==
            (test >= 11 ? GENERATIVEQC_STATUS_ABI_MISMATCH : GENERATIVEQC_STATUS_INVALID_ARGUMENT),
        "malformed query was accepted");
    require(!std::memcmp(&output, &original, sizeof(output)), "error changed output");
  }
  // A union provides enough aligned storage for either public layout. The shared
  // header advertises the larger capacity; input is consumed before publication.
  union Storage {
    generativeqc_correlation_budget_v1 input;
    generativeqc_correlation_limits_v1 output;
  } storage{};
  storage.input = {sizeof(storage), GENERATIVEQC_ABI_VERSION, 1e-5, 1e-4, 3e-5};
  require(generativeqc_correlation_convergence_query_v1(&storage.input, &storage.output) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "overlapping query failed");
  require(storage.output.scale == 1 && storage.output.ccsd_residual_tolerance == 1e-6,
          "overlap changed policy");
  require(generativeqc_correlation_convergence_query_v1(nullptr, &storage.output) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "null input was admitted");
  require(generativeqc_correlation_convergence_query_v1(&storage.input, nullptr) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "null output was admitted");
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
