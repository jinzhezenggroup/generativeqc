#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "generativeqc/saddle.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Provider {
  unsigned energy_calls{}, hessian_calls{}, fail_energy_call{}, fail_hessian_call{};
  unsigned observations{};
  bool cancel_progress{}, reject_steps{}, asymmetric{}, minimum{};
};

int evaluate(void* user, uint32_t stage, const double* coordinates, size_t count, double* energy,
             double* gradient, double* hessian) {
  auto& provider = *static_cast<Provider*>(user);
  require(count == 3, "callback dimensions changed");
  if (stage == 0) {
    require(energy && gradient && !hessian, "energy/gradient callback pointers");
    if (++provider.energy_calls == provider.fail_energy_call) return 1;
    *energy = (provider.minimum ? 1 : -1) * coordinates[0] * coordinates[0] +
              coordinates[1] * coordinates[1] + 2 * coordinates[2] * coordinates[2];
    gradient[0] = provider.reject_steps ? 1 : (provider.minimum ? 2 : -2) * coordinates[0];
    gradient[1] = 2 * coordinates[1];
    gradient[2] = 4 * coordinates[2];
  } else {
    require(stage == 1 && hessian && !energy && !gradient, "Hessian callback pointers");
    if (++provider.hessian_calls == provider.fail_hessian_call) return 1;
    std::fill(hessian, hessian + 9, 0);
    hessian[0] = provider.minimum ? 2 : -2;
    hessian[4] = 2;
    hessian[8] = 4;
    if (provider.asymmetric) hessian[1] = 1e-3;
  }
  return 0;
}

int progress(void* user, const generativeqc_saddle_receipt_v1* receipt, const double* coordinates,
             size_t count) {
  auto& provider = *static_cast<Provider*>(user);
  ++provider.observations;
  require(receipt->has_valid_state && coordinates && count == 3, "invalid progress state");
  return provider.cancel_progress;
}

int cancel(void*) { return 1; }

int molecular(void*, uint32_t stage, const double*, size_t count, double* energy, double* gradient,
              double* hessian) {
  if (count != 9) return 1;
  if (stage == 0) {
    *energy = 0;
    std::fill(gradient, gradient + count, 0);
  } else {
    std::fill(hessian, hessian + count * count, 0);
    const double reaction[] = {-1, 2, -1}, stretch[] = {-1, 0, 1};
    for (size_t first = 0; first < 3; ++first)
      for (size_t second = 0; second < 3; ++second)
        for (size_t axis = 0; axis < 3; ++axis)
          hessian[(3 * first + axis) * count + 3 * second + axis] =
              axis == 0 ? stretch[first] * stretch[second] - reaction[first] * reaction[second]
                        : reaction[first] * reaction[second];
  }
  return 0;
}

generativeqc_saddle_options_v1 controls() {
  generativeqc_saddle_options_v1 options{};
  options.struct_size = sizeof(options);
  options.abi_version = GENERATIVEQC_ABI_VERSION;
  options.max_attempts = 100;
  options.max_coordinates = 300;
  options.maximum_numeric_bytes = 16 * 1024 * 1024;
  options.energy_change = 1e-6;
  options.gradient_rms = 3e-4;
  options.gradient_max = 4.5e-4;
  options.displacement_rms_angstrom = 1.2e-3;
  options.displacement_max_angstrom = 1.8e-3;
  options.initial_radius = 0.1;
  options.minimum_radius = 1e-8;
  options.maximum_radius = 0.5;
  options.curvature_tolerance = 1e-7;
  options.raw_symmetry_tolerance = 1e-10;
  options.minimum_mode_overlap = 0.5;
  options.acceptance_ratio = 0.1;
  return options;
}

struct Run {
  generativeqc_saddle_receipt_v1 receipt{};
  std::array<double, 3> coordinates{}, gradient{}, mode{};
  std::array<double, 9> hessian{};
  std::array<double, 3> curvatures{};
  std::array<char, 128> detail{};

  generativeqc_status execute(Provider& provider, const generativeqc_saddle_options_v1& options,
                              std::array<double, 3> initial = {0.4, -0.3, 0.2},
                              generativeqc_saddle_cancellation_v1 cancellation = nullptr) {
    receipt.struct_size = sizeof(receipt);
    receipt.abi_version = GENERATIVEQC_ABI_VERSION;
    return generativeqc_saddle_correct_v1(
        &options, initial.data(), initial.size(), nullptr, evaluate, progress, cancellation,
        &provider, &receipt, coordinates.data(), gradient.data(), hessian.data(), mode.data(),
        curvatures.data(), detail.data(), detail.size());
  }
};
}  // namespace

int main() {
  auto options = controls();
  Provider provider;
  Run run;
  require(run.execute(provider, options) == GENERATIVEQC_STATUS_SUCCESS &&
              run.receipt.status == GENERATIVEQC_SADDLE_FIRST_ORDER &&
              run.receipt.stationary_verified && run.receipt.negative_curvatures == 1 &&
              run.receipt.active_modes == 3 && run.receipt.rigid_modes == 0 &&
              run.receipt.energy_gradient_calls == provider.energy_calls &&
              run.receipt.hessian_calls == provider.hessian_calls,
          "native saddle C API did not converge with coherent work counters");
  require(run.hessian[0] == -2 && run.hessian[4] == 2 && run.hessian[8] == 4 &&
              std::abs(run.mode[0]) > 0.99 && run.curvatures[0] == -2,
          "native saddle outputs do not describe the final raw curvature");

  provider = {};
  provider.minimum = true;
  require(run.execute(provider, options, {0, 0, 0}) == GENERATIVEQC_STATUS_SUCCESS &&
              run.receipt.status == GENERATIVEQC_SADDLE_OTHER_INDEX &&
              run.receipt.stationary_verified && run.receipt.negative_curvatures == 0,
          "stationary minimum was mislabeled a saddle");
  provider = {};
  provider.fail_hessian_call = 2;
  require(run.execute(provider, options) == GENERATIVEQC_STATUS_SUCCESS &&
              run.receipt.status == GENERATIVEQC_SADDLE_HESSIAN_FAILED &&
              run.receipt.has_valid_state && !run.receipt.stationary_verified &&
              run.coordinates == std::array<double, 3>{0.4, -0.3, 0.2} && run.hessian[0] == -2 &&
              std::strstr(run.detail.data(), "Hessian callback failed"),
          "Hessian failure did not retain the last accepted state");
  provider = {};
  provider.fail_energy_call = 2;
  require(run.execute(provider, options) == GENERATIVEQC_STATUS_SUCCESS &&
              run.receipt.status == GENERATIVEQC_SADDLE_EVALUATION_FAILED &&
              run.receipt.has_valid_state && !run.receipt.stationary_verified,
          "SCF callback failure retained success semantics");
  provider = {};
  provider.asymmetric = true;
  run.hessian.fill(19);
  require(run.execute(provider, options) == GENERATIVEQC_STATUS_SUCCESS &&
              run.receipt.status == GENERATIVEQC_SADDLE_HESSIAN_FAILED &&
              !run.receipt.has_valid_state && run.hessian[0] == 19,
          "asymmetric raw Hessian was cleaned or published as valid");
  provider = {};
  provider.cancel_progress = true;
  require(run.execute(provider, options) == GENERATIVEQC_STATUS_SUCCESS &&
              run.receipt.status == GENERATIVEQC_SADDLE_CANCELLED && run.receipt.has_valid_state &&
              provider.energy_calls == 1 && provider.hessian_calls == 1,
          "progress cancellation allowed more electronic work");
  provider = {};
  require(run.execute(provider, options, {0.4, 0, 0}, cancel) == GENERATIVEQC_STATUS_SUCCESS &&
              run.receipt.status == GENERATIVEQC_SADDLE_CANCELLED && !run.receipt.has_valid_state &&
              provider.energy_calls == 0 && provider.hessian_calls == 0,
          "preparation cancellation consumed electronic work");

  provider = {};
  provider.reject_steps = true;
  options.minimum_radius = 1e-100;
  require(run.execute(provider, options, {0.4, 0, 0}) == GENERATIVEQC_STATUS_SUCCESS &&
              run.receipt.status == GENERATIVEQC_SADDLE_ATTEMPT_LIMIT &&
              run.receipt.attempts == 100 && !run.receipt.stationary_verified &&
              run.receipt.accepted_steps == 0,
          "rejected trials did not consume the 100-attempt budget");

  options = controls();
  options.maximum_numeric_bytes = 1;
  provider = {};
  run.coordinates.fill(17);
  run.receipt.status = 99;
  require(run.execute(provider, options) == GENERATIVEQC_STATUS_RESOURCE_LIMIT &&
              run.coordinates[0] == 17 && run.receipt.status == 99 && !provider.energy_calls,
          "budget rejection mutated outputs or called electronic work");
  options = controls();
  options.max_attempts = 101;
  require(run.execute(provider, options) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !provider.energy_calls,
          "out-of-contract attempt cap reached callbacks");
  options = controls();
  ++options.abi_version;
  require(
      run.execute(provider, options) == GENERATIVEQC_STATUS_ABI_MISMATCH && !provider.energy_calls,
      "wrong ABI version reached callbacks");
  options = controls();
  options.gradient_rms = -1;
  require(run.execute(provider, options) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !provider.energy_calls,
          "invalid numerical controls reached callbacks");

  options = controls();
  options.project_rigid_motions = 1;
  std::array<double, 9> geometry{-1, 0, 0, 0, 0, 0, 1, 0, 0};
  std::array<double, 9> coordinates{}, gradient{}, mode{}, curvatures{};
  std::array<double, 81> hessian{};
  const auto status = generativeqc_saddle_correct_v1(
      &options, geometry.data(), geometry.size(), nullptr, molecular, nullptr, nullptr, nullptr,
      &run.receipt, coordinates.data(), gradient.data(), hessian.data(), mode.data(),
      curvatures.data(), run.detail.data(), run.detail.size());
  require(status == GENERATIVEQC_STATUS_SUCCESS &&
              run.receipt.status == GENERATIVEQC_SADDLE_FIRST_ORDER &&
              run.receipt.stationary_verified && run.receipt.active_modes == 4 &&
              run.receipt.rigid_modes == 5 && run.receipt.negative_curvatures == 1 &&
              std::abs(curvatures[0] + 6) < 1e-12 && coordinates == geometry && hessian[0] == 0 &&
              hessian[3] == 2,
          "installed callback API lost the projected spectrum or altered the raw Hessian");
  std::cout << "native saddle C ABI: convergence, outputs, failure, cancellation, budget and cap\n";
}
