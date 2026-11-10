#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "api/error.hpp"
#include "generativeqc/saddle.h"
#include "geometry/saddle_optimizer.hpp"

namespace {
namespace saddle = generativeqc::geometry::saddle;

uint32_t termination(saddle::Status status) {
  switch (status) {
    case saddle::Status::first_order_saddle:
      return GENERATIVEQC_SADDLE_FIRST_ORDER;
    case saddle::Status::stationary_other_index:
      return GENERATIVEQC_SADDLE_OTHER_INDEX;
    case saddle::Status::stationary_ambiguous_curvature:
      return GENERATIVEQC_SADDLE_AMBIGUOUS_CURVATURE;
    case saddle::Status::attempt_limit:
      return GENERATIVEQC_SADDLE_ATTEMPT_LIMIT;
    case saddle::Status::target_not_negative:
      return GENERATIVEQC_SADDLE_TARGET_NOT_NEGATIVE;
    case saddle::Status::mode_tracking_lost:
      return GENERATIVEQC_SADDLE_MODE_TRACKING_LOST;
    case saddle::Status::trust_radius_exhausted:
      return GENERATIVEQC_SADDLE_TRUST_EXHAUSTED;
    case saddle::Status::evaluation_failed:
      return GENERATIVEQC_SADDLE_EVALUATION_FAILED;
    case saddle::Status::hessian_failed:
      return GENERATIVEQC_SADDLE_HESSIAN_FAILED;
    case saddle::Status::cancelled:
      return GENERATIVEQC_SADDLE_CANCELLED;
  }
  throw std::logic_error("unknown native saddle termination");
}

generativeqc_saddle_receipt_v1 observation(const saddle::Result& result, std::size_t bytes) {
  generativeqc_saddle_receipt_v1 receipt{};
  receipt.struct_size = sizeof(receipt);
  receipt.abi_version = GENERATIVEQC_ABI_VERSION;
  receipt.status = termination(result.status);
  receipt.has_valid_state = result.has_valid_state;
  receipt.stationary_verified = result.stationary_verified;
  receipt.attempts = result.attempts;
  receipt.accepted_steps = result.accepted_steps;
  receipt.energy_gradient_calls = result.energy_gradient_calls;
  receipt.hessian_calls = result.hessian_calls;
  receipt.negative_curvatures = result.negative_curvatures;
  receipt.near_zero_curvatures = result.near_zero_curvatures;
  receipt.rigid_modes = result.rigid_modes;
  receipt.active_modes = result.curvatures.size();
  receipt.numeric_capacity_bytes = bytes;
  receipt.energy = result.energy;
  receipt.energy_change = result.energy_change;
  receipt.gradient_rms = result.gradient_rms;
  receipt.gradient_max = result.gradient_max;
  receipt.displacement_rms_angstrom = result.displacement_rms_angstrom;
  receipt.displacement_max_angstrom = result.displacement_max_angstrom;
  receipt.trust_radius = result.trust_radius;
  receipt.target_curvature = result.target_curvature;
  receipt.target_overlap = result.target_overlap;
  return receipt;
}
}  // namespace

extern "C" generativeqc_status generativeqc_saddle_correct_v1(
    const generativeqc_saddle_options_v1* options, const double* initial_coordinates,
    size_t coordinate_count, const double* initial_mode, generativeqc_saddle_evaluator_v1 evaluator,
    generativeqc_saddle_progress_v1 progress, generativeqc_saddle_cancellation_v1 cancelled,
    void* user, generativeqc_saddle_receipt_v1* receipt, double* coordinates, double* gradient,
    double* hessian, double* mode, double* curvatures, char* detail, size_t detail_capacity) {
  if (!options || !initial_coordinates || !evaluator || !receipt || !coordinates || !gradient ||
      !hessian || !mode || !curvatures || !detail || !detail_capacity)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (options->struct_size != sizeof(*options) || receipt->struct_size != sizeof(*receipt) ||
      options->abi_version != GENERATIVEQC_ABI_VERSION ||
      receipt->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (!coordinate_count || coordinate_count % 3 || options->max_coordinates > 300 ||
      coordinate_count > options->max_coordinates || options->project_rigid_motions > 1 ||
      !options->max_attempts || options->max_attempts > 100)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto numeric_bytes = sizeof(double) * (16 * coordinate_count * coordinate_count +
                                               (options->max_attempts + 64) * coordinate_count +
                                               16 * options->max_attempts);
  if (numeric_bytes > options->maximum_numeric_bytes) return GENERATIVEQC_STATUS_RESOURCE_LIMIT;
  try {
    saddle::Options controls;
    controls.project_rigid_motions = options->project_rigid_motions;
    controls.max_attempts = options->max_attempts;
    controls.max_coordinates = options->max_coordinates;
    controls.energy_change = options->energy_change;
    controls.gradient_rms = options->gradient_rms;
    controls.gradient_max = options->gradient_max;
    controls.displacement_rms_angstrom = options->displacement_rms_angstrom;
    controls.displacement_max_angstrom = options->displacement_max_angstrom;
    controls.initial_radius = options->initial_radius;
    controls.minimum_radius = options->minimum_radius;
    controls.maximum_radius = options->maximum_radius;
    controls.curvature_tolerance = options->curvature_tolerance;
    controls.raw_symmetry_tolerance = options->raw_symmetry_tolerance;
    controls.minimum_mode_overlap = options->minimum_mode_overlap;
    controls.acceptance_ratio = options->acceptance_ratio;
    if (initial_mode) controls.initial_mode.assign(initial_mode, initial_mode + coordinate_count);
    bool observer_cancelled = false;
    const auto result = saddle::correct(
        {initial_coordinates, coordinate_count}, controls,
        [&](std::span<const double> position) {
          saddle::Evaluation state;
          state.gradient.resize(coordinate_count);
          if (evaluator(user, 0, position.data(), coordinate_count, &state.energy,
                        state.gradient.data(), nullptr))
            throw std::runtime_error("native saddle energy/gradient callback failed");
          return state;
        },
        [&](std::span<const double> position) {
          std::vector<double> matrix(coordinate_count * coordinate_count);
          if (evaluator(user, 1, position.data(), coordinate_count, nullptr, nullptr,
                        matrix.data()))
            throw std::runtime_error("native saddle raw-Hessian callback failed");
          return matrix;
        },
        [&] { return observer_cancelled || (cancelled && cancelled(user)); },
        [&](const saddle::Result& current) {
          if (progress) {
            const auto event = observation(current, numeric_bytes);
            observer_cancelled =
                progress(user, &event, current.coordinates.data(), coordinate_count) != 0;
          }
        });
    const auto completed = observation(result, numeric_bytes);
    std::copy(result.coordinates.begin(), result.coordinates.end(), coordinates);
    if (result.has_valid_state) {
      std::copy(result.gradient.begin(), result.gradient.end(), gradient);
      std::copy(result.hessian.begin(), result.hessian.end(), hessian);
      std::copy(result.target_mode.begin(), result.target_mode.end(), mode);
      std::copy(result.curvatures.begin(), result.curvatures.end(), curvatures);
    }
    const auto length = std::min(result.failure.size(), detail_capacity - 1);
    std::memcpy(detail, result.failure.data(), length);
    detail[length] = '\0';
    *receipt = completed;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(nullptr);
  }
}
