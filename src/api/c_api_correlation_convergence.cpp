#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

#include "core/cbs_extrapolation.hpp"
#include "core/correlation_convergence.hpp"
#include "generativeqc/generativeqc.h"

namespace {
generativeqc_correlation_limits_v1 make_limits(
    const generativeqc::core::CorrelationConvergence& policy) {
  return {sizeof(generativeqc_correlation_limits_v1),
          GENERATIVEQC_ABI_VERSION,
          policy.scale,
          policy.scf_energy(),
          policy.scf_density(),
          policy.cc_energy(),
          policy.cc_residual(),
          policy.lambda_residual(),
          policy.response_solve(),
          policy.orbital_residual(),
          policy.stationarity()};
}

// CBS formerly used math.fsum. A single compensated correction can lose the
// bit that decides a midpoint for near-singular plans. Keep all six terms in
// an error-free FP64 expansion (smallest magnitude first), then round once.
double coefficient_l1(const std::array<double, 6>& coefficients) {
  std::array<double, 6> partials{};
  std::size_t count = 0;
  for (const double coefficient : coefficients) {
    double carry = std::abs(coefficient);
    std::size_t kept = 0;
    for (std::size_t index = 0; index < count; ++index) {
      double tail = partials[index];
      if (std::abs(carry) < std::abs(tail)) std::swap(carry, tail);
      const double rounded = carry + tail;
      if (!std::isfinite(rounded)) return std::numeric_limits<double>::infinity();
      const double residual = tail - (rounded - carry);
      if (residual != 0) partials[kept++] = residual;
      carry = rounded;
    }
    partials[kept++] = carry;
    count = kept;
  }
  double leading = partials[--count];
  while (count != 0) {
    const double tail = partials[--count];
    const double rounded = leading + tail;
    if (!std::isfinite(rounded)) return rounded;
    const double residual = tail - (rounded - leading);
    leading = rounded;
    if (residual == 0) continue;
    // At a midpoint the next partial determines which side the exact sum is
    // on. Step one ULP only when that partial reinforces the midpoint residual.
    if (count && (residual > 0) == (partials[count - 1] > 0)) {
      const double step = 2 * residual;
      const double adjusted = leading + step;
      if (adjusted - leading == step) leading = adjusted;
    }
    break;
  }
  return leading;
}
}  // namespace

extern "C" generativeqc_status generativeqc_correlation_convergence_query_v1(
    const generativeqc_correlation_budget_v1* descriptor,
    generativeqc_correlation_limits_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (descriptor->struct_size < sizeof(*descriptor) ||
      descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      output->struct_size < sizeof(*output) || output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    const auto policy = generativeqc::core::CorrelationConvergence::from_budget(
        {descriptor->energy_abs, descriptor->force_max_abs, descriptor->force_rms});
    // Finish reading input before publishing, including for overlapping buffers.
    *output = make_limits(policy);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    // No C++ exception may escape the public C boundary, including allocation
    // failure while the shared validator constructs an invalid-input diagnostic.
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

extern "C" generativeqc_status generativeqc_cbs_convergence_query_v1(
    const generativeqc_cbs_convergence_descriptor_v1* descriptor,
    generativeqc_cbs_convergence_result_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto valid = [](const auto& value) {
    return value.struct_size >= sizeof(value) && value.abi_version == GENERATIVEQC_ABI_VERSION;
  };
  if (!valid(*descriptor) || !valid(*output)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (!descriptor->extrapolation || descriptor->correction_basis > 2 ||
      descriptor->default_standard > 1)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!valid(*descriptor->extrapolation) ||
      (descriptor->total_budget && !valid(*descriptor->total_budget)))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    const auto& plan = *descriptor->extrapolation;
    const auto finite = [](double value) { return std::isfinite(value); };
    if (!std::ranges::all_of(plan.coefficients, finite) ||
        !std::ranges::all_of(plan.hf_weights, finite) ||
        !std::ranges::all_of(plan.mp2_weights, finite) || plan.coefficients[4] != 1.0 ||
        plan.coefficients[5] != -1.0)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t basis = 0; basis < 2; ++basis) {
      if (plan.coefficients[basis] != plan.hf_weights[basis] - plan.mp2_weights[basis] ||
          plan.coefficients[2 + basis] != plan.mp2_weights[basis])
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    std::array<double, 6> coefficients;
    std::ranges::copy(plan.coefficients, coefficients.begin());
    // The correction subtraction and extrapolation use the very same MP2
    // result on a shared basis; only its net coefficient amplifies error.
    if (descriptor->correction_basis < 2) {
      coefficients[2 + descriptor->correction_basis] += coefficients[5];
      coefficients[5] = 0.0;
    }
    const double amplification = coefficient_l1(coefficients);
    if (!std::isfinite(amplification) || amplification < 1.0)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    generativeqc_correlation_budget_v1 total{sizeof(total), GENERATIVEQC_ABI_VERSION, 0, 0, 0};
    if (const auto* budget = descriptor->total_budget) {
      total.energy_abs = budget->energy_abs;
      total.force_max_abs = budget->force_max_abs;
      total.force_rms = budget->force_rms;
      if (!finite(total.energy_abs) || total.energy_abs <= 0.0 || !finite(total.force_max_abs) ||
          total.force_max_abs <= 0.0 || !finite(total.force_rms) || total.force_rms < 0.0)
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    } else if (descriptor->default_standard) {
      total.energy_abs = 1e-5;
      total.force_max_abs = 1e-4;
      total.force_rms = 3e-5;
    }
    generativeqc_correlation_budget_v1 component{
        sizeof(component), GENERATIVEQC_ABI_VERSION, total.energy_abs / amplification,
        total.force_max_abs / amplification, total.force_rms / amplification};
    if ((total.energy_abs > 0 && component.energy_abs == 0) ||
        (total.force_max_abs > 0 && component.force_max_abs == 0) ||
        (total.force_rms > 0 && component.force_rms == 0))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto policy = generativeqc::core::CorrelationConvergence::from_budget(
        {component.energy_abs, component.force_max_abs, component.force_rms});
    *output = {sizeof(*output), GENERATIVEQC_ABI_VERSION, amplification, total,
               component,       make_limits(policy)};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

extern "C" generativeqc_status generativeqc_cbs_extrapolation_query_v1(
    const generativeqc_cbs_extrapolation_descriptor_v1* descriptor,
    generativeqc_cbs_extrapolation_plan_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (descriptor->struct_size < sizeof(*descriptor) ||
      descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      output->struct_size < sizeof(*output) || output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (descriptor->reserved || descriptor->hf_extrapolation > 1)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    const auto plan = generativeqc::core::CbsExtrapolation::make(
        descriptor->low_cardinal, descriptor->high_cardinal, descriptor->correlation_exponent,
        descriptor->hf_exponent, descriptor->hf_extrapolation == 0);
    generativeqc_cbs_extrapolation_plan_v1 result{};
    result.struct_size = sizeof(result);
    result.abi_version = GENERATIVEQC_ABI_VERSION;
    std::copy(plan.hf_weights.begin(), plan.hf_weights.end(), result.hf_weights);
    std::copy(plan.mp2_weights.begin(), plan.mp2_weights.end(), result.mp2_weights);
    std::copy(plan.coefficients.begin(), plan.coefficients.end(), result.coefficients);
    *output = result;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

extern "C" generativeqc_status generativeqc_cbs_compose_v1(
    const generativeqc_cbs_extrapolation_plan_v1* plan,
    const generativeqc_cbs_observables_v1* components, generativeqc_cbs_composition_v1* output) {
  if (!plan || !components || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto valid = [](const auto& descriptor) {
    return descriptor.struct_size >= sizeof(descriptor) &&
           descriptor.abi_version == GENERATIVEQC_ABI_VERSION;
  };
  if (!valid(*plan) || !valid(*components) || !valid(*output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  const std::size_t count = components->force_count;
  if (count % 3 || count > std::numeric_limits<std::size_t>::max() / 4 / sizeof(double) ||
      (count ? !output->force_blocks || output->force_capacity < 4 * count
             : output->force_blocks || output->force_capacity))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  for (const auto* forces : components->forces)
    if ((forces != nullptr) != (count != 0)) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const std::array hf{plan->hf_weights[0], plan->hf_weights[1]};
  const std::array mp2{plan->mp2_weights[0], plan->mp2_weights[1]};
  const auto finite = [](double value) { return std::isfinite(value); };
  if (!std::ranges::all_of(hf, finite) || !std::ranges::all_of(mp2, finite) ||
      !std::ranges::all_of(components->energy, finite))
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  std::array<double, 6> values;
  std::copy_n(components->energy, values.size(), values.begin());
  const auto energies = generativeqc::core::CbsExtrapolation::compose(hf, mp2, values, true);
  if (!std::ranges::all_of(energies, finite)) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  try {
    // Stage all four outputs before any caller-visible write. A nonfinite
    // late component must not leave a plausible-looking partial result.
    std::vector<double> staged(4 * count);
    for (std::size_t coordinate = 0; coordinate < count; ++coordinate) {
      for (std::size_t role = 0; role < values.size(); ++role)
        values[role] = components->forces[role][coordinate];
      if (!std::ranges::all_of(values, finite)) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
      const auto forces = generativeqc::core::CbsExtrapolation::compose(hf, mp2, values, false);
      if (!std::ranges::all_of(forces, finite)) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
      for (std::size_t block = 0; block < forces.size(); ++block)
        staged[block * count + coordinate] = forces[block];
    }
    if (count) std::copy(staged.begin(), staged.end(), output->force_blocks);
    std::copy(energies.begin(), energies.end(), output->energy);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}
