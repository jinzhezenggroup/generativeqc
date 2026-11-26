#include <algorithm>
#include <limits>
#include <new>
#include <vector>

#include "core/cbs_extrapolation.hpp"
#include "core/correlation_convergence.hpp"
#include "generativeqc/generativeqc.h"

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
    *output = {sizeof(*output),           GENERATIVEQC_ABI_VERSION, policy.scale,
               policy.scf_energy(),       policy.scf_density(),     policy.cc_energy(),
               policy.cc_residual(),      policy.lambda_residual(), policy.response_solve(),
               policy.orbital_residual(), policy.stationarity()};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    // No C++ exception may escape the public C boundary, including allocation
    // failure while the shared validator constructs an invalid-input diagnostic.
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
