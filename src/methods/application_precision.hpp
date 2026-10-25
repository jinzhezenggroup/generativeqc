#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "core/correlation_execution.hpp"
#include "methods/method.hpp"

namespace generativeqc::methods::detail {

/** Private budget transport for mean-field arithmetic and correlated stopping.
 * Each family has its own capability query and admission policy. The public method
 * descriptor remains the prefix. Callers must query the appropriate feature before
 * submitting this size; preparation copies the allowances into its own plan.
 * Unlike the legacy KS extension, these budgets do not request SCF AUTO. */
struct ApplicationPrecisionMethodOptions {
  generativeqc_method_descriptor base;
  double energy{}, maximum_force{}, rms_force{};
};
static_assert(offsetof(ApplicationPrecisionMethodOptions, base) == 0);
static_assert(offsetof(ApplicationPrecisionMethodOptions, energy) ==
              sizeof(generativeqc_method_descriptor));

/** Private additive execution extension. Zero budgets retain strict stopping;
 * requesting mixed preconvergence never fabricates an observable allowance.
 * Decode before copying the legacy prefix into a prepared owner. */
struct CorrelationExecutionMethodOptions {
  ApplicationPrecisionMethodOptions base;
  std::uint32_t df_preconvergence{};
  std::uint32_t reserved{};
};
static_assert(offsetof(CorrelationExecutionMethodOptions, base) == 0);
static_assert(offsetof(CorrelationExecutionMethodOptions, df_preconvergence) ==
              sizeof(ApplicationPrecisionMethodOptions));

/** Parse before family dispatch, including incomplete or invalid extensions.
 * An all-zero extension is equivalent to the legacy request. No mathematical
 * admission or inference of executed FP32 work occurs at this transport layer. */
inline std::array<double, 3> application_precision_budget(
    const generativeqc_method_descriptor& descriptor) {
  if (descriptor.abi_version != GENERATIVEQC_ABI_VERSION ||
      descriptor.struct_size < sizeof(generativeqc_method_descriptor))
    throw MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH, "method precision prefix ABI mismatch");
  if (descriptor.struct_size == sizeof(generativeqc_method_descriptor)) return {};
  if (descriptor.struct_size < sizeof(ApplicationPrecisionMethodOptions))
    throw MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH, "incomplete method precision extension");
  const auto& request = reinterpret_cast<const ApplicationPrecisionMethodOptions&>(descriptor);
  const std::array<double, 3> budget{request.energy, request.maximum_force, request.rms_force};
  bool requested = false;
  for (const auto allowance : budget) {
    if (!std::isfinite(allowance) || allowance < 0.0)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "application precision allowances must be finite and nonnegative");
    requested = requested || allowance != 0.0;
  }
  if (requested && budget[0] == 0.0)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "application precision requires an energy allowance");
  return budget;
}

/** Legacy descriptors keep their existing shape/allowance-dependent schedule.
 * Incomplete extensions and unknown selectors fail before native allocation. */
inline core::DfPreconvergence correlation_preconvergence(
    const generativeqc_method_descriptor& descriptor) {
  (void)application_precision_budget(descriptor);
  if (descriptor.struct_size <= sizeof(ApplicationPrecisionMethodOptions))
    return core::DfPreconvergence::Legacy;
  if (descriptor.struct_size < sizeof(CorrelationExecutionMethodOptions))
    throw MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH,
                      "incomplete correlated execution extension");
  const auto& request = reinterpret_cast<const CorrelationExecutionMethodOptions&>(descriptor);
  if (request.reserved || request.df_preconvergence > 3)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "invalid correlated preconvergence selector");
  return static_cast<core::DfPreconvergence>(request.df_preconvergence);
}

}  // namespace generativeqc::methods::detail
