#ifndef GENERATIVEQC_METHODS_DFT_ADMISSION_HPP
#define GENERATIVEQC_METHODS_DFT_ADMISSION_HPP

#include <cmath>
#include <string>
#include <string_view>

#include "dft/global_hybrid_composition.hpp"
#include "generativeqc/generativeqc.h"

#if GENERATIVEQC_HAS_CUDA
#include "generated_split_hybrid_registry.cuh"
#endif

namespace generativeqc::methods::detail {

/** Descriptor-only admission before either single or batch scientific owners.
 * A registered split semilocal pair requires its exact-exchange contribution:
 * a missing/zero K term must not take the ordinary pure-semilocal bypass.
 * Names/coefficients come from the generated registry, never another table. */
inline generativeqc_status validate_split_hybrid_descriptor(
    const generativeqc_method_descriptor& descriptor, generativeqc_backend backend,
    std::string& detail) {
  const auto reject = [&](generativeqc_status status, const char* message) {
    detail = message;
    return status;
  };
  detail.clear();
  const auto* input = descriptor.ks_options;
  if (!input) return GENERATIVEQC_STATUS_SUCCESS;
  if (input->struct_size < sizeof(generativeqc_ks_options) ||
      input->abi_version != GENERATIVEQC_ABI_VERSION)
    return reject(GENERATIVEQC_STATUS_ABI_MISMATCH, "KS execution-plan ABI mismatch");
  if ((input->semilocal_components == nullptr) != (input->semilocal_component_count == 0) ||
      (input->exchange_terms == nullptr) != (input->exchange_term_count == 0))
    return reject(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "inconsistent KS component pointer/count");
#if GENERATIVEQC_HAS_CUDA
  if (input->semilocal_component_count != 2) return GENERATIVEQC_STATUS_SUCCESS;
  const auto& first = input->semilocal_components[0];
  const auto& second = input->semilocal_components[1];
  if (!first.component_id || !second.component_id || !*first.component_id ||
      !*second.component_id || !std::isfinite(first.coefficient) ||
      !std::isfinite(second.coefficient))
    return reject(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "invalid KS semilocal component");
  const auto code = dft::generated::split_hybrid_functional_code(
      std::string_view(first.component_id), std::string_view(second.component_id));
  if (!code) return GENERATIVEQC_STATUS_SUCCESS;
  if (input->spin_channels != 1 && input->spin_channels != 2)
    return reject(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "invalid KS spin channel count");
  // XC point evaluation is independent of the full-range J/K approximation.
  // Only a manifest-qualified complete fitted J/K and force consumer may be
  // substituted for Direct; other registered hybrids remain Direct-only.
  const bool direct_jk = descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE;
  const bool fitted_jk = (descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_CUDA ||
                          descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_AUTO) &&
                         dft::cuda_global_hybrid_df_admitted(code);
  // Direct arithmetic belongs to the shared Fock modules, independently of XC.
  // Fitted consumers retain their separate strict-precision contract.
  const bool precision_admitted =
      descriptor.precision_mode == GENERATIVEQC_PRECISION_FP64 ||
      (direct_jk && descriptor.precision_mode == GENERATIVEQC_PRECISION_AUTO);
  if (backend != GENERATIVEQC_BACKEND_CUDA || !precision_admitted || (!direct_jk && !fitted_jk) ||
      input->xc_execution_schedule != GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED)
    return reject(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                  "split-global-hybrid KS requires CUDA, qualified Direct/DF J/K, "
                  "device-fused XC and supported provider precision");
  if (first.coefficient != 1.0 || second.coefficient != 1.0 ||
      input->semilocal_range_omega != 0.0 || input->has_nonlocal_correlation != 0 ||
      descriptor.method == GENERATIVEQC_METHOD_PBE_D4_RKS)
    return reject(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                  "split-global-hybrid KS requires its unmodified electronic composition");
  if (input->exchange_term_count != 1)
    return reject(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                  "split-global-hybrid KS requires one nonzero full-range exact-exchange term");
  const auto composition = dft::global_hybrid_composition(code);
  if (!composition.present || !composition.fixed_semilocal)
    return reject(GENERATIVEQC_STATUS_INTERNAL_ERROR,
                  "split-hybrid registry composition is missing");
  const auto& exchange = input->exchange_terms[0];
  const double exact = composition.exact_exchange;
  const double divisor = input->spin_channels == 1 ? 2.0 : 1.0;
  if (exchange.operator_kind != GENERATIVEQC_KS_EXCHANGE_FULL_RANGE || exchange.omega != 0.0 ||
      exchange.coefficient != exact || exchange.fock_coefficient != -exact / divisor)
    return reject(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                  "split-global-hybrid KS exact exchange disagrees with the generated composition");
#else
  (void)backend;
#endif
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // namespace generativeqc::methods::detail
#endif
