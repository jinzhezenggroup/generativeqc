#pragma once

#include "scf/provider_initialization.hpp"
#include "tensor/cuda_error.hpp"

namespace generativeqc::scf {

class ProviderIdentityDriverError : public std::runtime_error {
 public:
  explicit ProviderIdentityDriverError(std::uint32_t status)
      : std::runtime_error("CUDA RHF provider identity driver query failed: status=" +
                           std::to_string(status)),
        status_(status) {}
  std::uint32_t status() const noexcept { return status_; }

 private:
  std::uint32_t status_;
};

/** Bind the native runtime observation, never request-supplied evidence. The
 * ordinary owner calls this only for an explicit request after establishing its
 * current context and before creating any vendor handle. */
inline ResolvedProviderInitialization bind_reference_provider_initialization(
    const std::optional<ProviderInitializationRequest>& request, ReferenceProviders providers,
    std::optional<ProviderInitializationEigenFamily> family, int device,
    const runtime::CudaProviderIdentityObservation& observation) {
  if (!explicit_provider_initialization(request))
    return resolve_reference_provider_initialization(request, providers, {});
  if (observation.status == runtime::ContextIdentityStatus::query_error) {
    if (observation.error_domain == runtime::IdentityErrorDomain::cuda)
      generativeqc_tensor::cuda_check(static_cast<cudaError_t>(observation.error_status));
    throw ProviderIdentityDriverError(observation.error_status);
  }
  (void)provider_initialization_allowance(request, providers);
  if (observation.status != runtime::ContextIdentityStatus::available || !observation.identity)
    throw ProviderInitializationUnavailable(
        "CUDA RHF provider initialization identity unavailable");
  if (observation.identity->context.device != device)
    throw std::invalid_argument("CUDA RHF provider identity belongs to another device");
  for (unsigned index = 0; index != 4; ++index) {
    const bool required = index == 0 || index == 3 || (index == 1 && providers.blas) ||
                          (index == 2 && providers.solver);
    if (required && !observation.library_pins[index])
      throw ProviderInitializationUnavailable(
          "CUDA RHF provider identity has no loaded-object lease");
  }
  return resolve_reference_provider_initialization(
      request, providers, ProviderInitializationIdentity{*observation.identity, family, 1});
}

}  // namespace generativeqc::scf
