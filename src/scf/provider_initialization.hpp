#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

#include "runtime/cuda_provider_identity.hpp"
#include "runtime/resource_ledger.hpp"
#include "solver/cuda/symmetric_eigen_provider.hpp"

namespace generativeqc::scf {

using ProviderInitializationEigenFamily = solver::cuda::SymmetricEigenFamily;
enum class ProviderInitializationMode { legacy_default, explicit_qualified };

struct ProviderInitializationIdentity {
  runtime::CudaProviderIdentity runtime;
  std::optional<ProviderInitializationEigenFamily> eigen_family;
  std::uint32_t policy_abi{1};
  bool operator==(const ProviderInitializationIdentity&) const = default;
};

struct ProviderInitializationRequest {
  ProviderInitializationMode mode{ProviderInitializationMode::legacy_default};
  std::size_t allowance_bytes{};
  std::optional<ProviderInitializationIdentity> expected_identity;
  bool operator==(const ProviderInitializationRequest&) const = default;
};

struct ReferenceProviders {
  bool blas{}, solver{};
  std::uint8_t mask() const noexcept { return std::uint8_t(blas) | (std::uint8_t(solver) << 1); }
};

class ProviderInitializationUnavailable : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct ResolvedProviderInitialization {
  ProviderInitializationMode mode{ProviderInitializationMode::legacy_default};
  std::size_t allowance_bytes{};
  std::optional<ProviderInitializationIdentity> identity;
  std::size_t retained_capacity(std::size_t observed) const noexcept {
    return mode == ProviderInitializationMode::explicit_qualified ? allowance_bytes : observed;
  }
};

inline bool explicit_provider_initialization(
    const std::optional<ProviderInitializationRequest>& request) noexcept {
  return request && request->mode == ProviderInitializationMode::explicit_qualified;
}

/** Resolve only the declared allowance before the existing numeric preflight.
 * The current context/build observation must still bind it before vendor create.
 * An active ledger has no proven reservation basis in this bounded interface. */
inline std::size_t provider_initialization_allowance(
    const std::optional<ProviderInitializationRequest>& request, ReferenceProviders providers) {
  const auto legacy = (std::size_t(providers.blas) + std::size_t(providers.solver)) * (96ULL << 20);
  if (!request) return legacy;
  if (request->mode == ProviderInitializationMode::legacy_default) {
    if (request->allowance_bytes || request->expected_identity)
      throw std::invalid_argument("legacy provider initialization cannot carry an override");
    return legacy;
  }
  if (request->mode != ProviderInitializationMode::explicit_qualified ||
      !request->allowance_bytes || request->allowance_bytes > static_cast<std::size_t>(INT64_MAX) ||
      !providers.mask() || !request->expected_identity)
    throw std::invalid_argument("invalid explicit provider initialization request");
  if (runtime::active_device_resource_ledger)
    throw ProviderInitializationUnavailable(
        "provider initialization has unknown active ledger basis");
  const auto& expected = *request->expected_identity;
  const auto& identity = expected.runtime;
  if (expected.policy_abi != 1 || identity.provider != runtime::CudaProviderKind::Nvidia ||
      identity.providers != providers.mask() || identity.context.device < 0 ||
      identity.context.driver_version <= 0 || identity.context.runtime_version <= 0 ||
      expected.eigen_family.has_value() != providers.solver)
    throw std::invalid_argument("provider initialization qualification domain mismatch");
  if (expected.eigen_family &&
      *expected.eigen_family != ProviderInitializationEigenFamily::jacobi_batched &&
      *expected.eigen_family != ProviderInitializationEigenFamily::xsyev_batched &&
      *expected.eigen_family != ProviderInitializationEigenFamily::xsyevd)
    throw std::invalid_argument("invalid provider initialization eigen family");
  for (unsigned index = 0; index != 4; ++index) {
    const bool required = index == 0 || index == 3 || (index == 1 && providers.blas) ||
                          (index == 2 && providers.solver);
    const auto& build = identity.build_ids[index];
    if (build.has_value() != required ||
        (build && (build->empty() || build->size() > 128 || build->size() % 2 ||
                   !std::all_of(build->begin(), build->end(), [](char value) {
                     return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
                   }))))
      throw std::invalid_argument("invalid provider initialization loaded-build identity");
  }
  return request->allowance_bytes;
}

inline ResolvedProviderInitialization resolve_reference_provider_initialization(
    const std::optional<ProviderInitializationRequest>& request, ReferenceProviders providers,
    const std::optional<ProviderInitializationIdentity>& observation) {
  const auto allowance = provider_initialization_allowance(request, providers);
  if (!explicit_provider_initialization(request))
    return {ProviderInitializationMode::legacy_default, allowance, {}};
  if (!observation)
    throw ProviderInitializationUnavailable(
        "CUDA RHF provider initialization identity unavailable");
  if (*request->expected_identity != *observation)
    throw std::invalid_argument("CUDA RHF provider initialization identity mismatch");
  return {ProviderInitializationMode::explicit_qualified, allowance, observation};
}

/** Reject unsupported request domains before any route/profitability probe.
 * Provider selection is checked again against its actual flags before numeric
 * preflight; this early check uses only the caller's declared domain. */
inline void preflight_reference_provider_initialization(
    const std::optional<ProviderInitializationRequest>& request, bool reference_export) {
  if (!request) return;
  if (request->mode == ProviderInitializationMode::legacy_default) {
    (void)provider_initialization_allowance(request, {});
    return;
  }
  if (!reference_export || !request->expected_identity)
    throw std::invalid_argument(
        "explicit provider initialization requires a qualified reference export");
  const auto mask = request->expected_identity->runtime.providers;
  (void)provider_initialization_allowance(request, {bool(mask & 1), bool(mask & 2)});
}

inline bool same_provider_initialization_request(
    const std::optional<ProviderInitializationRequest>& first,
    const std::optional<ProviderInitializationRequest>& second) {
  if (!explicit_provider_initialization(first) && !explicit_provider_initialization(second))
    return true;
  return first == second;
}

}  // namespace generativeqc::scf
