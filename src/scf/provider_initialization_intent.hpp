#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "scf/provider_initialization.hpp"

namespace generativeqc::scf {

/** A copied reservation request with no authority over context/provider identity.
 * It has no mutator; binding belongs to the later native physical-reference owner. */
class ProviderInitializationIntent {
 public:
  explicit ProviderInitializationIntent(std::uint64_t allowance) {
    if (!allowance || allowance > static_cast<std::uint64_t>(INT64_MAX) ||
        allowance > std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("invalid provider initialization intent allowance");
    allowance_bytes_ = static_cast<std::size_t>(allowance);
  }
  std::size_t allowance_bytes() const noexcept { return allowance_bytes_; }
  bool operator==(const ProviderInitializationIntent&) const = default;

 private:
  std::size_t allowance_bytes_{};
};

/** Select exactly one reference allowance. The pending SDK value supplies no
 * identity; binding and lifetime admission still belong to the physical owner. */
inline std::size_t declared_reference_provider_allowance(
    const std::optional<ProviderInitializationRequest>& qualified,
    const std::optional<ProviderInitializationIntent>& pending, ReferenceProviders providers) {
  if (!pending) return provider_initialization_allowance(qualified, providers);
  if (qualified)
    throw std::invalid_argument(
        "native qualification and pending SDK intent are mutually exclusive");
  if (runtime::active_device_resource_ledger)
    throw ProviderInitializationUnavailable("provider intent has unknown active ledger basis");
  if (!providers.mask())
    throw ProviderInitializationUnavailable(
        "pending provider intent has no selected vendor domain");
  return pending->allowance_bytes();
}

}  // namespace generativeqc::scf
