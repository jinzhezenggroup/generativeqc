#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "scf/provider_initialization.hpp"
#include "scf/provider_initialization_intent.hpp"
#include "scf/provider_initialization_teardown.hpp"

namespace generativeqc::scf {

class ProviderInitializationObservationFailure : public std::runtime_error {
 public:
  ProviderInitializationObservationFailure(runtime::IdentityErrorDomain domain,
                                           std::uint32_t status)
      : std::runtime_error("provider initialization observation failed: status=" +
                           std::to_string(status)),
        domain_(domain),
        status_(status) {}
  runtime::IdentityErrorDomain domain() const noexcept { return domain_; }
  std::uint32_t status() const noexcept { return status_; }

 private:
  runtime::IdentityErrorDomain domain_;
  std::uint32_t status_;
};

/** Identity/lease contract for one native resource generation, not a CUDA
 * resource destructor. Bind only after the owner has normally established its
 * context. Observers cannot create/select/retain/reset a context. Concrete
 * ownership must retain this contract and its handles on unconfirmed retirement;
 * this object's destructor never attempts hardware cleanup or certifies it. */
class PendingProviderInitializationGeneration {
 public:
  PendingProviderInitializationGeneration(std::optional<ProviderInitializationIntent> intent,
                                          ReferenceProviders providers,
                                          std::optional<ProviderInitializationEigenFamily> family,
                                          int device)
      : intent_(std::move(intent)), providers_(providers), family_(family), device_(device) {}
  PendingProviderInitializationGeneration(const PendingProviderInitializationGeneration&) = delete;
  PendingProviderInitializationGeneration& operator=(
      const PendingProviderInitializationGeneration&) = delete;

  std::size_t declared_allowance() const {
    if (!intent_) return provider_initialization_allowance({}, providers_);
    if (!providers_.mask() || device_ < 0)
      throw std::invalid_argument("pending provider intent has no physical provider domain");
    if (runtime::active_device_resource_ledger)
      throw ProviderInitializationUnavailable(
          "provider initialization has unknown active ledger basis");
    return intent_->allowance_bytes();
  }

  template <class Observe>
  void bind_current(Observe&& observe) {
    if (bound_ || retired_)
      throw std::logic_error("provider initialization generation is already bound or retired");
    const auto allowance = declared_allowance();
    if (!intent_) {
      resolved_ = {ProviderInitializationMode::legacy_default, allowance, {}};
      bound_ = true;
      return;
    }
    const auto observation = std::forward<Observe>(observe)();
    auto candidate = resolve(observation);
    resolved_ = std::move(candidate);
    pins_ = observation.library_pins;
    bound_ = true;
  }

  template <class Observe>
  void require_reuse(Observe&& observe) {
    if (!bound_) throw std::logic_error("provider initialization generation is not bound");
    if (retired_)
      throw ProviderInitializationUnavailable("provider initialization generation is retired");
    if (!intent_) return;
    (void)declared_allowance();
    try {
      const auto observation = std::forward<Observe>(observe)();
      const auto candidate = resolve(observation);
      if (candidate.identity != resolved_.identity || observation.library_pins != pins_)
        throw ProviderInitializationUnavailable(
            "provider initialization actual owner generation changed");
    } catch (...) {
      retired_ = true;
      throw;
    }
  }

  template <class Observe, class Teardown>
  ProviderInitializationTeardownReport retire_current(Observe&& observe, Teardown&& teardown) {
    if (!intent_ || !bound_)
      return {{}, "no bound explicit generation; ordinary ownership handles cleanup"};
    retired_ = true;
    if (teardown_confirmed_) return retirement_;
    try {
      const auto observation = std::forward<Observe>(observe)();
      const auto current = actual_identity(observation);
      if (current != resolved_.identity || observation.library_pins != pins_)
        return retirement_ = {{},
                              "teardown unavailable: original context/loaded owner is not current"};
      auto result = std::forward<Teardown>(teardown)();
      retirement_ = std::move(result);
      if (retirement_.confirmed()) {
        pins_ = {};
        teardown_confirmed_ = true;
      }
    } catch (const ProviderInitializationObservationFailure& error) {
      retirement_ = {{}, std::string("teardown unavailable: ") + error.what()};
      throw;
    } catch (const std::exception& error) {
      retirement_ = {{}, std::string("teardown unavailable: ") + error.what()};
    } catch (...) {
      retirement_ = {{}, "teardown unavailable: callback did not supply complete status"};
    }
    return retirement_;
  }

  const ResolvedProviderInitialization& resolved() const {
    if (!bound_) throw std::logic_error("provider initialization generation is not bound");
    return resolved_;
  }
  bool has_bound_identity() const noexcept { return bound_ && resolved_.identity.has_value(); }
  bool retired() const noexcept { return retired_; }

 private:
  ProviderInitializationIdentity actual_identity(
      const runtime::CudaProviderIdentityObservation& observation) const {
    if (observation.status == runtime::ContextIdentityStatus::query_error)
      throw ProviderInitializationObservationFailure(observation.error_domain,
                                                     observation.error_status);
    if (observation.status != runtime::ContextIdentityStatus::available || !observation.identity ||
        observation.loaded_status != runtime::LoadedIdentityStatus::available)
      throw ProviderInitializationUnavailable(
          "provider initialization actual identity unavailable");
    if (observation.identity->context.device != device_)
      throw std::invalid_argument(
          "provider initialization actual identity belongs to another device");
    for (unsigned index = 0; index != 4; ++index) {
      const bool required = index == 0 || index == 3 || (index == 1 && providers_.blas) ||
                            (index == 2 && providers_.solver);
      if (required && !observation.library_pins[index])
        throw ProviderInitializationUnavailable(
            "provider initialization actual identity has no lease");
    }
    return {*observation.identity, family_, 1};
  }

  ResolvedProviderInitialization resolve(
      const runtime::CudaProviderIdentityObservation& observation) const {
    const auto actual = actual_identity(observation);
    // The native observation is the sole source of this key. No SDK transport
    // field can provide expected context/build values or a qualification bit.
    const ProviderInitializationRequest qualified{ProviderInitializationMode::explicit_qualified,
                                                  intent_->allowance_bytes(), actual};
    return resolve_reference_provider_initialization(qualified, providers_, actual);
  }

  const std::optional<ProviderInitializationIntent> intent_;
  const ReferenceProviders providers_;
  const std::optional<ProviderInitializationEigenFamily> family_;
  const int device_;
  bool bound_{}, retired_{}, teardown_confirmed_{};
  ResolvedProviderInitialization resolved_;
  std::array<std::shared_ptr<void>, 4> pins_;
  ProviderInitializationTeardownReport retirement_;
};

}  // namespace generativeqc::scf
