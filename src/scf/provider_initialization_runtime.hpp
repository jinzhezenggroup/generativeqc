#pragma once

#include <atomic>
#include <memory>
#include <mutex>

#include "scf/provider_initialization_generation.hpp"

namespace generativeqc::integrals {
class ElectronInteractionSource;
}
namespace generativeqc::core {
struct System;
}

namespace generativeqc::scf {
struct CudaRhfBucketPlan;

struct ProviderReferenceStreamUse {
  void* stream{};
  void* event{};
  bool pending{}, recorded{};
};

/** Shared native lifetime of one opt-in physical reference generation. Its
 * implementation owns the bucket, graphs and consumer events before creation.
 * No caller supplies a runtime identity, qualification bit or teardown status. */
class PendingReferenceProviderOwner {
 public:
  virtual ~PendingReferenceProviderOwner() = default;
  virtual void attach_bucket(CudaRhfBucketPlan*) = 0;
  virtual CudaRhfBucketPlan* bucket() const noexcept = 0;
  virtual void bind_current(ReferenceProviders,
                            std::optional<ProviderInitializationEigenFamily>) = 0;
  virtual void require_current() = 0;
  virtual const ResolvedProviderInitialization& resolved() const = 0;
  virtual const std::array<std::shared_ptr<void>, 4>& pins() const = 0;
  virtual ProviderInitializationTeardownReport close_current() = 0;
  virtual bool closed() const noexcept = 0;
  virtual bool identity_bound() const noexcept = 0;
  virtual std::size_t allowance_bytes() const noexcept = 0;
  virtual void mark_uncertain(const std::string&, std::optional<std::uint32_t>) = 0;
  virtual ProviderReferenceStreamUse& prepare_use(void*) = 0;
  virtual bool finish_use_current(ProviderReferenceStreamUse&) = 0;
  virtual bool finish_uses_current() = 0;
  virtual std::shared_ptr<std::atomic<bool>> acquire_source_lease() = 0;
  virtual std::unique_lock<std::recursive_mutex> acquire_borrow_lock() = 0;
  virtual void release_source_lease(const std::shared_ptr<std::atomic<bool>>&) noexcept = 0;
  virtual void register_cache_slot(CudaRhfBucketPlan**, CudaRhfBucketPlan** previous = nullptr) = 0;
  virtual bool source_lease_active() const noexcept = 0;
  virtual void begin_reference_replay() = 0;
};

/** Inert metadata construction only; first identity bind follows normal native
 * context establishment. Existing reference/endpoint envelopes are not vouchers. */
std::shared_ptr<PendingReferenceProviderOwner> make_pending_reference_provider_owner(
    ProviderInitializationIntent, int device, std::size_t reference_budget,
    std::size_t endpoint_budget);
std::shared_ptr<const integrals::ElectronInteractionSource> make_pending_rhf_reference_source(
    core::System&&, std::shared_ptr<PendingReferenceProviderOwner>, std::size_t retained_bytes);
}  // namespace generativeqc::scf
