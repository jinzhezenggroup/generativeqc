#ifndef GENERATIVEQC_RUNTIME_RESOURCE_LEDGER_HPP
#define GENERATIVEQC_RUNTIME_RESOURCE_LEDGER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "runtime/resource_usage.hpp"

namespace generativeqc::runtime {

struct DeviceAllocationEvent {
  std::uint64_t kind{};
  std::uint64_t generation{};
  std::uint64_t bytes{};
};

/** Opt-in, finite event storage owned independently of an observation handle.
 * The registry mutex serializes snapshots, records and capture shutdown. Slot
 * exhaustion invalidates coverage rather than allocating in a replay path. */
struct DeviceAllocationJournal {
  std::vector<DeviceAllocationEvent> initial;
  std::vector<DeviceAllocationEvent> events;
  std::size_t limit{};
  std::uint64_t dropped{};
  bool recording{true};

  void record(std::uint64_t kind, std::uint64_t generation, std::size_t bytes) noexcept {
    if (!recording) return;
    if (events.size() == limit) {
      if (dropped != std::numeric_limits<std::uint64_t>::max()) ++dropped;
      return;
    }
    events.push_back({kind, generation, bytes});
  }
};

/** Numeric device allocations owned by one prepared resource request.
 *
 * The shared owner outlives a Python observation scope: cached native buffers
 * remain charged between calls, and can be freed on a different host thread.
 * CUDA context, graphs, pools and library-internal allocations are not measured
 * numeric allocations. Native retained-provider allowances can reserve admission
 * capacity separately; callers must not report those reservations as CUDA bytes
 * allocated or add them to independently withheld planner allowances.
 */
struct DeviceResourceLedger {
  std::size_t limit{};
  int device{};
  std::size_t live{};
  std::size_t peak{};
  std::size_t allocations{};
  std::size_t rejected{};
  bool active{};
  /** Successful owned requests in the current binding, independent of frees.
   * Overflow invalidates v2 observations, never changes allocation policy. */
  std::uint64_t requested_bytes{};
  bool requested_bytes_overflow{};
  std::shared_ptr<DeviceAllocationJournal> journal;
  std::size_t provider_reserved{}, provider_reservation_peak{}, admission_peak{};
  std::uint64_t declared_peak{};
  std::uint64_t native_host_peak{}, native_host_samples{};
  std::uint64_t native_cuda_arena_peak{}, native_cuda_arena_samples{};
};

struct DeviceAllocationOwner {
  std::shared_ptr<DeviceResourceLedger> ledger;
  std::size_t bytes{};
  std::uint64_t generation{};
};

inline std::mutex device_resource_mutex;
inline std::unordered_map<void*, DeviceAllocationOwner> device_allocation_owners;
inline std::uint64_t device_allocation_generation{};
inline thread_local std::shared_ptr<DeviceResourceLedger> active_device_resource_ledger;
inline thread_local std::size_t native_device_resource_scope_depth{};

/** Bounded prepared inventories own one serialized host workspace, even if an
 * optional observer has ended or no frontend observation was requested. */
inline unsigned resource_cpu_worker_limit() noexcept {
  return native_device_resource_scope_depth ? 1 : cpu_resource_observation.cpu_worker_limit;
}

/** Begin one synchronous admission scope, retaining charges from prior calls.
 * Observation and native prepared execution use the same counters and owner;
 * a foreign or concurrently active ledger is never implicitly adopted. */
inline bool bind_device_resource_ledger(std::shared_ptr<DeviceResourceLedger> ledger) {
  if (!ledger || active_device_resource_ledger) return false;
  std::lock_guard<std::mutex> lock(device_resource_mutex);
  if (ledger->active) return false;
  ledger->active = true;
  ledger->peak = ledger->live;
  ledger->allocations = 0;
  ledger->rejected = 0;
  ledger->requested_bytes = 0;
  ledger->requested_bytes_overflow = false;
  ledger->provider_reservation_peak = ledger->provider_reserved;
  ledger->admission_peak = ledger->live + ledger->provider_reserved;
  active_device_resource_ledger = std::move(ledger);
  return true;
}

/** End an observer-owned binding, but never detach an executing native owner. */
inline void unbind_device_resource_ledger() noexcept {
  if (native_device_resource_scope_depth || !active_device_resource_ledger) return;
  {
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    active_device_resource_ledger->active = false;
  }
  active_device_resource_ledger.reset();
}

/** Native prepared-operation admission independent of Python/CPU observation.
 * Same-owner nesting borrows the binding without resetting its counters. The
 * outer native scope restores the unbound state on every return/exception;
 * legacy observation may still own an already-bound identical ledger. */
class DeviceResourceScope {
 public:
  explicit DeviceResourceScope(std::shared_ptr<DeviceResourceLedger> ledger)
      : ledger_(std::move(ledger)) {
    if (!ledger_) return;
    if (native_device_resource_scope_depth == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("native device resource scope depth overflow");
    if (active_device_resource_ledger != ledger_) {
      if (!bind_device_resource_ledger(ledger_))
        throw std::invalid_argument("native device resource owner is already bound");
      owns_binding_ = true;
    }
    if (native_device_resource_scope_depth == 0) {
      {
        std::lock_guard<std::mutex> lock(device_resource_mutex);
        ledger_->native_host_peak = ledger_->native_host_samples = 0;
        ledger_->native_cuda_arena_peak = ledger_->native_cuda_arena_samples = 0;
      }
      if (!cpu_resource_observation.active) {
        previous_observation_ = cpu_resource_observation;
        cpu_resource_observation = {true, 0, 0, 1};
        owns_observation_ = true;
      }
    }
    ++native_device_resource_scope_depth;
  }
  DeviceResourceScope(const DeviceResourceScope&) = delete;
  DeviceResourceScope& operator=(const DeviceResourceScope&) = delete;
  ~DeviceResourceScope() {
    if (!ledger_) return;
    if (owns_observation_) {
      {
        std::lock_guard<std::mutex> lock(device_resource_mutex);
        ledger_->native_host_peak = cpu_resource_observation.peak_bytes;
        ledger_->native_host_samples = cpu_resource_observation.samples;
        ledger_->native_cuda_arena_peak = cpu_resource_observation.cuda_arena_peak_bytes;
        ledger_->native_cuda_arena_samples = cpu_resource_observation.cuda_arena_samples;
      }
      cpu_resource_observation = previous_observation_;
    }
    --native_device_resource_scope_depth;
    if (owns_binding_) unbind_device_resource_ledger();
  }

 private:
  std::shared_ptr<DeviceResourceLedger> ledger_;
  bool owns_binding_{};
  bool owns_observation_{};
  CpuResourceObservation previous_observation_{};
};

/** Remaining request capacity, including retained native provider reservations.
 * This is shape-only admission, not a device/context/RSS measurement. */
inline std::size_t device_resource_available(int device) {
  const auto ledger = active_device_resource_ledger;
  if (!ledger) return std::numeric_limits<std::size_t>::max();
  if (device != ledger->device) throw std::invalid_argument("resource request/device mismatch");
  std::lock_guard<std::mutex> lock(device_resource_mutex);
  if (ledger->live > ledger->limit || ledger->provider_reserved > ledger->limit - ledger->live)
    return 0;
  return ledger->limit - ledger->live - ledger->provider_reserved;
}

/** Reserve a retained provider allowance without inventing a numeric allocation.
 * The lease follows the actual native owner, including release on another host
 * thread or after the observation handle has closed. Destroy the provider before
 * releasing its allowance. No CUDA operation is needed for admission. */
class DeviceProviderAllowance {
 public:
  DeviceProviderAllowance(int device, std::size_t bytes) : ledger_(active_device_resource_ledger) {
    if (!ledger_ || !bytes) return;
    if (device != ledger_->device)
      throw std::invalid_argument("provider allowance/device mismatch");
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    if (ledger_->live > ledger_->limit ||
        ledger_->provider_reserved > ledger_->limit - ledger_->live ||
        bytes > ledger_->limit - ledger_->live - ledger_->provider_reserved) {
      ++ledger_->rejected;
      throw std::bad_alloc();
    }
    bytes_ = bytes;
    ledger_->provider_reserved += bytes;
    ledger_->provider_reservation_peak =
        std::max(ledger_->provider_reservation_peak, ledger_->provider_reserved);
    ledger_->admission_peak =
        std::max(ledger_->admission_peak, ledger_->live + ledger_->provider_reserved);
  }
  DeviceProviderAllowance(const DeviceProviderAllowance&) = delete;
  DeviceProviderAllowance& operator=(const DeviceProviderAllowance&) = delete;
  ~DeviceProviderAllowance() {
    if (!bytes_) return;
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    ledger_->provider_reserved -= bytes_;
  }

 private:
  std::shared_ptr<DeviceResourceLedger> ledger_;
  std::size_t bytes_{};
};

}  // namespace generativeqc::runtime

#endif
