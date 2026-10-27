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

#include "resource_usage.hpp"

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

/** Selected inventories may include consumers outside the native method owner.
 * Keep that authority distinct from a cap whose future native owners can each
 * reserve their own complete bound. Neither mode implies exclusive ownership. */
enum class DeviceResourceAdmission : std::uint32_t { SelectedInventory = 0, NumericCap = 1 };

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
  DeviceResourceAdmission admission{DeviceResourceAdmission::SelectedInventory};
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
  /** Unspent native endpoint promises. These are neither live CUDA bytes nor
   * provider allocations; allocating within a promise converts this charge. */
  std::size_t capacity_reserved{};
  std::uint64_t declared_peak{};
  std::uint64_t native_host_peak{}, native_host_samples{};
  std::uint64_t native_cuda_arena_peak{}, native_cuda_arena_samples{};
};

/** Shared bookkeeping only: the explicit reservation owner closes the promise.
 * Allocation records may retain this metadata after close without keeping its
 * unused capacity reserved or requiring a locking shared_ptr destructor. */
struct DeviceCapacityPromise {
  std::shared_ptr<DeviceResourceLedger> ledger;
  std::size_t remaining{};
  bool open{true};
};

struct DeviceAllocationOwner {
  std::shared_ptr<DeviceResourceLedger> ledger;
  std::size_t bytes{};
  std::uint64_t generation{};
  std::shared_ptr<DeviceCapacityPromise> promise;
};

inline std::mutex device_resource_mutex;
inline std::unordered_map<void*, DeviceAllocationOwner> device_allocation_owners;
inline std::uint64_t device_allocation_generation{};
inline thread_local std::shared_ptr<DeviceResourceLedger> active_device_resource_ledger;
inline thread_local std::shared_ptr<DeviceCapacityPromise> active_device_capacity_promise;
inline thread_local std::size_t native_device_resource_scope_depth{};
inline thread_local bool device_resource_unbind_pending{};

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
  device_resource_unbind_pending = false;
  return true;
}

/** End a binding after the last native borrower returns, even if its original
 * observer ends first. A still-live outer binding is otherwise preserved. */
inline void unbind_device_resource_ledger() noexcept {
  if (native_device_resource_scope_depth) {
    device_resource_unbind_pending = true;
    return;
  }
  if (!active_device_resource_ledger) return;
  {
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    active_device_resource_ledger->active = false;
  }
  active_device_resource_ledger.reset();
  device_resource_unbind_pending = false;
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
    std::uint64_t observation_generation{};
    if (!native_device_resource_scope_depth && !cpu_resource_observation.active) {
      observation_generation = next_cpu_resource_observation_generation();
      if (!observation_generation)
        throw std::overflow_error("CPU resource observation generation exhausted");
    }
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
      if (observation_generation) {
        previous_observation_ = cpu_resource_observation;
        cpu_resource_observation = {true, 0, 0, 1};
        cpu_resource_observation.generation = observation_generation;
        observation_generation_ = observation_generation;
      }
    }
    ++native_device_resource_scope_depth;
  }
  DeviceResourceScope(const DeviceResourceScope&) = delete;
  DeviceResourceScope& operator=(const DeviceResourceScope&) = delete;
  ~DeviceResourceScope() {
    if (!ledger_) return;
    if (observation_generation_ && cpu_resource_observation.generation == observation_generation_) {
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
    if (owns_binding_ || (!native_device_resource_scope_depth && device_resource_unbind_pending))
      unbind_device_resource_ledger();
  }

 private:
  std::shared_ptr<DeviceResourceLedger> ledger_;
  bool owns_binding_{};
  std::uint64_t observation_generation_{};
  CpuResourceObservation previous_observation_{};
};

/** Remaining unpromised capacity. Call only while holding the registry mutex.
 * Subtraction order avoids overflow even when checking a corrupted inventory. */
inline std::size_t device_resource_unpromised(const DeviceResourceLedger& ledger) noexcept {
  if (ledger.live > ledger.limit || ledger.provider_reserved > ledger.limit - ledger.live) return 0;
  const auto available = ledger.limit - ledger.live - ledger.provider_reserved;
  return ledger.capacity_reserved <= available ? available - ledger.capacity_reserved : 0;
}

/** Convert promised capacity into a numeric/provider charge under the same lock
 * as that charge. An owner cannot exceed its declared bound by borrowing spare
 * capacity. Unscoped allocations cannot consume any owner's future promise. */
inline bool device_resource_claim(const std::shared_ptr<DeviceResourceLedger>& ledger,
                                  const std::shared_ptr<DeviceCapacityPromise>& promise,
                                  std::size_t bytes) noexcept {
  if (promise) {
    if (promise->ledger != ledger || !promise->open || bytes > promise->remaining) return false;
    promise->remaining -= bytes;
    ledger->capacity_reserved -= bytes;
    return true;
  }
  return bytes <= device_resource_unpromised(*ledger);
}

/** Restore a retired charge for rebuild/replay while its owner is alive. After
 * close, only actual retained allocations remain charged. Requires the lock. */
inline void device_resource_restore(const std::shared_ptr<DeviceCapacityPromise>& promise,
                                    std::size_t bytes) noexcept {
  if (!promise || !promise->open) return;
  promise->remaining += bytes;
  promise->ledger->capacity_reserved += bytes;
}

/** Reserve a proven complete native endpoint bound before constructing owners.
 * The bound must include all retained and future consumers and rebuild peaks;
 * this class does not derive it from observed live/peak bytes. A caller-selected
 * external inventory alone is not evidence of such a bound. Scope only that
 * owner's mandatory construction/replay, leaving optional storage unscoped.
 * Destruction returns unused capacity; live allocations retain their charges.
 */
class DeviceCapacityReservation {
 public:
  DeviceCapacityReservation(std::shared_ptr<DeviceResourceLedger> ledger, std::size_t bytes)
      : promise_(std::make_shared<DeviceCapacityPromise>()) {
    if (!ledger) throw std::invalid_argument("capacity reservation requires a device ledger");
    promise_->ledger = std::move(ledger);
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    if (bytes > device_resource_unpromised(*promise_->ledger)) throw std::bad_alloc();
    promise_->remaining = bytes;
    promise_->ledger->capacity_reserved += bytes;
  }
  DeviceCapacityReservation(const DeviceCapacityReservation&) = delete;
  DeviceCapacityReservation& operator=(const DeviceCapacityReservation&) = delete;
  /** Optional optimization admission distinguishes a capacity miss from a
   * host allocation failure. The latter still propagates; a capacity miss
   * returns no promise and leaves the incumbent unpromised path available.
   * Test and charge are atomic with other owners sharing this ledger. */
  static std::unique_ptr<DeviceCapacityReservation> try_create(
      std::shared_ptr<DeviceResourceLedger> ledger, std::size_t bytes) {
    auto result = std::make_unique<DeviceCapacityReservation>(std::move(ledger), 0);
    bool admitted = false;
    {
      std::lock_guard<std::mutex> lock(device_resource_mutex);
      if (bytes <= device_resource_unpromised(*result->promise_->ledger)) {
        result->promise_->remaining = bytes;
        result->promise_->ledger->capacity_reserved += bytes;
        admitted = true;
      }
    }
    // Destroy rejected metadata after unlocking: the ordinary reservation
    // destructor owns that lock too, even for an empty promise.
    return admitted ? std::move(result) : nullptr;
  }
  ~DeviceCapacityReservation() {
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    promise_->ledger->capacity_reserved -= promise_->remaining;
    promise_->remaining = 0;
    promise_->open = false;
  }

 private:
  friend class DeviceCapacityScope;
  std::shared_ptr<DeviceCapacityPromise> promise_;
};

/** Bind a promise together with its native ledger. Same-owner nesting retains
 * the existing binding/counters; borrowing a different promise is explicit.
 * The previous promise is restored before the native ledger scope unwinds. */
class DeviceCapacityScope {
 public:
  explicit DeviceCapacityScope(const DeviceCapacityReservation& reservation)
      : DeviceCapacityScope(&reservation) {}
  /** A prepared owner without a promise keeps the current ledger but cannot
   * borrow a different owner's active promise through a nested call. */
  explicit DeviceCapacityScope(const DeviceCapacityReservation* reservation)
      : binding_(reservation ? reservation->promise_->ledger : active_device_resource_ledger),
        previous_(active_device_capacity_promise) {
    active_device_capacity_promise = reservation ? reservation->promise_ : nullptr;
  }
  DeviceCapacityScope(const DeviceCapacityScope&) = delete;
  DeviceCapacityScope& operator=(const DeviceCapacityScope&) = delete;
  ~DeviceCapacityScope() { active_device_capacity_promise = std::move(previous_); }

 private:
  DeviceResourceScope binding_;
  std::shared_ptr<DeviceCapacityPromise> previous_;
};

/** Withhold the mandatory promise while preparing explicitly admitted optional
 * storage. The ledger remains bound; allocations can consume only unpromised
 * capacity. This scope supplies no model bound or optional-admission authority. */
class DeviceUnpromisedCapacityScope {
 public:
  DeviceUnpromisedCapacityScope()
      : binding_(active_device_resource_ledger), previous_(active_device_capacity_promise) {
    active_device_capacity_promise.reset();
  }
  DeviceUnpromisedCapacityScope(const DeviceUnpromisedCapacityScope&) = delete;
  DeviceUnpromisedCapacityScope& operator=(const DeviceUnpromisedCapacityScope&) = delete;
  ~DeviceUnpromisedCapacityScope() { active_device_capacity_promise = std::move(previous_); }

 private:
  DeviceResourceScope binding_;
  std::shared_ptr<DeviceCapacityPromise> previous_;
};

/** Remaining request capacity, including retained native provider reservations.
 * This is shape-only admission, not a device/context/RSS measurement. */
inline std::size_t device_resource_available(int device) {
  const auto ledger = active_device_resource_ledger;
  if (!ledger) return std::numeric_limits<std::size_t>::max();
  if (device != ledger->device) throw std::invalid_argument("resource request/device mismatch");
  std::lock_guard<std::mutex> lock(device_resource_mutex);
  const auto& promise = active_device_capacity_promise;
  if (promise) return promise->ledger == ledger && promise->open ? promise->remaining : 0;
  return device_resource_unpromised(*ledger);
}

/** Reserve a retained provider allowance without inventing a numeric allocation.
 * The lease follows the actual native owner, including release on another host
 * thread or after the observation handle has closed. Destroy the provider before
 * releasing its allowance. No CUDA operation is needed for admission. */
class DeviceProviderAllowance {
 public:
  DeviceProviderAllowance(int device, std::size_t bytes)
      : ledger_(active_device_resource_ledger), promise_(active_device_capacity_promise) {
    if (!ledger_ || !bytes) return;
    if (device != ledger_->device)
      throw std::invalid_argument("provider allowance/device mismatch");
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    if (!device_resource_claim(ledger_, promise_, bytes)) {
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
    device_resource_restore(promise_, bytes_);
  }

 private:
  std::shared_ptr<DeviceResourceLedger> ledger_;
  std::shared_ptr<DeviceCapacityPromise> promise_;
  std::size_t bytes_{};
};

}  // namespace generativeqc::runtime

#endif
