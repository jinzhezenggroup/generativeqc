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
