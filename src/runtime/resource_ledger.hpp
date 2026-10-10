#ifndef GENERATIVEQC_RUNTIME_RESOURCE_LEDGER_HPP
#define GENERATIVEQC_RUNTIME_RESOURCE_LEDGER_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
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
 * CUDA context, graphs, pools and library-internal allocations are explicitly
 * outside this ledger. Their allowances are withheld by the global planner.
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

}  // namespace generativeqc::runtime

#endif
