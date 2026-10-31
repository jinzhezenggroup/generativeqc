// Host-only CUDA/Context dependency seam. Never evaluates a TensorIR kernel.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "metrics.hpp"
#include "runtime/cuda_device_scope.hpp"

enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
struct DeviceRange {
  const void* owner;
  std::uintptr_t begin;
  size_t bytes;
  int* status;
};
inline std::vector<DeviceRange> device_ranges;
inline bool device_bytes(const void* pointer, size_t bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  return std::any_of(device_ranges.begin(), device_ranges.end(), [&](const DeviceRange& range) {
    return address >= range.begin && address - range.begin <= range.bytes &&
           bytes <= range.bytes - (address - range.begin);
  });
}
enum class WorkKind { copy, fill };
struct PendingWork {
  WorkKind kind;
  void* to;
  const void* from;
  size_t bytes;
  int value;
  size_t sequence;
};
struct EventRecord {
  int stream;
  size_t watermark;
  bool recorded;
};
inline std::vector<PendingWork> pending;
inline std::map<int, EventRecord> events;
inline size_t submitted = 0;
inline int next_event = 0;
inline int failure = 0, synchronizations = 0;
inline int status_marker = 0;
struct CopyObservation {
  int kind;
  size_t bytes;
  std::uintptr_t source, target;
};
inline std::vector<CopyObservation> copies;
inline std::vector<int> work_devices;
inline int cudaMemcpyAsync(void* to, const void* from, size_t bytes, int kind, int stream) {
  work_devices.push_back(host_current_device);
  if (stream != 73) throw std::runtime_error("wrong owner stream");
  copies.push_back(
      {kind, bytes, reinterpret_cast<std::uintptr_t>(from), reinterpret_cast<std::uintptr_t>(to)});
  if (bytes && (!to || !from)) return 7;
  const bool dst_device = device_bytes(to, bytes), src_device = device_bytes(from, bytes);
  const bool roles_match = kind == cudaMemcpyHostToDevice     ? dst_device && !src_device
                           : kind == cudaMemcpyDeviceToHost   ? !dst_device && src_device
                           : kind == cudaMemcpyDeviceToDevice ? dst_device && src_device
                                                              : false;
  if (!roles_match) return 7;
  if (failure == 1) return 7;
  pending.push_back({WorkKind::copy, to, from, bytes, 0, ++submitted});
  return 0;
}
inline int finish_work(size_t watermark) {
  work_devices.push_back(host_current_device);
  ++synchronizations;
  if (failure == 2) {
    pending.clear();
    return 7;
  }
  for (const auto& work : pending) {
    if (work.sequence > watermark) continue;
    if (work.kind == WorkKind::copy) {
      // Inject a transport marker after preceding queued resets, before readback.
      // This dependency fault never evaluates a TensorIR arithmetic operation.
      if (status_marker)
        for (const auto& range : device_ranges) *range.status = status_marker;
      std::memcpy(work.to, work.from, work.bytes);
    } else
      std::memset(work.to, work.value, work.bytes);
  }
  pending.erase(std::remove_if(pending.begin(), pending.end(),
                               [=](const PendingWork& work) { return work.sequence <= watermark; }),
                pending.end());
  return 0;
}
inline int cudaStreamSynchronize(int stream) {
  if (stream != 73) throw std::runtime_error("wrong owner stream");
  return finish_work(submitted);
}
inline int cudaMemsetAsync(void* p, int value, size_t bytes, int stream) {
  if (stream != 73 || !device_bytes(p, bytes)) return 7;
  pending.push_back({WorkKind::fill, p, nullptr, bytes, value, ++submitted});
  return 0;
}
inline int cudaEventRecord(int event, int stream) {
  const auto entry = events.find(event);
  if (stream != 73 || entry == events.end()) return 7;
  entry->second = {stream, submitted, true};
  return 0;
}
inline int cudaEventSynchronize(int event) {
  const auto record = events.find(event);
  if (record == events.end() || record->second.stream != 73) return 7;
  // A recorded event captures preceding work, not later stream submissions.
  return finish_work(record->second.recorded ? record->second.watermark : 0);
}
inline int cudaEventElapsedTime(float* time, int begin, int end) {
  const auto first = events.find(begin), last = events.find(end);
  if (!time || first == events.end() || last == events.end() || !first->second.recorded ||
      !last->second.recorded)
    return 7;
  const size_t frontier = std::max(first->second.watermark, last->second.watermark);
  if (std::any_of(pending.begin(), pending.end(),
                  [frontier](const PendingWork& work) { return work.sequence <= frontier; }))
    return 7;
  *time = 0;
  return 0;
}

namespace generativeqc_tensor {
inline void error_text(char* p, size_t size, const char* message) {
  if (p && size) {
    std::strncpy(p, message, size - 1);
    p[size - 1] = 0;
  }
}
inline void cuda_check(int status) {
  if (status) throw std::runtime_error("injected CUDA failure");
}
struct Context {
  std::mutex mutex;
  std::vector<unsigned char> storage;
  unsigned char* arena;
  int* error;
  size_t live_device_delta = 0;
  int device = 0, stream = 73, begin = ++next_event, end = ++next_event;
  bool static_ready = true;
  Metrics metrics;
  Context(size_t bytes, size_t error_offset) : storage(bytes, 0xa5), arena(storage.data()) {
    if (error_offset % alignof(int) || error_offset > bytes || sizeof(int) > bytes - error_offset)
      throw std::runtime_error("invalid host status reservation");
    error = new (arena + error_offset) int(0x01010101);
    metrics.owned_device_bytes = bytes;
    device_ranges.push_back({this, reinterpret_cast<std::uintptr_t>(arena), bytes, error});
    events[begin] = {stream, 0, false};
    events[end] = {stream, 0, false};
  }
  ~Context() {
    device_ranges.erase(
        std::remove_if(device_ranges.begin(), device_ranges.end(),
                       [this](const DeviceRange& range) { return range.owner == this; }),
        device_ranges.end());
    events.erase(begin);
    events.erase(end);
  }
  void check_device() {
    if (failure == 3) throw std::runtime_error("injected device mismatch");
    if (host_current_device != device) throw std::runtime_error("wrong current device");
  }
  uint64_t device_delta() const { return live_device_delta; }
  template <class F>
  void section(int, double&, F f) {
    f();
  }
};
}  // namespace generativeqc_tensor
using namespace generativeqc_tensor;

struct MutexHold {
  std::mutex state;
  std::condition_variable changed;
  bool ready = false, holding = false, release = false;
  std::thread worker;
  explicit MutexHold(Context* ctx)
      : worker([this, ctx] {
          std::unique_lock<std::mutex> owner(ctx->mutex, std::defer_lock);
          const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
          while (!owner.try_lock() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          std::unique_lock<std::mutex> gate(state);
          ready = true;
          holding = owner.owns_lock();
          changed.notify_one();
          if (holding) changed.wait_for(gate, std::chrono::seconds(10), [this] { return release; });
          holding = false;
        }) {
    std::unique_lock<std::mutex> gate(state);
    changed.wait_for(gate, std::chrono::seconds(4), [this] { return ready; });
  }
  bool active() {
    std::lock_guard<std::mutex> gate(state);
    return holding;
  }
  ~MutexHold() {
    {
      std::lock_guard<std::mutex> gate(state);
      release = true;
    }
    changed.notify_one();
    worker.join();
  }
};

extern "C" void* host_hold_mutex(void* p) {
  if (!p) return nullptr;
  try {
    auto* hold = new MutexHold(static_cast<Context*>(p));
    if (hold->active()) return hold;
    delete hold;
  } catch (...) {
  }
  return nullptr;
}
extern "C" int host_mutex_held(void* p) { return static_cast<MutexHold*>(p)->active(); }
extern "C" void host_release_mutex(void* p) { delete static_cast<MutexHold*>(p); }
extern "C" size_t host_work_sequence() { return submitted; }

extern "C" void* host_owner(size_t bytes, size_t error_offset) {
  return new Context(bytes, error_offset);
}
extern "C" void host_status_marker(int value) { status_marker = value; }
extern "C" void host_metrics(void* p, size_t provider, size_t prepare, size_t live) {
  auto& ctx = *static_cast<Context*>(p);
  ctx.metrics.provider_retained_bytes = provider;
  ctx.metrics.prepare_device_delta = prepare;
  ctx.live_device_delta = live;
}
extern "C" void host_destroy(void* p) {
  // Negative controls can omit a drain. Discard their fake borrowed pointers.
  pending.clear();
  delete static_cast<Context*>(p);
}
extern "C" void* host_arena(void* p) { return static_cast<Context*>(p)->arena; }
extern "C" void host_capacity(void* p, size_t bytes) {
  static_cast<Context*>(p)->metrics.owned_device_bytes = bytes;
}
extern "C" void host_static_ready(void* p, int ready) {
  static_cast<Context*>(p)->static_ready = ready;
}
extern "C" void host_failure(int mode) { failure = mode; }
extern "C" void host_owner_device(void* p, int device) {
  static_cast<Context*>(p)->device = device;
}
extern "C" int host_selected_device() { return host_current_device; }
extern "C" void host_fail_device(int device) { host_failed_device = device; }
extern "C" void host_clear_work_devices() { work_devices.clear(); }
extern "C" int host_work_on_device(int device) {
  return !work_devices.empty() &&
         std::all_of(work_devices.begin(), work_devices.end(),
                     [device](int observed) { return observed == device; });
}
extern "C" int host_synchronizations() { return synchronizations; }
extern "C" size_t host_pending_work() { return pending.size(); }
extern "C" void host_clear_copy_kinds() { copies.clear(); }
extern "C" int host_copy_matches(int upload) {
  const int expected = upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
  return !copies.empty() &&
         std::all_of(copies.begin(), copies.end(),
                     [expected](const CopyObservation& copy) { return copy.kind == expected; });
}
extern "C" int host_copy_covers(size_t bytes) {
  // Complete contiguous object coverage, independent of SDK transaction shape.
  for (const auto& origin : copies) {
    if (origin.kind != cudaMemcpyDeviceToHost || !origin.bytes ||
        !device_bytes(reinterpret_cast<const void*>(origin.source), bytes))
      continue;
    std::vector<bool> covered(bytes, false);
    for (const auto& copy : copies) {
      if (copy.kind != cudaMemcpyDeviceToHost || copy.source < origin.source ||
          copy.target < origin.target)
        continue;
      const auto offset = copy.source - origin.source;
      if (copy.target - origin.target != offset || offset >= bytes || copy.bytes > bytes - offset)
        continue;
      std::fill(covered.begin() + offset, covered.begin() + offset + copy.bytes, true);
    }
    if (std::all_of(covered.begin(), covered.end(), [](bool byte) { return byte; })) return 1;
  }
  return 0;
}
