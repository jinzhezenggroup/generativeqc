// Host-only CUDA/Context dependency seam. Never evaluates a TensorIR kernel.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "metrics.hpp"

enum { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
struct DeviceRange {
  const void* owner;
  std::uintptr_t begin;
  size_t bytes;
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
inline std::vector<int> copy_kinds;
inline int cudaMemcpyAsync(void* to, const void* from, size_t bytes, int kind, int stream) {
  if (stream != 73) throw std::runtime_error("wrong owner stream");
  copy_kinds.push_back(kind);
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
  ++synchronizations;
  if (failure == 2) {
    pending.clear();
    return 7;
  }
  for (const auto& work : pending) {
    if (work.sequence > watermark) continue;
    if (work.kind == WorkKind::copy)
      std::memcpy(work.to, work.from, work.bytes);
    else
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
  int arithmetic_error = 0x01010101;  // A stale status, nonzero in every byte.
  int* error = &arithmetic_error;
  int device = 0, stream = 73, begin = ++next_event, end = ++next_event;
  bool static_ready = true;
  Metrics metrics;
  explicit Context(size_t bytes) : storage(bytes, 0xa5), arena(storage.data()) {
    metrics.owned_device_bytes = bytes;
    device_ranges.push_back({this, reinterpret_cast<std::uintptr_t>(arena), bytes});
    device_ranges.push_back({this, reinterpret_cast<std::uintptr_t>(error), sizeof(int)});
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
  }
  size_t device_delta() { return 0; }
  template <class F>
  void section(int, double&, F f) {
    f();
  }
};
}  // namespace generativeqc_tensor
using namespace generativeqc_tensor;

extern "C" void* host_owner(size_t bytes) { return new Context(bytes); }
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
extern "C" int host_synchronizations() { return synchronizations; }
extern "C" size_t host_pending_work() { return pending.size(); }
extern "C" void host_clear_copy_kinds() { copy_kinds.clear(); }
extern "C" int host_copy_matches(int upload) {
  const int expected = upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
  return !copy_kinds.empty() && std::all_of(copy_kinds.begin(), copy_kinds.end(),
                                            [expected](int kind) { return kind == expected; });
}
