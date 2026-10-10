#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace generativeqc::runtime {

/** Optional private source observer, unrelated to the scientific execution route.
 * The synchronous callback runs at an owner boundary, never inside a CUDA API
 * callback. Handles are borrowed only for that call; collectors must record
 * generation IDs rather than treating reused pointer addresses as identities.
 * Thread-local binding deliberately does not claim coverage of other threads. */
using ResidencyObserver = void (*)(const std::uint64_t*, std::size_t, void*);
inline constexpr std::size_t kResidencySourceFields = 10;
enum class ResidencyGraphEvent : std::uint64_t {
  definition = 1,
  launch_begin = 2,
  launch_end = 3,
  destruction = 4,
};
enum class ResidencyGraphRole : std::uint64_t { hf_iteration = 1, hf_post_eigensolver = 2 };

struct ResidencyObservation {
  ResidencyObserver callback{};
  void* context{};
  std::uint64_t errors{};
  bool dispatching{};
};
inline thread_local ResidencyObservation residency_observation;
inline std::atomic<std::uint64_t> residency_graph_generation{1};

/** Reject replacement or mutation during dispatch; callback/context lifetime is
 * owned by the caller and remains valid through a matching thread-local detach. */
inline int bind_residency_observer(ResidencyObserver callback, void* context) noexcept {
  auto& observation = residency_observation;
  if (!callback || observation.callback || observation.dispatching) return 1;
  observation = {callback, context, 0, false};
  return 0;
}

inline int unbind_residency_observer(ResidencyObserver callback, void* context,
                                     std::uint64_t* errors) noexcept {
  auto& observation = residency_observation;
  if (!callback || observation.callback != callback || observation.context != context ||
      observation.dispatching || !errors)
    return 1;
  *errors = observation.errors;
  observation = {};
  return 0;
}

/** Reserve a process-local lifetime ID even without a collector, so attaching
 * after graph construction cannot manufacture a previously observed definition.
 * Zero is an explicit exhausted/unknown identity, never a wrapped generation. */
inline std::uint64_t next_residency_graph_generation() noexcept {
  auto current = residency_graph_generation.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<std::uint64_t>::max()) {
    if (residency_graph_generation.compare_exchange_weak(current, current + 1,
                                                         std::memory_order_relaxed))
      return current;
  }
  return 0;
}

/** Dispatch any versioned source record without allowing observation failures
 * to change scientific execution. Callers own their fixed scalar layouts. */
inline void dispatch_residency_source(const std::uint64_t* values, std::size_t count) noexcept {
  auto& observation = residency_observation;
  if (!observation.callback) return;
  const auto fail = [&]() {
    if (observation.errors != std::numeric_limits<std::uint64_t>::max()) ++observation.errors;
  };
  if (observation.dispatching) {
    fail();
    return;
  }
  observation.dispatching = true;
  try {
    observation.callback(values, count, observation.context);
  } catch (...) {
    fail();
  }
  observation.dispatching = false;
}

/** Emit the fixed v1 scalar ABI: version, event, lifetime, source role, flags,
 * borrowed graph/exec/stream handles, CUDA status, reserved. Observation failure
 * cannot change CUDA status, graph ownership, numerical work, or retry behavior. */
inline void observe_residency_graph(ResidencyGraphEvent event, std::uint64_t generation,
                                    ResidencyGraphRole role, std::uint64_t flags, const void* graph,
                                    const void* executable, const void* stream,
                                    std::uint64_t status = 0) noexcept {
  if (!residency_observation.callback) return;
  const std::array<std::uint64_t, kResidencySourceFields> values{
      1,
      static_cast<std::uint64_t>(event),
      generation,
      static_cast<std::uint64_t>(role),
      flags,
      reinterpret_cast<std::uintptr_t>(graph),
      reinterpret_cast<std::uintptr_t>(executable),
      reinterpret_cast<std::uintptr_t>(stream),
      status,
      0};
  dispatch_residency_source(values.data(), values.size());
}

}  // namespace generativeqc::runtime
