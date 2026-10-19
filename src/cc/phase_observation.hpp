#pragma once
#include <array>
#include <atomic>
#include <exception>
#include <limits>

#include "cc/phase_observation_contract.hpp"
#include "runtime/residency_observer.hpp"

namespace generativeqc::cc::phase_observation {
// Native implementation only. No borrowed callback/context/code pointer is
// retained here. The existing dispatcher owns the currently live binding.
struct Control {
  std::uint64_t epoch{}, permit{}, parent{}, plan{}, active{}, errors{};
};
inline thread_local Control control;
inline std::atomic<std::uint64_t> source_generation{1};
inline bool enabled() noexcept { return control.permit != 0; }
inline std::uint64_t next_identity() noexcept {
  auto n = source_generation.load(std::memory_order_relaxed);
  while (n != std::numeric_limits<std::uint64_t>::max())
    if (source_generation.compare_exchange_weak(n, n + 1, std::memory_order_relaxed)) return n;
  return 0;
}
inline bool owner_live() noexcept {
  if (!runtime::residency_binding_epoch_admissible() ||
      control.epoch != runtime::residency_binding_epoch.value ||
      !runtime::residency_observation.callback) {
    control.errors = 1;
    return false;
  }
  return true;
}
inline int enable(std::uint64_t contract, Observer callback, void* context,
                  std::uint64_t* permit) noexcept {
  auto& o = runtime::residency_observation;
  if (contract != kPhaseLayout || !callback || !permit || control.permit ||
      !runtime::residency_binding_epoch_admissible() || o.dispatching || o.errors ||
      o.callback != callback || o.context != context)
    return 1;
  const auto token = next_identity();
  if (!token) return 1;
  control = {runtime::residency_binding_epoch.value, token, 0, 0, 0, 0};
  *permit = token;
  return 0;
}
inline int finish(std::uint64_t permit, std::uint64_t* errors) noexcept {
  if (!permit || control.permit != permit || !errors || control.active ||
      runtime::residency_observation.dispatching)
    return 1;
  (void)owner_live();
  *errors = control.errors || runtime::residency_observation.errors;
  control = {};
  return 0;
}

/** Inclusive calls, host-only metadata preparation, GPU submissions and existing
 * waits are distinct sites. No source duration by itself proves GPU completion,
 * pure CPU numerical production, idle time or a causal overlap opportunity. */
class Scope {
 public:
  Scope(Site site, std::uint64_t index = 0, std::uint64_t count = 0, std::uintptr_t stream = 0,
        std::uint64_t dependency = 0) noexcept {
    if (!control.permit) return;
    if (!owner_live() || runtime::residency_observation.dispatching) {
      control.errors = 1;
      return;
    }
    const auto id = next_identity();
    if (!id) {
      control.errors = 1;
      return;
    }
    previous_parent_ = control.parent;
    previous_plan_ = control.plan;
    row_ = {kPhaseLayout,   1,     id,           control.parent, static_cast<std::uint64_t>(site),
            index,          count, control.plan, dependency,     stream,
            control.permit, 0};
    identity_ = id;
    exceptions_ = std::uncaught_exceptions();
    control.parent = id;
    ++control.active;
    active_ = true;
    runtime::dispatch_residency_source(row_.data(), row_.size());
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() { close(); }
  std::uint64_t identity() const noexcept { return identity_; }
  void close(std::uint64_t status = 0) noexcept {
    if (!active_) return;
    if (owner_live()) {
      row_[1] = 2;
      row_[11] = std::uncaught_exceptions() > exceptions_ ? 1 : status;
      runtime::dispatch_residency_source(row_.data(), row_.size());
    }
    control.parent = previous_parent_;
    control.plan = previous_plan_;
    --control.active;
    active_ = false;
  }

 private:
  // Disabled path leaves record storage uninitialized: no ID reservation,
  // record fill, explicit heap/clock/I/O or CUDA operation. Stack/TLS/branch
  // footprint remains real and must not be advertised as whole-library zero.
  std::array<std::uint64_t, kPhaseWords> row_;
  std::uint64_t previous_parent_, previous_plan_;
  int exceptions_;
  bool active_ = false;
  std::uint64_t identity_ = 0;
};

/** Read already-prepared native plan facts once, after existing allocation and
 * preparation fallbacks. This does not choose a plan, probe capacity, allocate,
 * synchronize, query CUDA or change the scientific budget. No retirement or
 * allocation-lifetime claim is attached to this immutable admission snapshot. */
inline void admitted_plan(PlanOwner owner, std::uint64_t naux, std::uint64_t primal_batch,
                          std::uint64_t replay_batch, std::uint64_t route,
                          std::uint64_t maximum_bytes, std::uint64_t numeric_high_water,
                          std::uint64_t device_high_water) noexcept {
  if (!control.permit) return;
  if (!owner_live() || !control.parent || runtime::residency_observation.dispatching) {
    control.errors = 1;
    return;
  }
  const auto id = next_identity();
  if (!id) {
    control.errors = 1;
    return;
  }
  const std::array<std::uint64_t, kAdmissionWords> record{kAdmissionLayout,
                                                          1,
                                                          id,
                                                          control.parent,
                                                          static_cast<std::uint64_t>(owner),
                                                          naux,
                                                          primal_batch,
                                                          replay_batch,
                                                          route,
                                                          maximum_bytes,
                                                          numeric_high_water,
                                                          device_high_water,
                                                          control.permit,
                                                          0};
  runtime::dispatch_residency_source(record.data(), record.size());
  control.plan = id;
}
}  // namespace generativeqc::cc::phase_observation
