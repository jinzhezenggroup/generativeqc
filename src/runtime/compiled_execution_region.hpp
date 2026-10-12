// Shared lifecycle for compiler-qualified prepared execution regions.
#pragma once

#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace generativeqc::runtime {

/** Host-control states, not scientific solve generations or CUDA events. */
enum class CompiledExecutionState : std::uint8_t { Unbound, Bound, Running, Warm, Failed, Closed };

struct CompiledExecutionBinding {
  std::string qualification;
  int device = 0;
  void* stream = nullptr;
  const void* arena = nullptr;
  const void* library = nullptr;
  int backend = -1;

  bool operator==(const CompiledExecutionBinding& other) const {
    return qualification == other.qualification && device == other.device &&
           stream == other.stream && arena == other.arena && library == other.library &&
           backend == other.backend;
  }
};

struct CompiledExecutionMetrics {
  std::uint64_t bindings = 0;
  std::uint64_t invalidations = 0;
  std::uint64_t executions = 0;
  std::uint64_t failures = 0;
  std::uint64_t recoveries = 0;
  std::uint64_t submissions = 0;
};

class CompiledExecutionRegion {
 public:
  const CompiledExecutionMetrics& metrics() const noexcept { return metrics_; }
  const std::string& reason() const noexcept { return reason_; }
  bool bound() const noexcept { return bound_; }
  bool warmed() const noexcept { return warmed_; }
  bool failed() const noexcept { return failed_; }
  bool running() const noexcept { return running_; }
  bool closed() const noexcept { return closed_; }
  CompiledExecutionState state() const noexcept {
    if (closed_) return CompiledExecutionState::Closed;
    if (running_) return CompiledExecutionState::Running;
    if (failed_) return CompiledExecutionState::Failed;
    if (warmed_) return CompiledExecutionState::Warm;
    return bound_ ? CompiledExecutionState::Bound : CompiledExecutionState::Unbound;
  }

  const CompiledExecutionBinding& binding() const noexcept { return binding_; }

  bool matches(const CompiledExecutionBinding& binding) const noexcept {
    return bound_ && binding_ == binding;
  }

  bool bind(CompiledExecutionBinding binding) {
    require_idle();
    if (binding.qualification.empty())
      throw std::invalid_argument("compiled execution qualification is empty");
    if (matches(binding)) return false;
    reason_ = "compiled execution region bound";
    if (bound_) ++metrics_.invalidations;
    binding_ = std::move(binding);
    bound_ = true;
    warmed_ = failed_ = false;
    ++metrics_.bindings;
    return true;
  }

  /** No-throw teardown compatibility. False rejects mutation of a running or
   * closed gate; callers must retire borrowers before retrying. */
  bool invalidate() noexcept {
    if (running_ || closed_) return false;
    if (bound_ || warmed_ || failed_) ++metrics_.invalidations;
    binding_ = {};
    bound_ = warmed_ = failed_ = false;
    try {
      reason_ = "compiled execution region invalidated";
    } catch (const std::bad_alloc&) {
      // Diagnostics must not terminate no-throw invalidation under memory pressure.
      reason_.clear();
    }
    return true;
  }

  /** Begin one consumer transaction. The consumer revokes its existing result
   * leases first and fences borrowed asynchronous storage before completion.
   * This gate neither creates a solve generation nor adds a device fence. */
  void begin() {
    require_bound();
    require_idle();
    if (failed_) throw std::logic_error("failed compiled execution region requires recovery");
    running_ = true;
    ++metrics_.submissions;
  }

  /** Close only the gate; the original backend owner destroys its resources.
   * Closing a running region is illegal because its borrowers may be live. */
  void close() {
    if (closed_) return;
    require_idle();
    invalidate();
    closed_ = true;
  }

  void mark_success() {
    require_bound();
    if (failed_)
      throw std::logic_error("failed compiled execution region must recover before success");
    reason_ = "compiled execution region completed";
    warmed_ = true;
    running_ = false;
    failed_ = false;
    ++metrics_.executions;
  }

  void mark_failure(std::string reason) {
    require_bound();
    failed_ = true;
    running_ = warmed_ = false;
    ++metrics_.failures;
    reason_ = reason.empty() ? "compiled execution region failed" : std::move(reason);
  }

  /** Preserve the original exception, including allocation failures. */
  void abort() noexcept {
    if (!bound_ || closed_) return;
    failed_ = true;
    running_ = warmed_ = false;
    ++metrics_.failures;
    reason_.clear();
  }

  void recover() {
    require_bound();
    require_idle();
    if (!failed_) return;
    reason_ = "compiled execution region recovered; warmup required";
    failed_ = false;
    warmed_ = false;
    ++metrics_.recoveries;
  }

 private:
  void require_idle() const {
    if (closed_) throw std::logic_error("compiled execution region is closed");
    if (running_) throw std::logic_error("compiled execution region is running");
  }

  void require_bound() const {
    if (closed_) throw std::logic_error("compiled execution region is closed");
    if (!bound_) throw std::logic_error("compiled execution region is not bound");
  }

  CompiledExecutionBinding binding_;
  CompiledExecutionMetrics metrics_;
  std::string reason_ = "compiled execution region not bound";
  bool bound_ = false;
  bool warmed_ = false;
  bool failed_ = false;
  bool running_ = false;
  bool closed_ = false;
};

}  // namespace generativeqc::runtime
