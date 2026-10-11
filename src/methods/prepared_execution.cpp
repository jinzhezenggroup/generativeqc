#include "methods/method.hpp"

namespace generativeqc::methods {
namespace {
runtime::CompiledExecutionBinding owner_binding(const void* owner, generativeqc_backend backend,
                                                int device) {
  // The retained method owns immutable scientific controls. Its exact source
  // comparisons and #929 generations remain authoritative; this binding only
  // prevents crossing an owner/target lifetime and is not an artifact cache key.
  return {"native-owned-method-plan", device, nullptr, owner, nullptr, static_cast<int>(backend)};
}
}  // namespace

PreparedCalculation::PreparedCalculation(generativeqc_backend backend, int device) {
  lifecycle_.bind(owner_binding(this, backend, device));
}

Result PreparedCalculation::execute(bool compute_forces) {
  if (lifecycle_.failed()) reset_execution();
  lifecycle_.begin();
  try {
    invalidate_result();
    auto result = execute_impl(compute_forces);
    if (result.convergence.converged) {
      lifecycle_.mark_success();
    } else {
      lifecycle_.mark_failure("endpoint did not converge");
    }
    return result;
  } catch (...) {
    lifecycle_.abort();
    try {
      invalidate_result();
    } catch (...) {
      // Cleanup cannot replace the numerical/driver/allocation exception.
    }
    throw;
  }
}

void PreparedCalculation::reset_execution() {
  if (lifecycle_.running() || lifecycle_.closed())
    throw std::logic_error("cannot reset a running or closed prepared calculation");
  invalidate_result();
  reset_execution_impl();
  if (lifecycle_.failed())
    lifecycle_.recover();
  else {
    const auto binding = lifecycle_.binding();
    lifecycle_.invalidate();
    lifecycle_.bind(binding);
  }
}

void PreparedCalculation::close_execution() {
  if (lifecycle_.closed()) return;
  if (lifecycle_.running()) throw std::logic_error("cannot close a running prepared calculation");
  invalidate_result();
  lifecycle_.close();
}

void PreparedCalculation::reject_execution() noexcept {
  if (lifecycle_.running() || lifecycle_.closed()) return;
  if (!lifecycle_.failed()) lifecycle_.abort();
  try {
    invalidate_result();
  } catch (...) {
  }
}

PreparedBatch::PreparedBatch(generativeqc_backend backend, int device) {
  lifecycle_.bind(owner_binding(this, backend, device));
}

std::vector<BatchItemResult> PreparedBatch::execute(const Coordinates& coordinates,
                                                    bool compute_forces) {
  if (lifecycle_.failed()) reset_execution();
  lifecycle_.begin();
  try {
    invalidate_result();
    auto results = execute_impl(coordinates, compute_forces);
    if (results.size() != size())
      throw std::logic_error("prepared batch returned an incomplete result fleet");
    // A completed ragged dispatch is not a claim that every item succeeded.
    // Item status, publication generations and last-good seeds stay native.
    lifecycle_.mark_success();
    return results;
  } catch (...) {
    lifecycle_.abort();
    try {
      invalidate_result();
    } catch (...) {
    }
    throw;
  }
}

void PreparedBatch::reset_execution() {
  if (lifecycle_.running() || lifecycle_.closed())
    throw std::logic_error("cannot reset a running or closed prepared batch");
  invalidate_result();
  clear_warm_starts();
  if (lifecycle_.failed())
    lifecycle_.recover();
  else {
    const auto binding = lifecycle_.binding();
    lifecycle_.invalidate();
    lifecycle_.bind(binding);
  }
}

void PreparedBatch::close_execution() {
  if (lifecycle_.closed()) return;
  if (lifecycle_.running()) throw std::logic_error("cannot close a running prepared batch");
  invalidate_result();
  lifecycle_.close();
}

void PreparedBatch::reject_execution() noexcept {
  if (lifecycle_.running() || lifecycle_.closed()) return;
  if (!lifecycle_.failed()) lifecycle_.abort();
  try {
    invalidate_result();
  } catch (...) {
  }
}
}  // namespace generativeqc::methods
