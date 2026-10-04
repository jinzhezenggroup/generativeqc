#pragma once

#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>

#include "scf/cuda_batch.hpp"

namespace generativeqc::methods::detail {

/** Retry only a memory-limited downstream phase after releasing its optional
 * executable owner. The detached physical reference remains valid. Completed
 * work/provenance belongs to the caller; a failed callback must unwind its
 * phase-local allocations before this single retry. */
template <class Operation>
decltype(auto) run_with_cuda_reference_budget(scf::CudaRhfBucketPlan** plan, std::size_t budget,
                                              Operation&& operation) {
  const auto run = [&]() -> decltype(auto) {
    const auto retained = plan ? scf::hf_cuda_retained_numeric_bytes(*plan) : 0;
    if (retained >= budget)
      throw std::length_error("retained CUDA RHF plan exhausts the numeric memory budget");
    return operation(budget - retained);
  };
  const auto retire = [&]() noexcept {
    if (!plan || !*plan) return false;
    scf::destroy_rhf_cuda_bucket_plan(std::exchange(*plan, nullptr));
    return true;
  };
  try {
    return run();
  } catch (const std::length_error&) {
    if (!retire()) throw;
  } catch (const std::bad_alloc&) {
    if (!retire()) throw;
  }
  return run();
}

/** Method-layer lifetime owner for the reusable CUDA RHF executable plan.
 *
 * The plan contains arenas, Graphs, provider handles and topology/schedule
 * state. It deliberately contains no scientific claim that a prior converged
 * density/reference is valid for the next geometry.
 */
class CorrelatedCudaReferencePlan {
 public:
  CorrelatedCudaReferencePlan() = default;
  ~CorrelatedCudaReferencePlan() { reset(); }

  CorrelatedCudaReferencePlan(const CorrelatedCudaReferencePlan&) = delete;
  CorrelatedCudaReferencePlan& operator=(const CorrelatedCudaReferencePlan&) = delete;

  CorrelatedCudaReferencePlan(CorrelatedCudaReferencePlan&& other) noexcept
      : plan_(std::exchange(other.plan_, nullptr)) {}

  CorrelatedCudaReferencePlan& operator=(CorrelatedCudaReferencePlan&& other) noexcept {
    if (this != &other) {
      reset();
      plan_ = std::exchange(other.plan_, nullptr);
    }
    return *this;
  }

  scf::CudaRhfBucketPlan** slot() noexcept { return &plan_; }
  const scf::CudaRhfBucketPlan* get() const noexcept { return plan_; }
  std::size_t owned_device_bytes() const noexcept { return scf::hf_cuda_owned_device_bytes(plan_); }
  std::size_t retained_numeric_bytes() const noexcept {
    return scf::hf_cuda_retained_numeric_bytes(plan_);
  }

  void reset() noexcept {
    scf::destroy_rhf_cuda_bucket_plan(plan_);
    plan_ = nullptr;
  }

 private:
  scf::CudaRhfBucketPlan* plan_{};
};

}  // namespace generativeqc::methods::detail
