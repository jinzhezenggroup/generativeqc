#pragma once

#include <cstddef>
#include <utility>

#include "scf/cuda_batch.hpp"

namespace generativeqc::methods::detail {

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

  void reset() noexcept {
    scf::destroy_rhf_cuda_bucket_plan(plan_);
    plan_ = nullptr;
  }

 private:
  scf::CudaRhfBucketPlan* plan_{};
};

}  // namespace generativeqc::methods::detail
