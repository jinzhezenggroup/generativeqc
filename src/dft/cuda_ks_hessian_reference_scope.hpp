#pragma once

#include <utility>

namespace generativeqc::dft {

/** Identifies the synchronous cold energy solve owned by a dense Hessian.
 * This is authority to try a phase-local cache, not a change to the SCF model
 * or a promise of additional device capacity. Ordinary batch calls never
 * enter this scope. */
class CudaKsHessianReferenceScope {
 public:
  CudaKsHessianReferenceScope() noexcept : previous_(std::exchange(active_, true)) {}
  CudaKsHessianReferenceScope(const CudaKsHessianReferenceScope&) = delete;
  CudaKsHessianReferenceScope& operator=(const CudaKsHessianReferenceScope&) = delete;
  ~CudaKsHessianReferenceScope() { active_ = previous_; }

  static bool active() noexcept { return active_; }

 private:
  inline static thread_local bool active_{};
  bool previous_{};
};

}  // namespace generativeqc::dft
