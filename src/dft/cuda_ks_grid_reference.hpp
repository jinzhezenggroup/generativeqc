#pragma once

#include <cuda_runtime_api.h>

#include <utility>

#include "dft/cuda_wb97mv_grid.hpp"
#include "methods/dft_method.hpp"

namespace generativeqc::dft {

/** Bind the fixed-grid component to actual immutable prepared-KS storage.
 * The batch and this nonmovable adapter must survive every synchronous call,
 * without concurrent batch access. Reauthentication compares the entire live
 * basis/quadrature/density binding, not detached labels. This adapter proves
 * lifetime and canonical validation only: evaluating WB97M-V at another
 * functional's density does NOT authorize a stationary WB97M-V Hessian.
 * The enclosing physical provider must separately validate the complete model.
 */
class CudaKsGridReference {
 public:
  CudaKsGridReference(methods::PreparedBatch& batch, std::size_t item,
                      CudaKsFinalStateToken expected)
      : batch_(batch), item_(item), expected_(std::move(expected)) {}
  CudaKsGridReference(const CudaKsGridReference&) = delete;
  CudaKsGridReference& operator=(const CudaKsGridReference&) = delete;
  CudaKsGridReference(CudaKsGridReference&&) = delete;
  CudaKsGridReference& operator=(CudaKsGridReference&&) = delete;

  /** Prepare/reuse scalar-only canonical validation, then borrow native storage.
   * Failure clears this adapter; no host basis/grid/density copy is staged. */
  generativeqc_status prepare(std::string& detail) {
    ready_ = false;
    binding_ = {};
    CudaKsResidentOrbitalFrameBinding frame;
    auto status = methods::detail::dft_cuda_resident_orbital_frame(batch_, item_, expected_, frame,
                                                                   detail, true);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status = methods::detail::dft_cuda_resident_grid_reference(batch_, item_, expected_, binding_,
                                                               detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    int device = -1;
    if (!binding_ || binding_.device_id != frame.device_id || binding_.density != frame.density ||
        binding_.nao != frame.nao || binding_.spins != frame.spins ||
        binding_.stream != frame.stream || cudaGetDevice(&device) != cudaSuccess ||
        device != binding_.device_id) {
      binding_ = {};
      detail = "native grid reference does not match the current canonical CUDA frame";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    ready_ = true;
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  /** Metadata is valid only while prepared and not revoked. */
  CudaWb97mvGridShape shape() const noexcept {
    if (!ready_) return {};
    return {binding_.atoms, binding_.primitives, binding_.nao, binding_.point_count,
            binding_.spins};
  }

  /** Borrow storage plus a callback which reborrows the actual live owner. */
  CudaWb97mvGridReference reference() noexcept {
    if (!ready_) return {};
    return {binding_.basis,     binding_.points,
            binding_.weights,   binding_.density,
            binding_.device_id, reinterpret_cast<cudaStream_t>(binding_.stream),
            authenticate,       this};
  }

 private:
  static cudaError_t authenticate(void* context) noexcept {
    auto& self = *static_cast<CudaKsGridReference*>(context);
    if (!self.ready_) return cudaErrorInvalidResourceHandle;
    try {
      CudaKsResidentGridBinding current;
      std::string detail;
      int device = -1;
      const auto status = methods::detail::dft_cuda_resident_grid_reference(
          self.batch_, self.item_, self.expected_, current, detail);
      if (status == GENERATIVEQC_STATUS_SUCCESS && current == self.binding_ &&
          cudaGetDevice(&device) == cudaSuccess && device == current.device_id)
        return cudaSuccess;
    } catch (...) {
      self.ready_ = false;
      return cudaErrorUnknown;
    }
    self.ready_ = false;
    return cudaErrorInvalidResourceHandle;
  }

  methods::PreparedBatch& batch_;
  std::size_t item_;
  CudaKsFinalStateToken expected_;
  CudaKsResidentGridBinding binding_{};
  bool ready_{};
};

}  // namespace generativeqc::dft
