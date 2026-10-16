#pragma once

#include <cuda_runtime_api.h>

#include <utility>

#include "methods/dft_method.hpp"
#include "response/cuda_cpks.hpp"

namespace generativeqc::response {

/** Bind CPKS to an actual prepared batch, not caller-supplied identity labels.
 * The enclosing endpoint must retain/exclusively use the batch for this object's
 * entire lifetime and consume the response synchronously. This adapter neither
 * owns the batch nor authorizes an induced-Fock provider. Its callback reborrows
 * the exact live frame and checks every pointer, stream, dimension and device at
 * the solver's entry/publication gates, without a detached numerical export.
 */
class CudaKsCpksReference {
 public:
  CudaKsCpksReference(methods::PreparedBatch& batch, std::size_t item,
                      dft::CudaKsFinalStateToken expected)
      : batch_(batch), item_(item), expected_(std::move(expected)) {}
  CudaKsCpksReference(const CudaKsCpksReference&) = delete;
  CudaKsCpksReference& operator=(const CudaKsCpksReference&) = delete;
  CudaKsCpksReference(CudaKsCpksReference&&) = delete;
  CudaKsCpksReference& operator=(CudaKsCpksReference&&) = delete;

  /** Perform/reuse the owner's physical/canonical scalar-only proof. */
  generativeqc_status prepare(std::string& detail) {
    ready_ = false;
    frame_ = {};
    const auto status = methods::detail::dft_cuda_resident_orbital_frame(batch_, item_, expected_,
                                                                         frame_, detail, true);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    int device = -1;
    if (!frame_ || cudaGetDevice(&device) != cudaSuccess || device != frame_.device_id) {
      frame_ = {};
      detail = "native CPKS frame does not belong to the current CUDA device";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    ready_ = true;
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  /** Metadata only. An unprepared or revoked adapter returns an invalid shape. */
  CudaCpksShape shape() const noexcept {
    if (!ready_) return {};
    return {frame_.nao, frame_.spins, {frame_.occupied[0], frame_.occupied[1]}};
  }

  /** Borrow the authenticated native panels and a non-owning callback context.
   * Keep this nonmovable adapter and its batch alive until the solve returns.
   * The returned value alone is not an independent state/lifetime proof.
   */
  CudaCpksReference reference() noexcept {
    if (!ready_) return {};
    return {frame_.coefficients,
            frame_.orbital_energies,
            CudaCpksLayout::column_major,
            reinterpret_cast<cudaStream_t>(frame_.stream),
            authenticate,
            this};
  }

 private:
  static cudaError_t authenticate(void* context) noexcept {
    auto& self = *static_cast<CudaKsCpksReference*>(context);
    if (!self.ready_) return cudaErrorInvalidResourceHandle;
    try {
      dft::CudaKsResidentOrbitalFrameBinding current;
      std::string detail;
      int device = -1;
      const auto status = methods::detail::dft_cuda_resident_orbital_frame(
          self.batch_, self.item_, self.expected_, current, detail);
      if (status == GENERATIVEQC_STATUS_SUCCESS && current == self.frame_ &&
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
  dft::CudaKsFinalStateToken expected_;
  dft::CudaKsResidentOrbitalFrameBinding frame_{};
  bool ready_{};
};

}  // namespace generativeqc::response
