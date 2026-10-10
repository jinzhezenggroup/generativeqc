#pragma once

#include <cuda_runtime_api.h>

#include "residency_boundaries.hpp"

namespace generativeqc::runtime {

/** Observation-only replacement for one async copy. Preserve zero-byte calls,
 * stream, visibility, errors and execution order; do not synchronize or replay. */
inline cudaError_t residency_memcpy_async(const ResidencyExecution& execution, ResidencyRole role,
                                          ResidencySite site, ResidencyPayload payload,
                                          void* destination, const void* source, std::size_t bytes,
                                          cudaMemcpyKind kind, cudaStream_t stream) noexcept {
  const auto direction = kind == cudaMemcpyDeviceToHost     ? ResidencyDirection::d2h
                         : kind == cudaMemcpyHostToDevice   ? ResidencyDirection::h2d
                         : kind == cudaMemcpyDeviceToDevice ? ResidencyDirection::d2d
                                                            : ResidencyDirection::none;
  ResidencyBoundary observation(execution, ResidencyOperationKind::transfer, role, site, payload,
                                direction, bytes);
  const auto status = cudaMemcpyAsync(destination, source, bytes, kind, stream);
  observation.finish(status);
  return status;
}

/** Match the HF input helper's established empty-upload behavior, not raw
 * cudaMemcpyAsync's zero-byte-call behavior. A skipped upload emits no transfer
 * boundary; nonempty uploads preserve exactly one H2D call and its status. */
inline cudaError_t residency_upload(const ResidencyExecution& execution, ResidencyRole role,
                                    ResidencySite site, ResidencyPayload payload, void* destination,
                                    const void* source, std::size_t bytes,
                                    cudaStream_t stream) noexcept {
  if (bytes == 0) return cudaSuccess;
  return residency_memcpy_async(execution, role, site, payload, destination, source, bytes,
                                cudaMemcpyHostToDevice, stream);
}

/** The synchronous API remains synchronous; an observer cannot turn it into an
 * async call, add an extra fence or conceal its implicit-blocking limitations. */
inline cudaError_t residency_memcpy(const ResidencyExecution& execution, ResidencyRole role,
                                    ResidencySite site, ResidencyPayload payload, void* destination,
                                    const void* source, std::size_t bytes,
                                    cudaMemcpyKind kind) noexcept {
  const auto direction = kind == cudaMemcpyDeviceToHost     ? ResidencyDirection::d2h
                         : kind == cudaMemcpyHostToDevice   ? ResidencyDirection::h2d
                         : kind == cudaMemcpyDeviceToDevice ? ResidencyDirection::d2d
                                                            : ResidencyDirection::none;
  ResidencyBoundary observation(execution, ResidencyOperationKind::transfer, role, site, payload,
                                direction, bytes);
  const auto status = cudaMemcpy(destination, source, bytes, kind);
  observation.finish(status);
  return status;
}

/** Count an existing source fence, not a fence inserted by the collector. */
inline cudaError_t residency_stream_synchronize(const ResidencyExecution& execution,
                                                ResidencyRole role, ResidencySite site,
                                                cudaStream_t stream) noexcept {
  ResidencyBoundary observation(execution, ResidencyOperationKind::stream_sync, role, site,
                                ResidencyPayload::none, ResidencyDirection::none, 0);
  const auto status = cudaStreamSynchronize(stream);
  observation.finish(status);
  return status;
}

/** Observe the existing host-blocking event API, not a stream fence or a
 * device-side stream wait. The event and error status pass through unchanged. */
inline cudaError_t residency_event_synchronize(const ResidencyExecution& execution,
                                               ResidencyRole role, ResidencySite site,
                                               cudaEvent_t event) noexcept {
  ResidencyBoundary observation(execution, ResidencyOperationKind::event_sync, role, site,
                                ResidencyPayload::none, ResidencyDirection::none, 0);
  const auto status = cudaEventSynchronize(event);
  observation.finish(status);
  return status;
}

}  // namespace generativeqc::runtime
