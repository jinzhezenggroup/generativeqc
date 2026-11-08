#pragma once

#include "residency_cuda.cuh"
#include "residency_df.hpp"

namespace generativeqc::runtime {

/** The metric owner supplies lineage after its real checked host transform.
 * An instance is an explicit source generation, never a pointer/byte match. */
inline cudaError_t df_metric_memcpy_async(const ResidencyExecution& execution, ResidencySite site,
                                          ResidencyPayload payload, void* destination,
                                          const void* source, std::size_t bytes,
                                          cudaMemcpyKind kind, cudaStream_t stream,
                                          std::uint64_t instance = 0,
                                          std::uint64_t dependency = 0) noexcept {
  ResidencyBoundary boundary(
      execution, ResidencyOperationKind::transfer, ResidencyRole::prepare, site, payload,
      kind == cudaMemcpyDeviceToHost ? ResidencyDirection::d2h : ResidencyDirection::h2d, bytes,
      instance, dependency);
  const auto status = cudaMemcpyAsync(destination, source, bytes, kind, stream);
  boundary.finish(status);
  return status;
}

}  // namespace generativeqc::runtime
