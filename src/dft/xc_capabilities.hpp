#pragma once

#include <cstdint>

namespace generativeqc::dft {

/** Admission status for reusable CUDA XC mechanisms.
 *
 * Unavailable means the required implementation/provider contract is absent.
 * QualificationRequired means the mechanism is structurally reusable but has
 * not passed the endpoint/numerical gates needed for production admission.
 * Qualified is the only state that may enable a fast path.
 */
enum class CudaXcCapability : std::uint8_t {
  Unavailable = 0,
  QualificationRequired = 1,
  Qualified = 2,
};

constexpr bool cuda_xc_capability_qualified(CudaXcCapability capability) noexcept {
  return capability == CudaXcCapability::Qualified;
}

struct CudaXcFastPathCapabilities {
  CudaXcCapability component_scaling{CudaXcCapability::Unavailable};
  CudaXcCapability mixed_ao_precision{CudaXcCapability::Unavailable};
  CudaXcCapability mixed_density_precision{CudaXcCapability::Unavailable};
  CudaXcCapability response{CudaXcCapability::Unavailable};
  CudaXcCapability graph_replay{CudaXcCapability::Unavailable};
};

}  // namespace generativeqc::dft
