#pragma once

#include <cuda_runtime_api.h>

#include "runtime/cuda_target_info.hpp"

namespace generativeqc::runtime {

/** Query the resource facts used by native contexts and compiler profiles.
 *
 * Every call reads the currently visible device. No ordinal/resource cache is
 * retained across context owners. On NVIDIA/Linux, selected runtime attributes
 * and optional driver name/memory queries avoid expensive unrelated properties.
 * Providers without that complete route retain cudaGetDeviceProperties.
 * Failure clears both outputs; callers never receive a partially qualified GPU.
 */
cudaError_t cuda_device_facts(int device, CudaTargetInfo& target, char (&name)[256]) noexcept;

}  // namespace generativeqc::runtime
