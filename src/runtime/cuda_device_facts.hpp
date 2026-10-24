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

/** Raise the current CUDA context's stack floor without lowering a live owner's
 * requirement. Serialize native owners' read/raise transactions; CUDA's limit
 * is context-wide, so independent get/set calls can race across host threads.
 * No device selection or context-limit cache is retained. */
cudaError_t cuda_ensure_stack_limit(std::size_t minimum_bytes);

}  // namespace generativeqc::runtime
