#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "dft/xc_capabilities.hpp"

namespace generativeqc::dft {

/** Physical E/vxc launch ABI. Spin is a layout argument, not an AOT variant.
 * Immutable point science is bound before execution; delta is response-only. */
using CudaXcPointLauncher = void (*)(cudaStream_t, const double*, const double*, std::size_t,
                                     std::size_t, double*, double*, int*, std::uint32_t, double,
                                     double, const double*);

/** Compact channel-major tiles, including a final partial tile. An executable
 * batch entry does not itself qualify a production batching schedule. */
using CudaXcPointBatchLauncher = void (*)(cudaStream_t, const double*, const double*, std::size_t,
                                          std::size_t, double*, double*, int*, std::uint32_t,
                                          double, double, std::size_t);

/** Process-lifetime AOT descriptor. All strings/callbacks are library-owned;
 * names are diagnostic, never native mathematical selectors. Ingredient masks
 * use the existing 1/7/15 ABI. Source identity binds science/traversal/launch
 * validation/layout bytes, not an installed cache key or measured resources.
 * Backend/product admission and optional mechanisms remain separate gates. */
struct CudaXcPointProgram {
  const char* identifier{};
  const char* expression_identity{};
  const char* source_identity{};
  std::uint32_t functional_code{}, ingredient_mask{}, domain_version{};
  std::size_t feature_terms{};
  CudaXcPointLauncher launch{};
  CudaXcPointBatchLauncher launch_batch{};
  CudaXcFastPathCapabilities fast_paths{};
};

}  // namespace generativeqc::dft
