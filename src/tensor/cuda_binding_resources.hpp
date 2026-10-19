#pragma once

#include <cstddef>

namespace generativeqc::tensor {

/** Retained preparation envelopes shared by bindings and shape-only planners.
 * These bounds exclude numeric arenas and opaque provider allocations. Keeping
 * them independent of CUDA headers permits context-free CPU SDK queries.
 */
inline constexpr std::size_t kCudaMatrixProductBindingHostBytes = 16U << 10;
inline constexpr std::size_t kCudaStructuredProductBindingHostBytes = 32U << 10;
inline constexpr std::size_t kCudaSymmetricProductBindingHostBytes = 16U << 10;

}  // namespace generativeqc::tensor
