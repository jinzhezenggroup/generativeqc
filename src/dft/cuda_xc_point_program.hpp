#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "dft/cuda_xc_point_value.hpp"
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

/** Paired [2,5,points] physical features from the resident geometry producer.
 * Unlike the SCF coefficient ABI, this retains both physical spin derivatives.
 * The caller owns points output slots and both sticky error words, and keeps
 * all buffers alive until the borrowed stream finishes consuming the tile.
 */
using CudaXcStationaryPointLauncher = void (*)(cudaStream_t, const double*, std::size_t,
                                               std::size_t, DevicePointValue*, int*, const int*);

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
  // The aggregate expression identity binds both CUDA spin layouts. CPU
  // descriptors expose individual identities, so compare like contracts at
  // admission instead of comparing the aggregate hash with its polarized part.
  // Restricted is separate only for the explicit total-density LDA program.
  const char* polarized_expression_identity{};
  const char* restricted_expression_identity{};
  CudaXcStationaryPointLauncher launch_stationary{};
  const char* polarized_executable_identity{};
  const char* restricted_executable_identity{};

  bool matches_cpu_science(std::string_view polarized, std::string_view restricted, unsigned mask,
                           unsigned version) const noexcept {
    return expression_identity && *expression_identity && polarized_expression_identity &&
           !polarized.empty() && polarized == polarized_expression_identity &&
           restricted == (restricted_expression_identity ? restricted_expression_identity : "") &&
           mask == ingredient_mask && version == domain_version;
  }
};

/** Prepared execution binding shared by curated, composed and automatic XC.
 * Resolve both launchers together so a later policy change cannot pair a batch
 * consumer with different point mathematics. A null batch callback means that
 * production batching is unqualified, even if a raw AOT batch entry exists.
 * The optional automatic descriptor retains its own scientific provenance;
 * curated method/domain identity continues to belong to the prepared KS model.
 */
struct CudaXcPointBinding {
  CudaXcPointLauncher launch{};
  CudaXcPointBatchLauncher launch_batch{};
  const CudaXcPointProgram* automatic_program{};
};

}  // namespace generativeqc::dft
