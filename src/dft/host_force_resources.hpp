#pragma once

#include <algorithm>
#include <cstddef>

#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft {

/** The native CPU stationary integral consumer's declared source/weight capacity.
 * Provider recurrence temporaries and allocator overhead are not part of this cap.
 */
inline constexpr std::size_t cpu_stationary_force_source_bytes = 64u << 20;

/** Becke logs and partitions, then one log derivative per atom and its average. */
inline std::size_t partition_derivative_elements(std::size_t atoms) {
  using runtime::size_add;
  using runtime::size_mul;
  return size_add(size_mul(3, size_mul(atoms, atoms)), size_mul(5, atoms));
}

/** Numeric buffers owned by the stationary semilocal XC pullback, including
 * the AO tile and whole-grid weight cotangents still live in the Becke phase.
 * Borrowed grid, AO basis and spin D/W are accounted by their owners.
 */
inline std::size_t cpu_stationary_xc_workspace_bytes(std::size_t atoms, std::size_t aos,
                                                     std::size_t points, std::size_t tile) {
  using runtime::size_add;
  using runtime::size_mul;
  const auto ao_tile = size_mul(10, size_mul(aos, std::min(tile, points)));
  const auto gradient_and_jets = size_add(size_mul(6, atoms), size_mul(4, aos));
  return size_mul(sizeof(double), size_add(size_add(ao_tile, size_add(gradient_and_jets, points)),
                                           atoms > 1 ? partition_derivative_elements(atoms) : 0));
}

}  // namespace generativeqc::dft
