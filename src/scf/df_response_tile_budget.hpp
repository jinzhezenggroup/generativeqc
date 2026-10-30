#pragma once

#include <cstddef>

namespace generativeqc::scf {

/** Recycle an authenticated dead projection only under response pressure.
 * Independent owned scratch is faster when its preferred panel already fits.
 * The retained factors must always fit separately, and the borrowed interval
 * must hold the algorithm's irreducible scratch. The caller bounds its panel
 * by that interval and retains an ordinary fallback if allocation still fails.
 * Sizes use one common unit (bytes or elements); subtraction avoids overflow.
 */
inline bool df_recycled_projection_preferred(std::size_t available, std::size_t retained,
                                             std::size_t minimum_scratch,
                                             std::size_t preferred_scratch,
                                             std::size_t borrowed_capacity) noexcept {
  return retained <= available && minimum_scratch <= borrowed_capacity &&
         preferred_scratch > available - retained;
}

/** Prefer optional work-reducing storage over the last few response columns.
 * workspace_bytes must be nondecreasing on [1, tile] and include every mandatory
 * allocation still to be made. Preserve the original tile if even its smallest
 * form cannot coexist with the optional storage: an optimization must not turn
 * a previously viable response into a capacity failure. This only plans space;
 * the caller must retain its ordinary fallback if the optional allocation fails.
 */
template <class WorkspaceBytes>
std::size_t df_response_tile_with_headroom(std::size_t tile, std::size_t available_bytes,
                                           std::size_t optional_bytes,
                                           WorkspaceBytes workspace_bytes) {
  if (!tile || !optional_bytes || optional_bytes > available_bytes ||
      workspace_bytes(1) > available_bytes - optional_bytes)
    return tile;
  const auto capacity = available_bytes - optional_bytes;
  std::size_t first = 1, last = tile;
  while (first < last) {
    const auto middle = first + (last - first) / 2 + 1;
    if (workspace_bytes(middle) <= capacity)
      first = middle;
    else
      last = middle - 1;
  }
  return first;
}

}  // namespace generativeqc::scf
