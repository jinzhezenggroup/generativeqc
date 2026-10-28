#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft::nlc {
struct Vv10CudaDeviceLayout {
  std::size_t point_count{};
  std::size_t tile_points{};
  std::size_t workspace_bytes{};
  bool features{};
  bool geometry{};
  bool mask_zero_weight_rows{};
};

/** Exact caller-owned workspace for resident CUDA VV10/rVV10 execution.
 * Input/output arrays and the numerical-error slot are caller-owned and are
 * not included in workspace_bytes.
 */
inline Vv10CudaDeviceLayout vv10_cuda_device_layout(std::size_t point_count,
                                                    std::size_t tile_points, bool features,
                                                    bool geometry,
                                                    bool mask_zero_weight_rows = false) {
  if (!point_count || !tile_points)
    throw std::invalid_argument("resident VV10 CUDA layout requires nonzero point/tile counts");
  constexpr std::size_t partner_threads = 128;
  const auto partner_blocks = 1 + (point_count - 1) / partner_threads;
  const auto arrays = std::size_t{4} + (features ? 3u : 0u);
  auto slots =
      runtime::size_mul(arrays, point_count, "resident VV10 CUDA workspace extent overflow");
  slots = runtime::size_add(slots, point_count,
                            "resident VV10 CUDA active-partner index extent overflow");
  slots = runtime::size_add(slots, partner_blocks,
                            "resident VV10 CUDA active-partner block extent overflow");
  slots = runtime::size_add(slots, std::size_t{1},
                            "resident VV10 CUDA active-partner count extent overflow");
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  return {point_count,
          std::min(tile_points, point_count),
          runtime::size_mul(slots, sizeof(double), "resident VV10 CUDA workspace byte overflow"),
          features,
          geometry,
          mask_zero_weight_rows};
}

}  // namespace generativeqc::dft::nlc
