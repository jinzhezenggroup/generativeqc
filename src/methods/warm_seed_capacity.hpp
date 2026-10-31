#pragma once

#include <algorithm>
#include <span>

#include "methods/batch_capabilities.hpp"
#include "runtime/bounded_workspace.hpp"

namespace generativeqc::methods {

struct WarmSeedCapacity {
  std::size_t density_bytes{}, coordinate_bytes{}, total_bytes{}, largest_density_bytes{};
};

/** Read immutable prepared dimensions, never warm_state(), which may export CUDA
 * density buffers even when the caller only wants their sizes. Include every slot
 * so the same bound is valid before SCF, after failures and with frozen seeds. */
inline WarmSeedCapacity warm_seed_capacity(const BatchWarmStarts& owner,
                                           std::span<const std::uint32_t> atom_counts) {
  WarmSeedCapacity result;
  for (std::size_t index = 0; index != atom_counts.size(); ++index) {
    const auto density = runtime::size_mul(owner.warm_density_size(index), sizeof(double));
    const auto coordinates = runtime::size_mul(atom_counts[index], 3 * sizeof(double));
    result.density_bytes = runtime::size_add(result.density_bytes, density);
    result.coordinate_bytes = runtime::size_add(result.coordinate_bytes, coordinates);
    result.largest_density_bytes = std::max(result.largest_density_bytes, density);
  }
  result.total_bytes = runtime::size_add(result.density_bytes, result.coordinate_bytes);
  return result;
}

}  // namespace generativeqc::methods
