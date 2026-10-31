#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "dft/cuda_grid_storage.hpp"
#include "dft/cuda_stationary_binding.hpp"
#include "dft/nonlocal_correlation/vv10_capacity.hpp"
#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft {
struct CudaStationaryCapacity {
  CudaGridStorage grid;
  std::size_t packed_elements{}, tile_points{}, geometry_lanes{}, geometry_threads{};
  std::size_t stationary_minimum_bytes{}, stationary_cached_bytes{};
  std::size_t minimum_device_bytes{}, cached_device_bytes{};
  /** Optional producer panels/provider belong to the cached consumer tier,
   * not to the emitted stationary arena or the mandatory strict grid layout. */
  std::size_t cached_grid_extra_bytes{}, nonlocal_device_bytes{};
  /** Exact derivative-jet masks are optional. The budget includes host offsets
   * conservatively as well as peak device discovery/retained storage. */
  std::size_t ao_map_bytes{}, ao_map_host_bytes{};
};

/** Stable v1 receipt shared by raw, descriptor and prepared-item queries. */
inline std::array<std::uint64_t, 9> cuda_stationary_capacity_values(
    const CudaStationaryCapacity& capacity) noexcept {
  return {capacity.grid.allocation_bytes,
          capacity.grid.provider_allowance,
          capacity.stationary_minimum_bytes + capacity.nonlocal_device_bytes,
          capacity.stationary_cached_bytes + capacity.cached_grid_extra_bytes +
              capacity.nonlocal_device_bytes,
          capacity.minimum_device_bytes,
          capacity.cached_device_bytes,
          capacity.tile_points,
          capacity.geometry_lanes,
          capacity.geometry_threads};
}

/** Shape-only capacity authority shared by native planning and construction.
 * Uses the statically bound emitted arena polynomial and the unchanged grid
 * layout. It performs no CUDA operation, constructs no scientific owner and
 * never authorizes a model/property capability. SCF/Fock/quadrature owners and
 * unobservable CUDA context storage remain outside this additional consumer. */
inline CudaStationaryCapacity cuda_stationary_capacity(
    const CudaStationaryBinding& binding, std::size_t atoms, std::size_t aos,
    std::size_t primitives, std::size_t grid_points, std::size_t tile_points,
    bool mixed_grid = false, std::size_t nonlocal_points = 0) {
  if (!binding.numeric_bytes || !atoms || !grid_points || !tile_points)
    throw std::invalid_argument("invalid native stationary capacity shape");
  CudaStationaryCapacity capacity;
  if (binding.model.nonlocal_b != 0 && !nonlocal_points) nonlocal_points = grid_points;
  capacity.tile_points = std::min(tile_points, std::max(grid_points, nonlocal_points));
  const auto geometry_lane_bytes = runtime::size_mul(atoms, 18U * sizeof(double));
  capacity.geometry_lanes = std::min({capacity.tile_points, binding.geometry_max_lanes,
                                      binding.geometry_max_scratch_bytes / geometry_lane_bytes});
  capacity.geometry_threads = std::min(capacity.geometry_lanes, binding.geometry_max_threads);
  capacity.stationary_minimum_bytes =
      binding.numeric_bytes(atoms, aos, primitives, capacity.tile_points, 1, binding.spin_blocks,
                            capacity.geometry_lanes, false);
  capacity.stationary_cached_bytes =
      binding.numeric_bytes(atoms, aos, primitives, capacity.tile_points, 1, binding.spin_blocks,
                            capacity.geometry_lanes, true);
  capacity.packed_elements = runtime::size_add(
      runtime::size_mul(atoms, 3),
      runtime::size_add(runtime::size_mul(primitives, 2), runtime::size_mul(aos, 16)));
  capacity.grid =
      cuda_grid_storage(capacity.packed_elements, aos, capacity.tile_points, binding.ao_jets, aos);
  const auto grid_upper =
      runtime::size_add(capacity.grid.allocation_bytes, capacity.grid.provider_allowance);
  capacity.minimum_device_bytes = runtime::size_add(grid_upper, capacity.stationary_minimum_bytes);
  capacity.cached_device_bytes = runtime::size_add(grid_upper, capacity.stationary_cached_bytes);
  // Restore the indexed force consumer for substantial ordinary grid work.
  // Force discovery inspects every required derivative jet at 1e-16; an SCF
  // order-one map cannot authorize dropping order-two force columns. This
  // crossover is a work estimate, independent of device or functional names.
  const auto dense_work = runtime::size_mul(grid_points, runtime::size_mul(aos, aos));
  if ((binding.ao_jets == 4 || binding.ao_jets == 10) && dense_work >= (std::size_t{1} << 35)) {
    const auto tiles = 1 + (grid_points - 1) / capacity.tile_points;
    const auto offsets = runtime::size_mul(runtime::size_add(tiles, 1), sizeof(std::size_t));
    const auto masks = runtime::size_mul(runtime::size_mul(tiles, runtime::size_add(aos, 31) / 32),
                                         sizeof(unsigned));
    const auto map_bytes =
        runtime::size_add(runtime::size_add(runtime::size_mul(2, offsets), masks),
                          runtime::size_mul(aos, sizeof(std::size_t)));
    if (map_bytes <= (16U << 20)) {
      capacity.ao_map_bytes = map_bytes;
      capacity.ao_map_host_bytes = offsets;
      capacity.cached_grid_extra_bytes = map_bytes;
      capacity.cached_device_bytes = runtime::size_add(capacity.cached_device_bytes, map_bytes);
    }
  }
  if (binding.model.nonlocal_b != 0) {
    const auto pair = nlc::vv10_cuda_device_layout(nonlocal_points, tile_points, true, true, true);
    capacity.nonlocal_device_bytes = runtime::size_add(
        pair.workspace_bytes,
        runtime::size_add(
            runtime::size_mul(runtime::size_add(runtime::size_mul(11, nonlocal_points), 1),
                              sizeof(double)),
            3 * sizeof(int)));
    capacity.minimum_device_bytes =
        runtime::size_add(capacity.minimum_device_bytes, capacity.nonlocal_device_bytes);
    capacity.cached_device_bytes =
        runtime::size_add(capacity.cached_device_bytes, capacity.nonlocal_device_bytes);
  }
  if (mixed_grid) {
    const auto mixed_bytes = cuda_grid_fp32_storage(aos, capacity.tile_points).device_upper_bytes;
    capacity.cached_grid_extra_bytes =
        runtime::size_add(capacity.cached_grid_extra_bytes, mixed_bytes);
    capacity.cached_device_bytes = runtime::size_add(capacity.cached_device_bytes, mixed_bytes);
  }
  return capacity;
}
}  // namespace generativeqc::dft
