#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "dft/cuda_grid_storage.hpp"
#include "dft/cuda_stationary_binding.hpp"
#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft {
struct CudaStationaryCapacity {
  CudaGridStorage grid;
  std::size_t packed_elements{}, tile_points{}, geometry_lanes{}, geometry_threads{};
  std::size_t stationary_minimum_bytes{}, stationary_cached_bytes{};
  std::size_t minimum_device_bytes{}, cached_device_bytes{};
};

/** Stable v1 receipt shared by raw, descriptor and prepared-item queries. */
inline std::array<std::uint64_t, 9> cuda_stationary_capacity_values(
    const CudaStationaryCapacity& capacity) noexcept {
  return {capacity.grid.allocation_bytes,
          capacity.grid.provider_allowance,
          capacity.stationary_minimum_bytes,
          capacity.stationary_cached_bytes,
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
inline CudaStationaryCapacity cuda_stationary_capacity(const CudaStationaryBinding& binding,
                                                       std::size_t atoms, std::size_t aos,
                                                       std::size_t primitives,
                                                       std::size_t grid_points,
                                                       std::size_t tile_points) {
  if (!binding.numeric_bytes || !atoms || !grid_points || grid_points % atoms || !tile_points)
    throw std::invalid_argument("invalid native stationary capacity shape");
  CudaStationaryCapacity capacity;
  capacity.tile_points = std::min(tile_points, grid_points);
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
  return capacity;
}
}  // namespace generativeqc::dft
