#pragma once

#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace generativeqc::dft {
struct CudaGridStorage {
  std::size_t numeric_bytes{}, error_offset{}, workspace_bytes{}, allocation_bytes{};
  static constexpr std::size_t provider_allowance = 96U << 20;
};

/** Pure capacity authority shared by the native grid producer and consumers.
 * This retains the existing arena/alignment/provider contracts; it does not
 * replace live provider/context receipts or authorize a force endpoint. */
inline CudaGridStorage cuda_grid_storage(std::size_t packed_size, std::size_t aos,
                                         std::size_t capacity, std::size_t jets,
                                         std::size_t active_capacity,
                                         const std::size_t* orbital_capacity = nullptr,
                                         std::size_t orbital_tile = 0) {
  const auto product = [](std::size_t left, std::size_t right) {
    if (right && left > std::numeric_limits<std::size_t>::max() / right)
      throw std::overflow_error("CUDA grid storage product overflow");
    return left * right;
  };
  const auto add = [](std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left)
      throw std::overflow_error("CUDA grid storage sum overflow");
    return left + right;
  };
  if (!aos || aos > INT_MAX || !capacity || capacity > INT_MAX || active_capacity > aos ||
      (jets != 1 && jets != 4 && jets != 10 && jets != 20) ||
      (orbital_capacity && (!orbital_tile || orbital_tile > INT_MAX ||
                            orbital_capacity[0] > INT_MAX || orbital_capacity[1] > INT_MAX)))
    throw std::invalid_argument("invalid CUDA grid storage shape");
  const auto active = active_capacity ? active_capacity : aos;
  const auto matrices = product(2, product(aos, aos));
  const auto tile = product(capacity, active);
  auto elements =
      add(add(packed_size, matrices), add(product(16, capacity), product(jets + 8, tile)));
  if (active_capacity)
    elements = add(elements, add(matrices, add(product(4, product(active, active)), active)));
  if (orbital_capacity)
    elements = add(elements, add(product(aos, add(orbital_capacity[0], orbital_capacity[1])),
                                 product(add(active, product(4, capacity)), orbital_tile)));
  const auto numeric = product(8, elements);
  const auto error_offset = product(add(numeric, 255) / 256, 256);
  const auto workspace = add(error_offset, 256);
  return {numeric, error_offset, workspace, add(workspace, 4U << 20)};
}
}  // namespace generativeqc::dft
