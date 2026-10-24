#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace generativeqc::scf {

/** Declared capacities from the topology-aware CUDA allocator and ECP owners.
 * This metadata-only shape neither constructs a provider nor admits execution.
 * Host staging is the allocator's second output, not a measured live peak.
 */
struct CudaHfResourceShape {
  std::uint64_t atoms, orbital_aos, cartesian_aos, shells, primitives, spins;
  std::uint64_t arena, host_staging, ecp_workspace;
};

namespace cuda_hf_capacity {
inline std::uint64_t sum(std::initializer_list<std::uint64_t> values) {
  constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  std::uint64_t result = 0;
  for (const auto value : values) {
    if (value > maximum - result) throw std::overflow_error("CUDA HF capacity sum overflows");
    result += value;
  }
  return result;
}

inline std::uint64_t product(std::uint64_t first, std::uint64_t second) {
  constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (first > maximum || second > maximum || (second && first > maximum / second))
    throw std::overflow_error("CUDA HF capacity product overflows");
  return first * second;
}
}  // namespace cuda_hf_capacity

/** Arena, retained host bound and ECP scratch for one direct-HF bucket.
 * Four staging copies cover growth, HostBatch packing, cached topology and a
 * cold retry retained beside a warm bucket. Pair-squared metadata bounds the
 * existing PSSS route; this is unchanged conservative capacity, not allocation
 * accounting. Every intermediate belongs to the planner's signed-int64 domain.
 */
inline std::array<std::uint64_t, 3> cuda_hf_item_resources(const CudaHfResourceShape& shape) {
  using namespace cuda_hf_capacity;
  for (const auto value :
       {shape.atoms, shape.orbital_aos, shape.cartesian_aos, shape.shells, shape.primitives,
        shape.spins, shape.arena, shape.host_staging, shape.ecp_workspace})
    product(value, 1);
  if (!shape.atoms || !shape.orbital_aos || shape.cartesian_aos < shape.orbital_aos ||
      !shape.shells || shape.primitives < shape.shells || (shape.spins != 1 && shape.spins != 2))
    throw std::invalid_argument("invalid CUDA HF resource shape");
  const auto pairs = product(shape.shells, sum({shape.shells, 1})) / 2;
  const auto metadata =
      product(2048, sum({1, shape.atoms, shape.shells, shape.primitives, shape.orbital_aos,
                         shape.cartesian_aos, pairs, product(pairs, pairs)}));
  const auto matrices =
      product(product(512, shape.spins), product(shape.orbital_aos, shape.orbital_aos));
  const auto host = sum({product(4, shape.host_staging), metadata, matrices,
                         product(384, shape.atoms), shape.ecp_workspace});
  return {shape.arena, host, shape.ecp_workspace};
}

/** Compose retained arenas/host storage and one serialized numerical retry.
 * Preserve max(arena) + max(ECP), even when their maxima belong to different
 * items. Replacing it with max(arena + ECP) silently tightens the old contract.
 * Appending validates all aggregate peaks before changing any fleet state.
 */
class CudaHfFleetResources {
 public:
  std::array<std::uint64_t, 3> append(const CudaHfResourceShape& shape, std::uint64_t copies = 1) {
    using namespace cuda_hf_capacity;
    if (!copies) throw std::invalid_argument("CUDA HF fleet repetition must be positive");
    const auto item = cuda_hf_item_resources(shape);
    const auto resident = sum({inventory_[0], product(copies, item[0])});
    const auto host = sum({inventory_[3], product(copies, item[1])});
    const auto arena = std::max(maximum_arena_, item[0]);
    const auto ecp = std::max(maximum_ecp_, item[2]);
    const auto retry = sum({arena, ecp});
    const auto peak = sum({resident, retry});
    inventory_ = {resident, retry, peak, host};
    maximum_arena_ = arena;
    maximum_ecp_ = ecp;
    return item;
  }

  /** Retained device, serialized retry, complete device peak, retained host. */
  std::array<std::uint64_t, 4> inventory() const noexcept { return inventory_; }

 private:
  std::array<std::uint64_t, 4> inventory_{};
  std::uint64_t maximum_arena_ = 0;
  std::uint64_t maximum_ecp_ = 0;
};
}  // namespace generativeqc::scf
