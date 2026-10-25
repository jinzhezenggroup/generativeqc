#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace generativeqc::scf {

/** Leaf capacities come from the existing CUDA allocator/provider queries.
 * Only compact topology and declared capacities cross this boundary; composing
 * them successfully never loads CUDA, allocates storage or admits a scientific consumer.
 */
struct CudaKsDeviceResourceItem {
  std::uint64_t cartesian_aos, atoms, shells, primitives;
  std::uint64_t state, xc, coulomb, quadrature_setup, matrix_provider, ecp_workspace;
  bool device_fused_xc;
  std::uint64_t stationary_minimum, stationary_cached;
};

namespace cuda_ks_capacity {
inline std::uint64_t sum(std::initializer_list<std::uint64_t> values) {
  constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  std::uint64_t result = 0;
  for (const auto value : values) {
    if (value > maximum - result) throw std::overflow_error("CUDA KS capacity sum overflows");
    result += value;
  }
  return result;
}

inline std::uint64_t product(std::uint64_t first, std::uint64_t second) {
  constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (first > maximum || second > maximum || (second && first > maximum / second))
    throw std::overflow_error("CUDA KS capacity product overflows");
  return first * second;
}
}  // namespace cuda_ks_capacity

/** State, effective XC, Coulomb, transient peak, quadrature and provider allowance.
 * The value-only one-electron exporter retires its temporary storage before J
 * preparation. Quadrature preparation instead overlaps the new Coulomb owner.
 * A native ECP force reuses the two-grid scratch while all SCF arenas and the
 * stationary consumer remain live. Its standalone one-electron bridge also
 * owns metadata and two compact outputs until that derivative call drains.
 */
inline std::array<std::uint64_t, 6> cuda_ks_device_item_resources(
    const CudaKsDeviceResourceItem& shape) {
  using namespace cuda_ks_capacity;
  if (!shape.cartesian_aos || !shape.atoms || !shape.shells || !shape.primitives)
    throw std::invalid_argument("invalid CUDA KS device resource shape");
  for (const auto value :
       {shape.state, shape.xc, shape.coulomb, shape.quadrature_setup, shape.matrix_provider,
        shape.ecp_workspace, shape.stationary_minimum, shape.stationary_cached})
    static_cast<void>(sum({value}));
  const auto shell_pairs = product(shape.shells, sum({shape.shells, 1})) / 2;
  const auto one_electron =
      product(64, sum({1, shape.atoms, shape.shells, shape.primitives, shape.cartesian_aos,
                       shell_pairs, product(shape.cartesian_aos, shape.cartesian_aos)}));
  auto setup =
      std::max({one_electron, shape.ecp_workspace, sum({shape.coulomb, shape.quadrature_setup})});
  if (shape.ecp_workspace && (shape.stationary_minimum || shape.stationary_cached)) {
    const auto retained = sum({shape.state, shape.device_fused_xc ? shape.xc : 0, shape.coulomb});
    // The existing exporter bound covers the bridge's AO/shell metadata. Its
    // extra Hcore/Pulay outputs coexist with ECP scratch, unlike energy setup.
    const auto force_scratch = sum({one_electron, product(48, shape.atoms), shape.ecp_workspace});
    setup = std::max(setup, sum({retained, force_scratch}));
  }
  return {shape.state,
          shape.device_fused_xc ? shape.xc : 0,
          shape.coulomb,
          setup,
          shape.quadrature_setup,
          shape.matrix_provider};
}

/** Compose retained arenas, opaque-provider allowances and serialized rebuilds.
 * A rebuilt item's retired storage need not coexist with its setup temporary;
 * only transient excess above that item's retained arenas is an additional phase.
 * Both stationary candidates coexist with the fleet during SCF replay. These
 * are declared numeric/provider bounds, not complete allocation or RSS receipts.
 * Appending validates both candidate peaks before changing any fleet state.
 */
class CudaKsDeviceFleetResources {
 public:
  std::array<std::uint64_t, 6> append(const CudaKsDeviceResourceItem& shape,
                                      std::uint64_t copies = 1) {
    using namespace cuda_ks_capacity;
    if (!copies) throw std::invalid_argument("CUDA KS fleet repetition must be positive");
    const auto item = cuda_ks_device_item_resources(shape);
    auto next_totals = totals_;
    for (std::size_t index = 0; index < 3; ++index)
      next_totals[index] = sum({totals_[index], product(copies, item[index])});
    next_totals[3] = sum({totals_[3], product(copies, item[5])});
    const auto retained = sum({item[0], item[1], item[2]});
    const auto extra = std::max(extra_, item[3] > retained ? item[3] - retained : 0);
    const auto minimum = sum({minimum_, product(copies, shape.stationary_minimum)});
    const auto cached = sum({cached_, product(copies, shape.stationary_cached)});
    const auto common =
        sum({next_totals[0], next_totals[1], next_totals[2], next_totals[3], extra});
    const auto minimum_peak = sum({common, minimum});
    const auto cached_peak = sum({common, cached});
    totals_ = next_totals;
    extra_ = extra;
    minimum_ = minimum;
    cached_ = cached;
    minimum_peak_ = minimum_peak;
    cached_peak_ = cached_peak;
    return item;
  }

  /** State/XC/J/provider totals, extra, stationary min/cached, and both peaks. */
  std::array<std::uint64_t, 9> inventory() const noexcept {
    return {totals_[0], totals_[1], totals_[2],    totals_[3],  extra_,
            minimum_,   cached_,    minimum_peak_, cached_peak_};
  }

 private:
  std::array<std::uint64_t, 4> totals_{};
  std::uint64_t extra_{0}, minimum_{0}, cached_{0}, minimum_peak_{0}, cached_peak_{0};
};

}  // namespace generativeqc::scf
