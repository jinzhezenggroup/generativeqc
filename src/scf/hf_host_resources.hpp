#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace generativeqc::scf {

namespace hf_host_resource_detail {
using Bytes = std::uint64_t;
inline constexpr Bytes maximum = std::numeric_limits<std::int64_t>::max();

inline Bytes sum(std::initializer_list<Bytes> values) {
  Bytes result = 0;
  for (const auto value : values) {
    if (value > maximum - result) throw std::overflow_error("CPU HF capacity sum overflows");
    result += value;
  }
  return result;
}

inline Bytes product(std::initializer_list<Bytes> values) {
  Bytes result = 1;
  for (const auto value : values) {
    if (value > maximum || (value && result > maximum / value))
      throw std::overflow_error("CPU HF capacity product overflows");
    result *= value;
  }
  return result;
}
}  // namespace hf_host_resource_detail

/** Compact CPU HF topology, with all auxiliary fields zero for direct HF.
 * ECP workspace comes from the existing two-grid capacity owner. This shape
 * describes numeric/value-metadata bounds, not allocator overhead or RSS.
 */
struct CpuHfHostResourceShape {
  std::uint64_t atoms, orbital_aos, cartesian_aos, spins, maximum_angular, diis_history;
  std::uint64_t auxiliary_aos, auxiliary_cartesian_aos, auxiliary_maximum_angular, ecp_workspace;
};

inline constexpr std::array<const char*, 4> cpu_hf_host_resource_keys{
    "preparation", "iteration", "finalization", "integral_state"};

/** Shared allocation-free sizing for CPU RHF/UHF, direct or density-fitted.
 * The existing DF preparation still retains conventional four-center integrals.
 * Jet derivatives and Cartesian/public tensors coexist during preparation;
 * analytic DF finalization also retains metric response and response tensors.
 * Workspaces are serialized phase maxima, not per-fleet retained capacities.
 * Every intermediate uses the planner's signed-int64 byte domain. Invalid
 * topology or overflow throws before any caller-visible publication.
 */
inline std::array<std::uint64_t, 4> cpu_hf_host_resources(const CpuHfHostResourceShape& shape) {
  using hf_host_resource_detail::product;
  using hf_host_resource_detail::sum;
  if (!shape.atoms || !shape.orbital_aos || shape.cartesian_aos < shape.orbital_aos ||
      (shape.spins != 1 && shape.spins != 2) || shape.maximum_angular > 4 ||
      shape.auxiliary_maximum_angular > 4 ||
      (shape.auxiliary_aos ? shape.auxiliary_cartesian_aos < shape.auxiliary_aos
                           : shape.auxiliary_cartesian_aos || shape.auxiliary_maximum_angular))
    throw std::invalid_argument("CPU HF inventory requires a valid through-g topology");

  const auto orbital_square = product({shape.orbital_aos, shape.orbital_aos});
  const auto cartesian_square = product({shape.cartesian_aos, shape.cartesian_aos});
  const auto derivatives = product({3, shape.atoms});
  const auto components = sum({derivatives, 1});
  const auto orbital_terms =
      sum({product({2, orbital_square}), product({orbital_square, orbital_square})});
  const auto cartesian_terms =
      sum({product({2, cartesian_square}), product({cartesian_square, cartesian_square})});
  const auto conventional =
      sum({product({8, components, orbital_terms}), product({8, derivatives})});
  const auto cartesian =
      sum({product({8, components, cartesian_terms}), product({8, derivatives})});
  /** The Jet allowance includes LP64 vector/value metadata, not just scalars. */
  const auto jet = sum({product({8, components}), 64});
  const auto angular = std::max(shape.maximum_angular, shape.auxiliary_maximum_angular);
  const auto order = std::max({4 * shape.maximum_angular,
                               2 * shape.maximum_angular + shape.auxiliary_maximum_angular,
                               2 * shape.auxiliary_maximum_angular});
  const auto hermite = product({6, angular + 3, angular + 3, 2 * angular + 4});
  /** One primitive quartet's full recurrence box is live at a time; contraction
   * length does not multiply this workspace by the primitive-quartet count. */
  const auto recurrence = product({jet, sum({product({order + 1, order + 1, order + 1, order + 1}),
                                             hermite, 256, derivatives})});
  auto preparation =
      sum({product({jet, cartesian_terms}),
           std::max(recurrence, sum({cartesian, conventional, product({8, orbital_terms})})),
           shape.ecp_workspace});
  auto resident = conventional;
  const auto history = sum({shape.diis_history, 1});
  const auto matrix =
      sum({product({8, shape.spins, orbital_square, sum({64, product({2, history})})}),
           product({16, history, history})});
  auto iteration = sum({resident, matrix});
  auto finalization = iteration;
  if (shape.auxiliary_aos) {
    const auto auxiliary_square = product({shape.auxiliary_aos, shape.auxiliary_aos});
    const auto count = sum({auxiliary_square, product({orbital_square, shape.auxiliary_aos})});
    const auto cartesian_count =
        sum({product({shape.auxiliary_cartesian_aos, shape.auxiliary_cartesian_aos}),
             product({cartesian_square, shape.auxiliary_cartesian_aos})});
    const auto raw = product({8, components, count});
    const auto cartesian_raw = product({8, components, cartesian_count});
    const auto transformed = product({8, orbital_square, shape.auxiliary_aos});
    preparation =
        std::max({preparation,
                  sum({conventional, product({jet, cartesian_count}),
                       std::max(recurrence, sum({cartesian_raw, raw, product({8, count})}))}),
                  sum({conventional, raw, transformed, product({128, auxiliary_square})})});
    resident = sum({resident, raw, transformed});
    iteration = sum(
        {resident, matrix, product({8, sum({shape.auxiliary_aos, product({4, orbital_square})})})});
    finalization = sum({resident, matrix,
                        product({8, sum({product({32, auxiliary_square}),
                                         product({2, orbital_square, shape.auxiliary_aos}),
                                         product({8, derivatives})})})});
  }
  return {preparation, iteration, finalization, resident};
}

/** Metadata retained by each prepared item in addition to its phase workspace.
 * Auxiliary shell/primitive counts must both be zero for a direct-HF item.
 */
struct CpuHfHostResourceItem {
  CpuHfHostResourceShape shape;
  std::uint64_t orbital_shells, orbital_primitives, auxiliary_shells, auxiliary_primitives;
};

/** Allocation-free CPU fleet sizing: sum retained owners, maximize workspace.
 * Homogeneous CLI batches can append a repeated item without materializing it.
 * Each append validates its complete signed-int64 peak before changing state;
 * neither an invalid later item nor overflow can leave a partial aggregate.
 * This is numeric/value-metadata capacity, not admission or allocator receipts.
 */
class CpuHfHostFleetResources {
 public:
  std::array<std::uint64_t, 4> append(const CpuHfHostResourceItem& item, std::uint64_t copies = 1) {
    using hf_host_resource_detail::product;
    using hf_host_resource_detail::sum;
    const auto phases = cpu_hf_host_resources(item.shape);
    if (!copies || !item.orbital_shells || item.orbital_shells > item.shape.orbital_aos ||
        item.orbital_primitives < item.orbital_shells ||
        (item.shape.auxiliary_aos
             ? !item.auxiliary_shells || item.auxiliary_shells > item.shape.auxiliary_aos ||
                   item.auxiliary_primitives < item.auxiliary_shells
             : item.auxiliary_shells || item.auxiliary_primitives))
      throw std::invalid_argument("CPU HF fleet requires valid shell/primitive counts");
    const auto retained =
        sum({product({8, sum({product({4, item.shape.spins, item.shape.orbital_aos,
                                       item.shape.orbital_aos}),
                              product({24, item.shape.atoms})})}),
             product({128, sum({item.orbital_primitives, item.orbital_shells, item.shape.atoms})}),
             product({128, sum({item.auxiliary_primitives, item.auxiliary_shells})})});
    const auto next_retained = sum({retained_, product({copies, retained})});
    const auto next_workspace = std::max({workspace_, phases[0], phases[1], phases[2]});
    const auto next_peak = sum({next_retained, next_workspace});
    retained_ = next_retained;
    workspace_ = next_workspace;
    peak_ = next_peak;
    return phases;
  }

  /** Stable fleet bridge ordering: retained bytes, workspace bytes, peak bytes. */
  std::array<std::uint64_t, 3> inventory() const { return {retained_, workspace_, peak_}; }

 private:
  std::uint64_t retained_{0}, workspace_{0}, peak_{0};
};

}  // namespace generativeqc::scf
