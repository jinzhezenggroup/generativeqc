#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace generativeqc::scf {

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
  using Bytes = std::uint64_t;
  constexpr Bytes maximum = std::numeric_limits<std::int64_t>::max();
  const auto sum = [](std::initializer_list<Bytes> values) {
    Bytes result = 0;
    for (const auto value : values) {
      if (value > maximum - result) throw std::overflow_error("CPU HF capacity sum overflows");
      result += value;
    }
    return result;
  };
  const auto product = [](std::initializer_list<Bytes> values) {
    Bytes result = 1;
    for (const auto value : values) {
      if (value > maximum || (value && result > maximum / value))
        throw std::overflow_error("CPU HF capacity product overflows");
      result *= value;
    }
    return result;
  };
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

}  // namespace generativeqc::scf
