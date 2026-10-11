#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace generativeqc::scf {

/** Shape-only CPU KS inventory shared by the native CLI and Python adapter.
 * This bounds named numeric/value-metadata owners, not allocator overhead,
 * process RSS or a complete endpoint allocation receipt. Successful queries
 * use scalar/stack storage, without provider discovery or scientific execution.
 */
struct CpuKsHostResourceShape {
  std::uint64_t atoms, orbital_aos, cartesian_aos, shells, primitives, spins;
  std::uint64_t maximum_angular, ecp_terms, grid_points, radial_points, polar_points;
  std::uint64_t tile_points, diis_history, max_iterations;
  bool first_ao_derivatives, nonlocal_correlation;
  std::uint64_t ecp_workspace;
};

/** Stable private bridge ordering; zero CUDA-only slots preserve the common
 * host inventory vocabulary without implying CPU ownership of those bindings.
 */
inline constexpr std::array<const char*, 16> cpu_ks_host_resource_keys{"metadata",
                                                                       "grid",
                                                                       "basis",
                                                                       "warm_and_matrices",
                                                                       "history",
                                                                       "provider",
                                                                       "solver_host",
                                                                       "solver_binding_host",
                                                                       "matrix_binding_host",
                                                                       "structured_binding_host",
                                                                       "xc_schedule_staging",
                                                                       "nonlocal_provider",
                                                                       "ao_grid_cache",
                                                                       "retained",
                                                                       "setup_workspace",
                                                                       "scf_workspace"};

/** Conservative simultaneous capacities for serialized CPU preparation/SCF.
 * Retained grid/provider/state survives every item; setup and SCF workspaces
 * are phase maxima, and the optional AO cache belongs to one active solve.
 * Checked arithmetic uses the shared planner's signed-int64 byte domain.
 * Invalid topology and overflow throw before returning a partial inventory.
 */
inline std::array<std::uint64_t, 16> cpu_ks_host_resources(const CpuKsHostResourceShape& shape) {
  using Bytes = std::uint64_t;
  constexpr Bytes maximum = std::numeric_limits<std::int64_t>::max();
  const auto sum = [](std::initializer_list<Bytes> values) {
    Bytes result = 0;
    for (const auto value : values) {
      if (value > maximum - result) throw std::overflow_error("CPU KS capacity sum overflows");
      result += value;
    }
    return result;
  };
  const auto product = [](std::initializer_list<Bytes> values) {
    Bytes result = 1;
    for (const auto value : values) {
      if (value > maximum || (value && result > maximum / value))
        throw std::overflow_error("CPU KS capacity product overflows");
      result *= value;
    }
    return result;
  };
  if (!shape.atoms || !shape.orbital_aos || shape.cartesian_aos < shape.orbital_aos ||
      !shape.shells || shape.shells > shape.orbital_aos || shape.primitives < shape.shells ||
      (shape.spins != 1 && shape.spins != 2) || shape.maximum_angular > 4 || !shape.grid_points ||
      !shape.radial_points || !shape.polar_points || !shape.tile_points || !shape.max_iterations)
    throw std::invalid_argument("CPU KS inventory requires a valid through-g topology");
  const auto orbital_square = product({shape.orbital_aos, shape.orbital_aos});
  const auto cartesian_square = product({shape.cartesian_aos, shape.cartesian_aos});
  const auto orbital_fourth = product({orbital_square, orbital_square});
  const auto cartesian_fourth = product({cartesian_square, cartesian_square});
  const auto packed = sum({product({3, shape.atoms}), product({2, shape.primitives}),
                           product({16, shape.orbital_aos})});
  const auto metadata =
      sum({8192, product({512, sum({shape.atoms, shape.shells, shape.primitives, shape.orbital_aos,
                                    shape.cartesian_aos, shape.ecp_terms})})});
  const auto grid = product({36, shape.grid_points});
  const auto basis = product({8, packed});
  const auto warm = sum({product({8, orbital_square, sum({4, product({4, shape.spins})})}),
                         product({96, shape.atoms})});
  const auto history = product({256, shape.max_iterations});
  const auto provider = product({8, sum({product({2, orbital_square}), orbital_fourth})});
  const auto nonlocal_provider = shape.nonlocal_correlation ? product({48, shape.grid_points}) : 0;
  const auto retained = sum({metadata, grid, basis, warm, history, provider, nonlocal_provider});
  const auto diis_slots = sum({shape.diis_history, 1});
  const auto matrix =
      sum({product({8, shape.spins, orbital_square, sum({128, product({2, diis_slots})})}),
           product({16, diis_slots, diis_slots})});
  const auto tile = shape.grid_points < shape.tile_points ? shape.grid_points : shape.tile_points;
  const auto xc_tile =
      sum({product({8, tile, shape.orbital_aos, shape.first_ao_derivatives ? 4u : 1u}),
           product({8, shape.spins, orbital_square})});
  constexpr Bytes cache_cap = 64 << 20;
  const auto cache = shape.first_ao_derivatives && shape.spins == 1 &&
                             shape.grid_points <= cache_cap / 32 / shape.orbital_aos
                         ? product({32, shape.grid_points, shape.orbital_aos})
                         : 0;
  const auto nonlocal_work = shape.nonlocal_correlation
                                 ? sum({nonlocal_provider, product({32, tile, shape.orbital_aos}),
                                        product({8, orbital_square, shape.spins == 2 ? 3u : 1u})})
                                 : 0;
  const auto angular_extent = sum({shape.maximum_angular, 3});
  const auto hermite =
      product({6, angular_extent, angular_extent, sum({product({2, shape.maximum_angular}), 4})});
  const auto order = sum({product({4, shape.maximum_angular}), 1});
  const auto recurrence =
      product({64, sum({product({order, order, order, order}), hermite, 256, shape.atoms})});
  const auto transform =
      product({32, sum({cartesian_fourth, orbital_fourth,
                        product({shape.cartesian_aos, shape.orbital_aos}), orbital_square})});
  const auto integrals =
      sum({product({64, sum({product({2, cartesian_square}), cartesian_fourth})}),
           recurrence > transform ? recurrence : transform});
  const auto quadrature = sum(
      {product({16, sum({shape.radial_points, shape.polar_points})}), product({8, shape.atoms})});
  return {metadata,
          grid,
          basis,
          warm,
          history,
          provider,
          0,
          0,
          0,
          0,
          0,
          nonlocal_provider,
          cache,
          retained,
          sum({integrals, quadrature, basis, shape.ecp_workspace}),
          sum({matrix, xc_tile, nonlocal_work, cache})};
}
}  // namespace generativeqc::scf
