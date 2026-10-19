#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include "scf/cuda/eigensolver_types.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "tensor/cuda_binding_resources.hpp"

namespace generativeqc::scf {

/** Shape-only KS inventory shared by the native CLI and Python adapter.
 * This bounds named numeric/value-metadata owners, not allocator overhead,
 * process RSS or a complete endpoint allocation receipt. Successful queries
 * use scalar/stack storage, without provider discovery or scientific execution.
 */
struct KsHostResourceShape {
  std::uint64_t atoms, orbital_aos, cartesian_aos, shells, primitives, spins;
  std::uint64_t maximum_angular, ecp_terms, grid_points, radial_points, polar_points;
  std::uint64_t tile_points, diis_history, max_iterations;
  bool first_ao_derivatives, nonlocal_correlation;
  std::uint64_t ecp_workspace;
};

using CpuKsHostResourceShape = KsHostResourceShape;

/** Execution-storage distinctions, not capability or scientific admission. */
enum class KsHostExecution { cpu, cuda_fused, cuda_host_unfused };

/** Stable private bridge ordering; zero CUDA-only slots preserve the common
 * host inventory vocabulary without implying CPU ownership of those bindings.
 */
inline constexpr std::array<const char*, 16> ks_host_resource_keys{"metadata",
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

inline constexpr auto& cpu_ks_host_resource_keys = ks_host_resource_keys;

/** Conservative simultaneous capacities for serialized preparation/host work.
 * Retained grid/provider/state survives every item; setup and SCF workspaces
 * are phase maxima, and the optional AO cache belongs to one active solve.
 * Checked arithmetic uses the shared planner's signed-int64 byte domain.
 * Invalid topology and overflow throw before returning a partial inventory.
 */
inline std::array<std::uint64_t, 16> ks_host_resources(
    const KsHostResourceShape& shape, KsHostExecution execution = KsHostExecution::cpu) {
  using Bytes = std::uint64_t;
  constexpr Bytes maximum = std::numeric_limits<std::int64_t>::max();
  const auto sum = [](std::initializer_list<Bytes> values) {
    Bytes result = 0;
    for (const auto value : values) {
      if (value > maximum - result) throw std::overflow_error("KS capacity sum overflows");
      result += value;
    }
    return result;
  };
  const auto product = [](std::initializer_list<Bytes> values) {
    Bytes result = 1;
    for (const auto value : values) {
      if (value > maximum || (value && result > maximum / value))
        throw std::overflow_error("KS capacity product overflows");
      result *= value;
    }
    return result;
  };
  if (execution != KsHostExecution::cpu && execution != KsHostExecution::cuda_fused &&
      execution != KsHostExecution::cuda_host_unfused)
    throw std::invalid_argument("unknown KS host execution storage");
  const bool cpu = execution == KsHostExecution::cpu;
  const bool host_xc = cpu || execution == KsHostExecution::cuda_host_unfused;
  // The resolved v4 wire shape uses polar=0 and radial=level (including 0).
  // This query only admits capacity; the scientific descriptor validates v4.
  const bool element_grid = shape.polar_points == 0 && shape.radial_points <= 9;
  if (!shape.atoms || !shape.orbital_aos || shape.cartesian_aos < shape.orbital_aos ||
      !shape.shells || shape.shells > shape.orbital_aos || shape.primitives < shape.shells ||
      (shape.spins != 1 && shape.spins != 2) || shape.maximum_angular > maximum ||
      (cpu && shape.maximum_angular > 4) || !shape.grid_points ||
      (!element_grid && (!shape.radial_points || !shape.polar_points)) || !shape.tile_points ||
      !shape.max_iterations)
    throw std::invalid_argument(
        "KS inventory requires valid topology; CPU is limited to through-g");
  const auto orbital_square = product({shape.orbital_aos, shape.orbital_aos});
  const auto cartesian_square = product({shape.cartesian_aos, shape.cartesian_aos});
  const auto orbital_fourth = cpu ? product({orbital_square, orbital_square}) : 0;
  const auto packed = sum({product({3, shape.atoms}), product({2, shape.primitives}),
                           product({16, shape.orbital_aos})});
  const auto metadata =
      sum({8192,
           product({512, sum({shape.atoms, shape.shells, shape.primitives, shape.orbital_aos,
                              shape.cartesian_aos, shape.ecp_terms})}),
           !cpu && !host_xc ? tensor::kCudaSymmetricProductBindingHostBytes : 0});
  const auto grid = product({36, shape.grid_points});
  const auto basis = product({8, packed});
  const auto warm = sum({product({8, orbital_square, sum({4, product({4, shape.spins})})}),
                         product({96, shape.atoms})});
  const auto history = product({256, shape.max_iterations});
  const auto provider = product({8, sum({product({2, orbital_square}), orbital_fourth})});
  const auto nonlocal_provider = shape.nonlocal_correlation ? product({48, shape.grid_points}) : 0;
  Bytes solver = 0;
  if (!cpu && shape.orbital_aos > cuda_execution::kSmallEigensolverLimit) {
    if (shape.orbital_aos > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("KS eigensolver dimension exceeds size_t");
    solver = sum(
        {ordinary_eigensolver_workspace_allowance(static_cast<std::size_t>(shape.orbital_aos))});
  }
  const Bytes solver_binding = cpu ? 0 : kOrdinaryEigensolverBindingHostBytes;
  const Bytes matrix_binding = cpu ? 0 : tensor::kCudaMatrixProductBindingHostBytes;
  const Bytes structured_binding = cpu ? 0 : tensor::kCudaStructuredProductBindingHostBytes;
  const auto xc_staging =
      !cpu && host_xc ? product({8, orbital_square, shape.spins == 2 ? 6u : 2u}) : 0;
  const auto retained =
      sum({metadata, grid, basis, warm, history, provider, nonlocal_provider, solver,
           solver_binding, matrix_binding, structured_binding, xc_staging});
  const auto diis_slots = sum({shape.diis_history, 1});
  const auto matrix =
      sum({product({8, shape.spins, orbital_square, sum({128, product({2, diis_slots})})}),
           product({16, diis_slots, diis_slots})});
  const auto tile = shape.grid_points < shape.tile_points ? shape.grid_points : shape.tile_points;
  const auto xc_tile =
      host_xc ? sum({product({8, tile, shape.orbital_aos, shape.first_ao_derivatives ? 4u : 1u}),
                     product({8, shape.spins, orbital_square})})
              : 0;
  constexpr Bytes cache_cap = 64 << 20;
  const auto cache = cpu && shape.first_ao_derivatives && shape.spins == 1 &&
                             shape.grid_points <= cache_cap / 32 / shape.orbital_aos
                         ? product({32, shape.grid_points, shape.orbital_aos})
                         : 0;
  const auto nonlocal_work = shape.nonlocal_correlation
                                 ? sum({nonlocal_provider, product({32, tile, shape.orbital_aos}),
                                        product({8, orbital_square, shape.spins == 2 ? 3u : 1u})})
                                 : 0;
  // Bound the radial shell descriptors and up to three cached angular rules,
  // including the original and temporary polar vectors during CUDA setup.
  // This shape-only ABI deliberately admits both unpruned and pruned grids.
  auto quadrature = sum(
      {product({16, sum({product({2, shape.radial_points}), product({3, shape.polar_points})})}),
       product({8, shape.atoms})});
  if (element_grid)
    quadrature = sum({quadrature, product({80, shape.grid_points}),
                      product({8, shape.atoms, shape.atoms}), product({64, shape.atoms}), 8});
  Bytes setup = 0;
  if (cpu) {
    const auto cartesian_fourth = product({cartesian_square, cartesian_square});
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
             std::max(recurrence, transform)});
    setup = sum({integrals, quadrature, basis, shape.ecp_workspace});
  } else {
    const auto pairs = product({shape.shells, sum({shape.shells, 1})}) / 2;
    quadrature = sum({quadrature, product({8, sum({product({4, shape.atoms}), 2 * (512 + 256)})})});
    setup =
        sum({product({2048, sum({1, shape.atoms, shape.shells, shape.primitives, shape.orbital_aos,
                                 shape.cartesian_aos, pairs, product({pairs, pairs})})}),
             product({64, sum({cartesian_square, orbital_square,
                               product({shape.orbital_aos, shape.cartesian_aos})})}),
             quadrature, basis, shape.ecp_workspace});
  }
  return {metadata,       grid,
          basis,          warm,
          history,        provider,
          solver,         solver_binding,
          matrix_binding, structured_binding,
          xc_staging,     nonlocal_provider,
          cache,          retained,
          setup,          sum({matrix, xc_tile, nonlocal_work, cache})};
}

/** Compatibility entry point for the existing private CPU item ABI. */
inline std::array<std::uint64_t, 16> cpu_ks_host_resources(const CpuKsHostResourceShape& shape) {
  return ks_host_resources(shape);
}

/** Aggregate retained owners and serialized workspace across a KS fleet.
 * Floors and retained allowances are explicitly supplied consumer capacities,
 * not admission of those consumers. A retained allowance coexists with every
 * preparation/replay, rather than being folded into a transient maximum.
 * Append validates the complete peak before
 * changing state; repetitions avoid materializing homogeneous descriptors.
 */
class KsHostFleetResources {
 public:
  explicit KsHostFleetResources(std::uint64_t workspace_floor = 0,
                                std::uint64_t retained_allowance = 0)
      : retained_(checked_sum(0, retained_allowance)),
        workspace_(checked_sum(0, workspace_floor)),
        peak_(checked_sum(retained_, workspace_)) {}

  std::array<std::uint64_t, 16> append(const KsHostResourceShape& shape, std::uint64_t copies = 1,
                                       KsHostExecution execution = KsHostExecution::cpu) {
    if (!copies) throw std::invalid_argument("KS fleet repetition must be positive");
    const auto item = ks_host_resources(shape, execution);
    auto next_totals = totals_;
    for (std::size_t index = 0; index < next_totals.size(); ++index)
      next_totals[index] = checked_sum(next_totals[index], checked_product(copies, item[index]));
    const auto next_retained = checked_sum(retained_, checked_product(copies, item[13]));
    const auto next_workspace = std::max({workspace_, item[14], item[15]});
    const auto next_peak = checked_sum(next_retained, next_workspace);
    totals_ = next_totals;
    retained_ = next_retained;
    workspace_ = next_workspace;
    peak_ = next_peak;
    return item;
  }

  /** Twelve persistent component totals, then retained/workspace/peak bytes. */
  std::array<std::uint64_t, 15> inventory() const {
    std::array<std::uint64_t, 15> result{};
    std::copy(totals_.begin(), totals_.end(), result.begin());
    result[12] = retained_;
    result[13] = workspace_;
    result[14] = peak_;
    return result;
  }

 private:
  static std::uint64_t checked_sum(std::uint64_t left, std::uint64_t right) {
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if (right > static_cast<std::uint64_t>(maximum) - left)
      throw std::overflow_error("KS fleet capacity sum overflows");
    return left + right;
  }

  static std::uint64_t checked_product(std::uint64_t left, std::uint64_t right) {
    constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (left > maximum || right > maximum || (right && left > maximum / right))
      throw std::overflow_error("KS fleet capacity product overflows");
    return left * right;
  }

  std::array<std::uint64_t, 12> totals_{};
  std::uint64_t retained_{0}, workspace_{0}, peak_{0};
};

using CpuKsHostFleetResources = KsHostFleetResources;

}  // namespace generativeqc::scf
