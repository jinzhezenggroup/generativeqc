#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>

#include "dft/host_force_resources.hpp"
#include "integrals/packed_spherical_eri.hpp"
#include "scf/cuda/eigensolver_types.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "tensor/cuda_binding_resources.hpp"

namespace generativeqc::scf {

/** Auxiliary topology for the single full-range, value-only CPU DF J owner.
 * Public dimensions describe retained factors; Cartesian dimensions describe
 * the generated integral build before any spherical projection. */
struct CpuKsDfHostShape {
  std::uint64_t aos, cartesian_aos, shells, primitives, maximum_angular;
};

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
  std::optional<CpuKsDfHostShape> cpu_df = std::nullopt;
};

using CpuKsHostResourceShape = KsHostResourceShape;

/** Execution-storage distinctions, not capability or scientific admission. */
enum class KsHostExecution {
  cpu,
  cuda_fused,
  cuda_host_unfused,
  cpu_packed,
  cpu_forces,
  cpu_packed_forces
};

constexpr bool ks_host_packed(KsHostExecution execution) noexcept {
  return execution == KsHostExecution::cpu_packed ||
         execution == KsHostExecution::cpu_packed_forces;
}

/** Select only the numeric layout; force capability remains a prepared property. */
constexpr KsHostExecution cpu_ks_host_execution(bool packed, bool forces) noexcept {
  if (packed) return forces ? KsHostExecution::cpu_packed_forces : KsHostExecution::cpu_packed;
  return forces ? KsHostExecution::cpu_forces : KsHostExecution::cpu;
}

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

/** Numeric extent of the value array for an already validated spherical basis. */
inline std::uint64_t cpu_ks_packed_value_bytes(std::uint64_t orbital_aos) {
  if (orbital_aos > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("packed KS orbital count exceeds size_t");
  return integrals::PackedSphericalEri::required_bytes(static_cast<std::size_t>(orbital_aos));
}

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
  const bool packed_eri = ks_host_packed(execution);
  const bool forces =
      execution == KsHostExecution::cpu_forces || execution == KsHostExecution::cpu_packed_forces;
  const bool cpu = execution == KsHostExecution::cpu || packed_eri || forces;
  if (!cpu && execution != KsHostExecution::cuda_fused &&
      execution != KsHostExecution::cuda_host_unfused)
    throw std::invalid_argument("unknown KS host execution storage");
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
  if (packed_eri && shape.maximum_angular > 3)
    throw std::invalid_argument("packed KS inventory requires a real spherical s/p/d/f topology");
  if (forces && (shape.nonlocal_correlation || shape.maximum_angular > 3))
    throw std::invalid_argument("CPU KS force inventory requires local XC and s/p/d/f topology");
  const auto& df = shape.cpu_df;
  if (df && (execution != KsHostExecution::cpu || shape.nonlocal_correlation || shape.ecp_terms ||
             shape.ecp_workspace || shape.maximum_angular > 3 || !df->aos ||
             df->cartesian_aos < df->aos || !df->shells || df->shells > df->aos ||
             df->primitives < df->shells || df->maximum_angular > 3))
    throw std::invalid_argument("CPU DF KS inventory requires all-electron s/p/d/f topology");
  const auto orbital_square = product({shape.orbital_aos, shape.orbital_aos});
  const auto cartesian_square = product({shape.cartesian_aos, shape.cartesian_aos});
  const auto orbital_fourth = cpu && !df ? product({orbital_square, orbital_square}) : 0;
  const auto packed = sum({product({3, shape.atoms}), product({2, shape.primitives}),
                           product({16, shape.orbital_aos})});
  const auto metadata =
      sum({8192,
           product({512, sum({shape.atoms, shape.shells, shape.primitives, shape.orbital_aos,
                              shape.cartesian_aos, shape.ecp_terms})}),
           !cpu && !host_xc ? tensor::kCudaSymmetricProductBindingHostBytes : 0,
           df ? product({1024, sum({shape.atoms, shape.shells, shape.primitives, df->shells,
                                    df->primitives, df->aos, df->cartesian_aos})})
              : 0});
  const auto grid = product({36, shape.grid_points});
  const auto basis = product({8, packed});
  const auto warm = sum({product({8, orbital_square, sum({4, product({4, shape.spins})})}),
                         product({96, shape.atoms}), forces ? product({24, shape.atoms}) : 0});
  const auto history = product({256, shape.max_iterations});
  // PreparedFockPlan retains the one-electron matrices plus exactly one value
  // array. The packed and dense arrays never coexist for a packed request.
  const auto eri_values =
      packed_eri ? cpu_ks_packed_value_bytes(shape.orbital_aos) : product({8, orbital_fourth});
  Bytes provider = sum({product({16, orbital_square}), eri_values});
  Bytes df_setup = 0, df_apply = 0;
  if (df) {
    const auto coordinates = product({3, shape.atoms});
    const auto metric = product({8, df->aos, df->aos});
    const auto tensor = product({8, orbital_square, df->aos});
    const auto one_electron =
        sum({product({16, orbital_square, sum({1, coordinates})}), product({8, coordinates})});
    // Raw metric/tensor and the whitened factor survive as distinct owners.
    // Charge the auxiliary-major factor once, either retained or held as
    // fallback scratch; metric rank truncation does not reduce these extents.
    provider = sum({one_electron, metric, product({3, tensor})});
    const auto cartesian_raw = sum({product({8, df->cartesian_aos, df->cartesian_aos}),
                                    product({8, cartesian_square, df->cartesian_aos})});
    const auto projection =
        product({128, sum({shape.cartesian_aos, shape.orbital_aos, df->aos, df->cartesian_aos,
                           product({shape.orbital_aos, shape.cartesian_aos}),
                           product({df->aos, df->cartesian_aos})})});
    // Cartesian Jet matrices, unpacked coordinate panels and the nuclear
    // gradient coexist during one-electron setup. The public panels of a
    // spherical projection are included in the provider allowance above.
    const auto one_electron_setup =
        product({64, sum({cartesian_square, shape.atoms, 256}), sum({1, coordinates})});
    const auto whitening = sum({product({4, metric}), product({64, df->aos})});
    df_setup = sum({projection, std::max({one_electron_setup, cartesian_raw, whitening})});
    df_apply = sum({product({8, df->aos}), product({16, orbital_square})});
  }
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
  Bytes force_workspace = 0;
  if (forces) {
    // Forces re-diagonalize and authenticate the retained final F[D]. The
    // eigensolve/matrix workspace is covered by the ordinary solver envelope;
    // the verified D/F/W/C/eigenvalues then remain live across both sources.
    const auto frame =
        product({8, shape.spins, sum({product({4, orbital_square}), shape.orbital_aos})});
    // Integral source scratch retires before XC. Its four coordinate buffers
    // survive the XC pullback and then overlap force assembly/publication.
    const auto xc = dft::cpu_stationary_xc_workspace_bytes(shape.atoms, shape.orbital_aos,
                                                           shape.grid_points, tile);
    const auto sources = product({96, shape.atoms});
    force_workspace = sum({frame, std::max({Bytes{dft::cpu_stationary_force_source_bytes},
                                            sum({xc, sources}), product({7 * 24, shape.atoms})})});
  }
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
  if (df) {
    setup = sum({df_setup, quadrature, basis});
  } else if (cpu) {
    // Both spherical value sources use the same shell-quartet visitor. Keep
    // the ordinary setup allowance; only the retained value extent changes.
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
  return {metadata,
          grid,
          basis,
          warm,
          history,
          provider,
          solver,
          solver_binding,
          matrix_binding,
          structured_binding,
          xc_staging,
          nonlocal_provider,
          cache,
          retained,
          setup,
          std::max(force_workspace, sum({matrix, xc_tile, nonlocal_work, cache, df_apply}))};
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
