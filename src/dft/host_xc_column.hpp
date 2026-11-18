#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

#include "dft/host_grid_column_geometry.hpp"

namespace generativeqc::dft {

struct HostGgaColumnPlan {
  std::size_t numeric_bytes{}, map_terms{}, scalar_point_visits{}, ao_point_visits{},
      ao_motion_visits{}, grid_pair_visits{}, tape_visits{};
};

struct HostGgaColumnWork {
  std::size_t columns_started{}, columns_completed{}, tiles_completed{}, point_visits{},
      map_terms{}, scalar_point_visits{}, ao_motion_visits{};
};

/** Semilocal nuclear partial and its right action from one grid source per
 * right direction. Primal/right AO features and canonical TensorIR pullbacks
 * consume shared spatial jets. Fixed ingredient-basis seeds of the exact linked
 * mixed-energy closure supply field adjoints and their right directions; no
 * molecular-coordinate replay, XC derivative formula or occupation factor is
 * introduced. The linked full ingredient-Hessian domain is unchanged.
 *
 * RKS uses total D, UKS occupation-one spin Ds. The immutable owner authenticates
 * density and geometry together and outlives this exclusive synchronous owner.
 * An explicit per-call D_right adds the right density response exactly once;
 * this owner neither solves nor certifies that response or SCF stationarity.
 * W response, integrals/ECP, nuclei and dispersion are not included. This is
 * not a complete molecular Hessian.
 * Mandatory complete-source work caps charge the fixed scalar closure passes,
 * generated maps and composed geometry. Numeric capacity includes all composed
 * owned scratch and O(atoms) compensated outputs, not borrowed density/grid,
 * provider stack, allocator overhead or RSS/time.
 */
class HostGgaNuclearColumn {
 public:
  static HostGgaColumnPlan plan(const HostGridColumnGeometry& geometry, std::size_t spins);
  HostGgaNuclearColumn(HostGridColumnGeometry& geometry, std::size_t spins,
                       std::span<const double> density, std::string_view method_identity,
                       HostGgaColumnPlan limits, std::function<void()> validate_current);
  ~HostGgaNuclearColumn();
  HostGgaNuclearColumn(const HostGgaNuclearColumn&) = delete;
  HostGgaNuclearColumn& operator=(const HostGgaNuclearColumn&) = delete;

  /** Atom/xyz right and an empty (frozen) or symmetric spin/AO D_right.
   * Transactional output is [energy,energy_right,partial_R(3N),partial_R_right(3N)].
   * The left nuclear partial holds D fixed; its right action includes both
   * geometry and supplied D_right. energy_right includes the D_right term too,
   * so it need not equal partial_R dot right when D_right is nonempty.
   * Output may alias either external input, never ground D or borrowed geometry.
   * Both input contents are authenticated through final publication. Arbitrary
   * supplied D_right does not imply a symmetric stationary molecular Hessian;
   * no symmetry or translation repair is performed.
   */
  void apply(std::span<const double> right, std::span<const double> density_right,
             std::span<double> output);
  [[nodiscard]] HostGgaColumnPlan resources() const noexcept;
  [[nodiscard]] HostGgaColumnWork work() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
