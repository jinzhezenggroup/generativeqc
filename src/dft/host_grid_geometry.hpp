#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/host_hessian_aot.hpp"

namespace generativeqc::dft {

struct HostGridGeometryPlan {
  std::size_t numeric_bytes{}, point_visits{}, ao_point_visits{}, grid_pair_visits{};
};

struct HostGridGeometryWork {
  std::size_t tiles_started{}, tiles_completed{}, point_visits{}, ao_point_visits{},
      grid_pair_visits{};
};

/** Bounded native AO/weight motion on a retained, owner-translating grid.
 * Ordinary normalized spherical/cartesian AO jets come from AoBasis; the linked
 * Taylor-AD leaf supplies their chain rule. The relative displacement is grid
 * owner motion minus AO-center motion. No displaced geometries or reference
 * engine are used. The basis, grid, and their immutable owner must outlive this
 * reader. validate_current must reject any in-place content or epoch change.
 * Pointer/shape replacement and disagreement between basis/grid centers are
 * independently rejected. A single reader must not overlap or reenter itself.
 *
 * The numeric bound includes all owned FP64 panels and grid derivative scratch,
 * not integer AO-center metadata, borrowed storage, provider/scalar stack,
 * allocator overhead or RSS. Work caps apply to each tile, not a full Hessian.
 */
class HostGridGeometry {
 public:
  static HostGridGeometryPlan plan(const AoBasis& basis, const MolecularGrid& grid,
                                   std::size_t tile_points);

  HostGridGeometry(const AoBasis& basis, const MolecularGrid& grid, std::size_t tile_points,
                   HostGridGeometryPlan limits, std::string_view ao_motion_identity,
                   std::function<void()> validate_current);

  /** Outputs are component/point/AO-major (value,x,y,z) and point-major weights.
   * Publish all four outputs only on success. They may alias external direction
   * storage, but not one another or any borrowed basis/grid storage.
   */
  void read(std::size_t begin, std::size_t count, std::span<const double> direction,
            std::span<double> ao, std::span<double> moving_ao, std::span<double> weights,
            std::span<double> moving_weights);

  [[nodiscard]] HostGridGeometryWork work() const noexcept { return work_; }
  [[nodiscard]] std::size_t scratch_bytes() const noexcept { return plan_.numeric_bytes; }
  [[nodiscard]] const AoBasis& basis() const noexcept { return basis_; }
  [[nodiscard]] const MolecularGrid& grid() const noexcept { return grid_; }
  [[nodiscard]] std::size_t tile_points() const noexcept { return tile_points_; }

  void validate_current() const;
  void reject_borrowed_alias(std::span<const double> values) const;

 private:
  const AoBasis& basis_;
  const MolecularGrid& grid_;
  std::size_t tile_points_, atoms_, primitives_, aos_, points_;
  HostGridGeometryPlan plan_;
  const host_hessian_aot::Program* motion_{};
  std::array<std::size_t, 12> inputs_{};
  std::array<const void*, 5> retained_{};
  std::function<void()> validate_current_;
  std::vector<std::size_t> ao_atoms_;
  std::unique_ptr<double[]> storage_;
  std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
  HostGridGeometryWork work_;
};

}  // namespace generativeqc::dft
