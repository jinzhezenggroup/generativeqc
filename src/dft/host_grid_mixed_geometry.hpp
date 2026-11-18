#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>

#include "dft/host_grid_geometry.hpp"

namespace generativeqc::dft {

struct HostGridMixedPlan {
  std::size_t numeric_bytes{}, point_visits{}, ao_point_visits{}, pair_visits{};
};

struct HostGridMixedWork {
  std::size_t tiles_started{}, tiles_completed{}, point_visits{}, ao_point_visits{}, pair_visits{};
};

/** Bounded mixed AO/Becke geometry of the exact retained owner-translating grid.
 * Reuses the canonical order-three spatial AO provider and compiler Taylor and
 * mixed-jet algebra also used by CUDA. Products use one common binary scale for
 * all four jet fields, preserving a zero primal's nonzero tangents. The explicit
 * borrowed pre-partition measures must come from this grid's atomic rule; no
 * division by a possibly zero partition weight or displaced evaluation occurs.
 * A dimensionless primal-partition gate precedes publication on every tile.
 *
 * The ordinary reader and atomic measures, authenticated together by the
 * mandatory validator, outlive this synchronous non-reentrant consumer. The
 * numeric ceiling includes both readers' FP64 scratch, not borrowed arrays,
 * integer metadata, scalar stack, allocator overhead or RSS. Per-tile work is
 * explicit; this is a geometry source, NOT an XC or stationary Hessian. Atom
 * coincidence and grid points on a nucleus are non-smooth and refused.
 */
class HostGridMixedGeometry {
 public:
  static HostGridMixedPlan plan(const HostGridGeometry& geometry);
  HostGridMixedGeometry(HostGridGeometry& geometry, std::span<const double> atomic_weights,
                        HostGridMixedPlan limits, std::function<void()> validate_current);
  ~HostGridMixedGeometry();
  HostGridMixedGeometry(const HostGridMixedGeometry&) = delete;
  HostGridMixedGeometry& operator=(const HostGridMixedGeometry&) = delete;

  /** Primal AO [component(4),point,AO]; moving AO [left/right/mixed,component,point,AO].
   * Primal weight [point]; moving weight [left/right/mixed,point]. All publish
   * together and may alias external directions, not one another or borrowed
   * geometry/measures. Left/right directions are independent atom/xyz arrays.
   */
  void read(std::size_t begin, std::size_t count, std::span<const double> left,
            std::span<const double> right, std::span<double> ao, std::span<double> moving_ao,
            std::span<double> weights, std::span<double> moving_weights);
  [[nodiscard]] HostGridMixedPlan resources() const noexcept;
  [[nodiscard]] HostGridMixedWork work() const noexcept;
  [[nodiscard]] const HostGridGeometry& geometry() const noexcept;
  void validate_current() const;
  void reject_borrowed_alias(std::span<const double> values) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft
