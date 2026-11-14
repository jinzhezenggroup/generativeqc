#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>

#include "dft/host_grid_geometry.hpp"

namespace generativeqc::dft {

struct HostGridColumnPlan {
  std::size_t numeric_bytes{}, point_visits{}, ao_point_visits{}, pair_visits{}, tape_visits{};
};

struct HostGridColumnWork {
  std::size_t tiles_started{}, tiles_completed{}, point_visits{}, ao_point_visits{}, pair_visits{},
      forward_nodes{}, reverse_nodes{};
};

/** Source-driven spatial AO jets and complete moving-partition curvature column.
 * Each tile evaluates ordinary normalized order-three AO jets once. Each point
 * records one Becke source and reverses its right JVP once, scattering directly
 * to physical nuclear coordinates, including the moving grid-point owner.
 * Compiler-generated scalar jets supply every local derivative; native code
 * owns only bounded scheduling, binary scales and physical incidence.
 *
 * The mandatory immutable owner authenticates the reader and borrowed atomic
 * measures together. Numeric capacity conservatively includes the reader,
 * tile outputs, right snapshot and the finite-slot reverse tape (metadata too).
 * Borrowed storage, allocator overhead, scalar stack and RSS/time are distinct.
 * This source is neither frozen XC energy nor a stationary Hessian.
 */
class HostGridColumnGeometry {
 public:
  static HostGridColumnPlan plan(const HostGridGeometry& geometry);
  HostGridColumnGeometry(HostGridGeometry& geometry, std::span<const double> atomic_weights,
                         HostGridColumnPlan limits, std::function<void()> validate_current);
  ~HostGridColumnGeometry();
  HostGridColumnGeometry(const HostGridColumnGeometry&) = delete;
  HostGridColumnGeometry& operator=(const HostGridColumnGeometry&) = delete;

  /** Outputs: AO [ordinary spatial component(20),point,AO] and partitioned
   * measures [value,right,gradient(3N),Hright(3N),point]. Both publish together,
   * may alias the external right direction, never each other or borrowed state.
   * The right leg is atom/xyz-major and authenticated through final publication.
   * No source replay or coordinate-squared AO/weight tensor is retained.
   */
  void read(std::size_t begin, std::size_t count, std::span<const double> right,
            std::span<double> ao_jets, std::span<double> weight_column);
  [[nodiscard]] HostGridColumnPlan resources() const noexcept;
  [[nodiscard]] HostGridColumnWork work() const noexcept;
  [[nodiscard]] const HostGridGeometry& geometry() const noexcept;
  [[nodiscard]] HostGridGeometry& geometry() noexcept;
  void validate_current() const;
  void reject_borrowed_alias(std::span<const double> values) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
