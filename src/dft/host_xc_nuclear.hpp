#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

#include "dft/host_grid_geometry.hpp"

namespace generativeqc::dft {

struct HostGgaNuclearPlan {
  std::size_t numeric_bytes{}, map_terms{}, ao_point_visits{}, grid_pair_visits{};
};

struct HostGgaNuclearWork {
  std::size_t directions_started{}, directions_completed{}, tiles_completed{}, point_visits{};
  std::size_t map_evaluations{}, scheduled_map_terms{};
};

/** Native fixed-density moving-grid GGA AO Fock direction, not a Hessian.
 * Includes AO motion in both density fields and matrix factors, plus the exact
 * retained quadrature-weight motion. Scalar XC partials/JVPs and contractions
 * are linked compiler programs, with no JIT, displaced-grid or oracle fallback.
 * RKS borrows one total D; UKS borrows two occupation-one spin Ds. The caller
 * authenticates the immutable density, method and geometry owner using the
 * mandatory validator; the exact linked method/spin closure is also checked.
 * Geometry and density owners must outlive the synchronous consumer.
 *
 * plan bounds the complete source's scheduled AO-point/pair visits and TensorIR
 * terms. numeric_bytes includes this consumer and the borrowed geometry reader's
 * FP64 arena, not metadata, borrowed basis/grid/D, provider/scalar stack, allocator
 * overhead or RSS. This is not a full endpoint resource/performance gate.
 * Output is transactional and may alias external direction but not borrowed
 * density/basis/grid. Calls on one consumer must not overlap/reenter.
 */
class HostGgaNuclearXc {
 public:
  static HostGgaNuclearPlan plan(const HostGridGeometry& geometry, std::size_t spins);

  HostGgaNuclearXc(HostGridGeometry& geometry, std::size_t spins, std::span<const double> density,
                   std::string_view method_identity, HostGgaNuclearPlan limits,
                   std::function<void()> validate_current);
  ~HostGgaNuclearXc();
  HostGgaNuclearXc(const HostGgaNuclearXc&) = delete;
  HostGgaNuclearXc& operator=(const HostGgaNuclearXc&) = delete;

  void apply(std::span<const double> direction, std::span<double> output);
  [[nodiscard]] HostGgaNuclearWork work() const noexcept;
  [[nodiscard]] HostGgaNuclearPlan resources() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft
