#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

#include "dft/host_grid_mixed_geometry.hpp"

namespace generativeqc::dft {

struct HostGgaCurvaturePlan {
  std::size_t numeric_bytes{}, map_terms{}, ao_point_visits{}, grid_pair_visits{};
};

struct HostGgaCurvatureWork {
  std::size_t pairs_started{}, pairs_completed{}, tiles_completed{}, point_visits{}, map_terms{};
};

/** Complete frozen-density semilocal energy mixed geometry, not a Hessian.
 * Both AO legs, Cartesian density/gradient fields, XC cross-spin ingredient
 * Hessians and moving partition weights share compiler-owned JVP/product maps.
 * RKS uses total D and UKS occupation-one spin Ds in the authenticated basis.
 * No D'/W', integral/ECP, nuclear repulsion or dispersion term enters this source.
 * This consumer does not itself certify that the borrowed density is stationary.
 * It retains the linked full ingredient-Hessian domain: nonvacuum empty-spin
 * endpoints may refuse rather than inventing a restricted directional limit.
 *
 * Mandatory bounds include both geometry readers' FP64 scratch, these panels
 * and one reused TensorIR arena, plus the complete source's semantic work.
 * Borrowed storage, metadata, scalar stack, allocator/RSS and total endpoint
 * time are distinct. All owners outlive exclusive synchronous invocations.
 */
class HostGgaNuclearCurvature {
 public:
  static HostGgaCurvaturePlan plan(const HostGridMixedGeometry& geometry, std::size_t spins);
  HostGgaNuclearCurvature(HostGridMixedGeometry& geometry, std::size_t spins,
                          std::span<const double> density, std::string_view method_identity,
                          HostGgaCurvaturePlan limits, std::function<void()> validate_current);
  ~HostGgaNuclearCurvature();
  HostGgaNuclearCurvature(const HostGgaNuclearCurvature&) = delete;
  HostGgaNuclearCurvature& operator=(const HostGgaNuclearCurvature&) = delete;

  /** Independent atom/xyz directions; output [energy,left,right,mixed].
   * The four scalars publish together, may alias external directions, never
   * density or borrowed geometry/measures. Compensated FP64 grid accumulation
   * retains signed contributions without changing functional or density policy.
   */
  void apply(std::span<const double> left, std::span<const double> right, std::span<double> output);
  [[nodiscard]] HostGgaCurvaturePlan resources() const noexcept;
  [[nodiscard]] HostGgaCurvatureWork work() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
