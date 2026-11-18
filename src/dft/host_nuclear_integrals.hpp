#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

#include "dft/ao_grid.hpp"
#include "scf/fock_build.hpp"

namespace generativeqc::dft {

/** Per-direction owned FP64 bytes and complete scheduled source work. */
struct HostNuclearIntegralPlan {
  std::size_t numeric_bytes{}, one_electron_primitives{}, eri_primitives{}, map_terms{};
};

enum class HostNuclearEriSchedule { Ordered, SymmetryUnique };
enum class HostNuclearFirstSchedule { Direction, CoordinatePanel };

struct HostNuclearIntegralOptions {
  HostNuclearIntegralPlan limits;
  /** Validate immutable borrowed state at most this many completed quartets apart (1--4096). */
  std::size_t validation_quartets{64};
  /** Unique integral evaluation preserves every ordered compiler weight/scatter.
   * Frozen-D first directions omit primitive derivatives for exactly common
   * quartet translations; their upper work limits remain conservative. Curvature
   * does not inherit that omission: exact zero relative right motion instead
   * requests a value/gradient-only primitive, retaining all moving-D/W terms.
   * Ordered keeps full mixed primitives as an exhaustive bounded comparison
   * path, not a different operator.
   */
  HostNuclearEriSchedule eri_schedule{HostNuclearEriSchedule::SymmetryUnique};
};

/** Cumulative actual work, including failed directions. ao_quartets counts
 * ordered compiler-map incidences, not symmetry-unique primitive evaluations.
 */
struct HostNuclearIntegralWork {
  std::size_t directions_started{}, directions_completed{}, ao_pairs{}, ao_quartets{};
  std::size_t one_electron_primitives{}, eri_primitives{}, projection_lanes{}, map_terms{};
  /** Actual derivative-order partition of eri_primitives, including failures.
   * Value-only retains scalar contractions for a single physical atom owner.
   * First-only includes first sources and lowered curvature quartets. All
   * partitions count against the same unchanged primitive-work admission cap.
   */
  std::size_t eri_value_only_primitives{}, eri_first_only_primitives{}, eri_mixed_primitives{};
  /** Subset of mixed primitives returning all independent-center second
   * derivatives in one call. Admission keeps the conservative right-direction
   * envelope; fewer calls do not imply proportionally less derivative algebra.
   */
  std::size_t eri_matrix_primitives{};
  /** Actual returned physical-coordinate first and ordered second partials,
   * including failed matrix calls, not a substitute for primitive-call counts
   * or the unchanged conservative directional-slot admission plan.
   */
  std::size_t eri_matrix_coordinates{}, eri_matrix_second_partials{};
  /** A resident coordinate panel is one source pass; replays perform only
   * bounded linear combinations, never new primitive or quartet-map work.
   */
  std::size_t coordinate_panels_completed{}, resident_applications{};
  /** Coordinate first-source contractions with supplied D'/W', not integral
   * primitives or first-preparation quartet-map work.
   */
  std::size_t density_response_map_terms{};
  /** Sparse public-polynomial density congruence contributions, not ERI visits. */
  std::size_t density_projection_terms{};
};

/** Scalar mixed pair or source-driven whole physical-coordinate column. */
enum class HostNuclearIntegralCurvatureSchedule { Mixed, Column, Matrix, ProjectedMatrix };

/** Native full-range frozen-D S'/[T+V_eff+J+K]' source, not a Hessian.
 * The linked normalized s/p/d/f primitive AD and public-AO projections are
 * reused directly. Every ordered quartet incidence is consumed once, with no
 * molecular ERI tensor. Direction mode needs no coordinate/AO panel. The optional
 * coordinate mode explicitly reserves a complete first-source panel and keeps
 * the original bounded direction mode as its resource fallback. Compiler VJP/JVPs of the stationary
 * objective supply spin/occupation/exchange factors and incidence contributions.
 * RKS borrows total D; UKS borrows occupation-one spin Ds. Finite indefinite
 * densities are allowed. Only an exact CPU full-range model matching the exact
 * linked MethodIR is admitted. No JIT, oracle or finite-difference fallback.
 *
 * V_eff uses ionic point charges. Local/nonlocal residual ECP projectors, XC,
 * nuclear repulsion, second derivatives and stationary assembly are SEPARATE
 * sources; this class must never stand in for a complete ECP Fock direction.
 * The enclosing owner must authenticate its actual immutable strategy, basis,
 * system, method and density in validate_current; borrowed objects outlive it.
 * It does not itself certify stationarity. Calls may not overlap/reenter.
 *
 * Mandatory limits bound owned FP64 heap panels, complete-source primitive
 * visits and generated quartet terms before execution. Primitive-visit plans
 * are direction-independent upper bounds, not counts of executed derivatives.
 * The full matrix's ERI plan prices each quartet's physical-center directions;
 * it does not charge every quartet the four-distinct-atom worst case.
 * Provider/scalar stack,
 * borrowed storage, metadata, allocator overhead/RSS and endpoint time are not
 * that numeric bound. Symmetry reuse reduces evaluations, not quartic scaling.
 */
class HostKsNuclearIntegrals {
 public:
  /** Conservative range for unit-JVP coefficient factoring. Larger finite
   * indefinite densities remain supported by the streamed direction schedule.
   * Keeping all density products below 1e100 avoids introducing overflow in
   * the unscaled compiler map when a physical integral tangent is tiny.
   */
  static bool coordinate_density_supported(std::span<const double> density) noexcept;
  static std::size_t density_response_terms(const AoBasis& basis, std::size_t spins);
  static HostNuclearIntegralPlan plan(
      const AoBasis& basis, std::size_t spins,
      HostNuclearEriSchedule schedule = HostNuclearEriSchedule::SymmetryUnique,
      HostNuclearFirstSchedule first_schedule = HostNuclearFirstSchedule::Direction);

  HostKsNuclearIntegrals(
      const core::System& system, const AoBasis& basis, const scf::ResolvedFockBuild& strategy,
      std::span<const double> density, std::string_view method_identity,
      HostNuclearIntegralOptions options, std::function<void()> validate_current,
      HostNuclearFirstSchedule first_schedule = HostNuclearFirstSchedule::Direction);
  ~HostKsNuclearIntegrals();
  HostKsNuclearIntegrals(const HostKsNuclearIntegrals&) = delete;
  HostKsNuclearIntegrals& operator=(const HostKsNuclearIntegrals&) = delete;

  /** Atom/xyz direction, one AO S' and spin/row/column-major integral Fock'.
   * Both outputs publish together, may alias external direction, and must not
   * overlap one another or borrowed density/system/basis storage.
   */
  void apply(std::span<const double> direction, std::span<double> overlap_direction,
             std::span<double> integral_fock_direction);
  /** CoordinatePanel mode only: authenticate and build all Cartesian first
   * sources in one primitive/ordered-map sweep. Unit integral JVPs supply the
   * compiler's linear Fock coefficients, reused for every physical partial.
   * A prepared panel replays arbitrary finite directions through apply().
   * Preparation is transactional: failures never make a partial panel usable.
   * Its full panel, replay staging and input snapshot are included in plan().
   */
  void prepare_coordinates();
  /** Prepared CoordinatePanel only: physical-coordinate moving-D/W contribution
   * to the integral gradient's right derivative. Inputs are supplied response
   * matrices, not displaced states. Both contents are snapshotted; output may
   * alias them but never borrowed stationary storage. No primitive work runs.
   */
  void apply_density_response(std::span<const double> density_right,
                              std::span<const double> weighted_density_right,
                              std::span<double> output, std::size_t max_map_terms);
  [[nodiscard]] HostNuclearIntegralPlan resources() const noexcept;
  [[nodiscard]] HostNuclearIntegralWork work() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/** Full-range integral Lagrangian mixed/column source, not a stationary Hessian.
 * Returns [value,left,right,mixed] for Tr(D(T+V_ionic)) + E_JK(D) - Tr(W S).
 * All method/spin/occupation/constraint weights come from linked compiler maps.
 * D is total density for RKS and occupation-one spin densities for UKS; W uses
 * matching spin blocks. Both may be finite symmetric indefinite matrices.
 * Normalized contracted Cartesian/spherical s/p/d/f AOs use canonical AOT jets.
 * Mixed pairs freeze D/W; columns can include explicitly supplied right D/W
 * response, which this owner neither solves nor certifies. Residual ECP, XC,
 * nuclear/classical terms and stationary assembly remain separate.
 *
 * The owner authenticates immutable system/basis/strategy/method/D/W contents
 * and epochs in validate_current. This component does not certify stationarity.
 * Mandatory limits cap owned FP64 heap storage, full primitive visits and
 * generated weight-map work, not borrowed memory, metadata, stack or RSS/time.
 * Every ordered AO pair/quartet incidence is consumed once; no AO4 or coordinate/AO
 * derivative tensor, runtime compiler, displaced integrals or oracle fallback.
 */
class HostKsNuclearIntegralCurvature {
 public:
  /** Preconditions for the optional polynomial-density congruence. An empty
   * density checks basis eligibility only, before final-state allocation.
   * Large finite densities and unusual polynomial scales retain public-AO
   * Matrix/Column schedules rather than risking unscaled-map overflow.
   */
  static bool projected_matrix_supported(const AoBasis& basis,
                                         std::span<const double> density) noexcept;
  static HostNuclearIntegralPlan plan(
      const AoBasis& basis, std::size_t spins,
      HostNuclearIntegralCurvatureSchedule schedule = HostNuclearIntegralCurvatureSchedule::Mixed,
      HostNuclearEriSchedule eri_schedule = HostNuclearEriSchedule::SymmetryUnique);
  HostKsNuclearIntegralCurvature(
      const core::System& system, const AoBasis& basis, const scf::ResolvedFockBuild& strategy,
      std::span<const double> density, std::span<const double> weighted_density,
      std::string_view method_identity, HostNuclearIntegralOptions options,
      std::function<void()> validate_current,
      HostNuclearIntegralCurvatureSchedule schedule = HostNuclearIntegralCurvatureSchedule::Mixed);
  ~HostKsNuclearIntegralCurvature();
  HostKsNuclearIntegralCurvature(const HostKsNuclearIntegralCurvature&) = delete;
  HostKsNuclearIntegralCurvature& operator=(const HostKsNuclearIntegralCurvature&) = delete;

  /** Two atom/xyz legs and four scalar outputs; all fields publish together.
   * External directions may alias output; borrowed-state aliases and reentry
   * refuse. Independent mathematical slots survive repeated physical atoms.
   */
  void apply(std::span<const double> left, std::span<const double> right, std::span<double> output);

  /** Column schedule only: [value,right,gradient(3*natoms),mixed(3*natoms)].
   * Left nuclear partial holds D/W fixed. Empty response spans mean frozen;
   * finite symmetric spin/AO D_right/W_right add their right action exactly
   * once through canonical weight JVPs, without extra primitive source visits.
   * right includes these weight tangents and need not equal gradient dot right.
   * Each ERI gradient/JVP is reused across all physical left coordinates;
   * finite mathematical-slot S/T/V jets scatter before any next primitive.
   * The plan counts three S/T and six V evaluations per radial expansion,
   * independent of the number of physical left coordinates. No whole-source
   * replay per left coordinate or coordinate/AO derivative storage is used.
   * Owned numeric storage includes physical-coordinate output and snapshots of
   * the external direction and both response matrices.
   * A full frozen matrix needs one such sweep per right coordinate, not one
   * scalar mixed sweep per coordinate pair. Stationary assembly is separate.
   * Output publishes transactionally and may alias any external right input;
   * all three input contents are authenticated before publication.
   */
  void apply_column(std::span<const double> right, std::span<const double> density_right,
                    std::span<const double> weighted_density_right, std::span<double> output);
  /** Matrix schedule only: frozen-D/W [L, physical gradient, row-major Hessian].
   * One ordered-weight sweep reuses primitive jets over at most nine local
   * right coordinates, independent of the global atom count. Physical owner
   * indices, not coincident coordinates, determine translation/scatter.
   * This does not include D'/W' or any non-integral stationary source.
   * Output publishes transactionally; no symmetrization or rigid-mode repair.
   */
  void apply_matrix(std::span<double> output);
  [[nodiscard]] HostNuclearIntegralPlan resources() const noexcept;
  [[nodiscard]] HostNuclearIntegralWork work() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft
