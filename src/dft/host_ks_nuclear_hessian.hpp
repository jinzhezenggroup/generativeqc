#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "dft/dispersion/host_d3_column.hpp"
#include "dft/host_ks_nuclear_response.hpp"
#include "dft/host_xc_column.hpp"

namespace generativeqc::dft {

/** Explicit reservations admit the whole composition before allocating children.
 * The response reservation includes its GMRES allowance. Other source numeric
 * reservations are their supplied caps, not estimates inferred after allocation.
 */
struct HostKsHessianLimits {
  std::size_t max_numeric_bytes{}, max_assembly_numeric_bytes{}, max_nuclear_pairs{};
  HostKsNuclearLimits response;
  HostNuclearIntegralOptions integrals;
  HostGgaColumnPlan xc;
  integrals::EcpDirectionLimits ecp;
  dispersion::HostD3ColumnLimits dispersion;
  double energy_absolute_tolerance{}, energy_relative_tolerance{};
};

struct HostKsHessianResources {
  std::size_t assembly_bytes{}, reserved_numeric_bytes{}, fixed_bound_bytes{}, gmres_bytes{};
  bool admitted{};
};

struct HostKsHessianDiagnostic {
  HostKsPerturbationDiagnostic response;
  double reconstructed_energy{}, verified_energy{}, energy_error{};
  integrals::EcpDirectionWork first_ecp_work, curvature_ecp_work;
  dispersion::HostD3ColumnWork dispersion_work;
  [[nodiscard]] bool converged() const noexcept { return response.converged(); }
};

struct HostKsHessianWork {
  std::size_t columns_started{}, columns_completed{}, nuclear_pairs{};
  HostKsNuclearWork response;
  HostNuclearIntegralWork integrals;
  HostGgaColumnWork xc;
  integrals::EcpDirectionWork ecp;
  dispersion::HostD3ColumnWork dispersion;
};

/** Complete stationary CPU full-range GGA-hybrid nuclear column around a
 * verified RKS/UKS state: coupled metric/Fock response -> total D'/full W' ->
 * integral, moving XC and residual-ECP columns, ionic nuclei and optional BJ.
 * Every electronic source shares the exact state/system/Fock/grid binding.
 * D3, when supplied, uses real atomic numbers and is added exactly once.
 * No finite differences, reference engine, runtime compiler or optimizer runs.
 * This development owner is not installed-public or large-system qualification.
 *
 * The aggregate reservation precedes all shape-dependent child allocation and
 * includes each enforced child numeric cap, assembly panels and bounded GMRES.
 * Borrowed verified SCF/Fock/basis/grid/geometry owners, provider/BLAS storage,
 * callback/allocator overhead, static tables, RSS and endpoint time are separate.
 * Those borrowed owners and their validator must outlive this synchronous owner.
 */
class HostKsNuclearHessian {
 public:
  static std::size_t assembly_bytes(std::size_t atoms, std::size_t basis, std::size_t spins);
  HostKsNuclearHessian(const VerifiedKsFinalState& state, const core::System& system,
                       const scf::PreparedFockPlan& fock, HostGridColumnGeometry& geometry,
                       std::string_view method_identity, std::optional<dispersion::D3Parameters> bj,
                       HostKsHessianLimits limits,
                       HostKsResponse::ValidateCurrent validate_current);
  ~HostKsNuclearHessian();
  HostKsNuclearHessian(const HostKsNuclearHessian&) = delete;
  HostKsNuclearHessian& operator=(const HostKsNuclearHessian&) = delete;

  /** Atom/xyz right; output [physical energy,gradient dot right,gradient,H@right].
   * The integral source's -Tr(W S) Lagrangian value is restored and checked
   * against verified physical components before publication, not misreported
   * as energy. A supplied-weight source's right scalar is not the physical
   * energy derivative. Full W' contributes to H@right, including occupied-block
   * energy response; no raw-matrix symmetrization/translation repair is used.
   * All fields publish together only after convergence and final full-frame/
   * actual external right authentication. Output may alias right, never a
   * borrowed frame or geometry. Calls may not overlap/reenter.
   */
  HostKsHessianDiagnostic apply(std::span<const double> right, std::span<double> output,
                                const response::GmresOptions& options);
  /** Optional matrix-endpoint first-source reuse. Admission allocates only;
   * callers must recheck complete work caps before preparing any integrals.
   * A false admission retains the bounded streamed schedule unchanged.
   */
  bool admit_coordinate_sources(const response::GmresOptions& options);
  void prepare_coordinate_sources();
  /** Additional explicit reservation for one source-driven frozen integral
   * matrix plus its replay staging. Requires admitted coordinate first sources.
   * Admission allocates/authenticates only, with no primitive/congruence work;
   * numerical/work errors never trigger fallback. ProjectedMatrix requires
   * explicitly supported density scales and owns its additional snapshots.
   */
  bool admit_integral_matrix(HostNuclearIntegralPlan allowance,
                             HostNuclearIntegralCurvatureSchedule schedule);
  void prepare_integral_matrix();
  /** Best-effort exact XC partial and native AO-prefix allocation inside the
   * already reserved response child, after selecting the integral schedule.
   */
  void admit_response_cache(const response::GmresOptions& options);
  /** Select at most four independent coordinate solves before caches/source
   * work. Retained D/W/right/column panels fit the original assembly/aggregate
   * caps; insufficient capacity or optional allocation failure retains width one.
   */
  std::size_t admit_response_batch(const response::GmresOptions& options,
                                   std::size_t maximum_directions = 4);
  /** Whole-coordinate consumer of the independent response batch. Assembly
   * reuses the ordinary physical column owner, without symmetry repair. Columns
   * become readable only when every response and assembly in the group passes.
   */
  std::vector<HostKsHessianDiagnostic> apply_coordinates(std::size_t first, std::size_t count,
                                                         const response::GmresOptions& options);
  /** Borrowed until the next action/admission; no caller or extra output panel
   * is hidden outside the admitted batch assembly arena.
   */
  [[nodiscard]] std::span<const double> coordinate_column(std::size_t index) const;
  [[nodiscard]] HostKsHessianResources resources(const response::GmresOptions& options) const;
  [[nodiscard]] HostKsHessianWork work() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft
