#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>

#include "dft/host_ks_perturbation.hpp"
#include "dft/host_nuclear_integrals.hpp"
#include "dft/host_xc_nuclear.hpp"
#include "integrals/ecp.hpp"

namespace generativeqc::dft {

/** Explicit per-direction source limits and composed owned-array ceiling. */
struct HostKsNuclearLimits {
  std::size_t max_numeric_bytes{}, perturbation_map_terms{}, assembly_map_terms{};
  HostNuclearIntegralOptions integrals;
  HostGgaNuclearPlan xc;
  integrals::EcpDirectionLimits ecp;
  /** Optional additional first-source panel, inside max_numeric_bytes, not an
   * increase to the original streamed integral child's numeric cap. Zero keeps
   * the bounded streamed schedule. Admission precedes all source work.
   */
  std::size_t coordinate_source_bytes{};
};

struct HostKsNuclearResources {
  std::size_t fixed_bound_bytes{}, gmres_workspace_bytes{}, assembly_map_terms{};
  bool admitted{};
  std::size_t response_ao_cache_bytes{}, response_ao_prefix_points{};
};

struct HostKsNuclearWork {
  std::size_t source_attempts{}, sources_completed{}, solve_attempts{}, solves_completed{};
  std::size_t induced_fock_actions{}, response_xc_point_visits{};
  std::size_t response_xc_evaluated_points{}, response_xc_cached_points{};
  std::size_t response_ao_tile_reads{}, response_ao_evaluated_points{}, response_ao_cached_points{};
  HostNuclearIntegralWork integrals;
  HostGgaNuclearWork xc;
  integrals::EcpDirectionWork last_ecp;
};

struct HostKsNuclearRequest {
  std::span<const double> direction;
  std::span<double> density_direction, weighted_density_direction;
};

struct HostKsNuclearBatchResult {
  std::vector<HostKsPerturbationDiagnostic> responses;
  /** Actual completed first-ECP work per physical direction, not the last
   * source repeated across a batch in an endpoint's semantic work receipt.
   */
  std::vector<integrals::EcpDirectionWork> first_ecp_work;
};

/** Native nuclear S'/F' and coupled D'/W' around one verified KS state.
 * Owns the fixed-grid response reader so its J/K/XC action and moving-grid
 * source cannot accidentally bind different grid readers. All sources borrow
 * the same normalized system/basis, prepared Fock and immutable density.
 * F' includes ionic attraction, residual ECP, full-range J/K and moving XC.
 * Integral/source algebra and metric/response maps remain compiler-owned.
 * No displaced production evaluation, reference engine or runtime compiler.
 *
 * This is first-order electronic nuclear response, NOT a complete Hessian or
 * transition-state endpoint. Nuclear repulsion/D3 and second-source derivatives
 * remain separate. The explicit ceiling includes owned source/response panels,
 * ECP workspace, assembly staging and GMRES, not borrowed stationary/provider
 * storage, BLAS/scalar stack, allocator overhead, RSS or total endpoint time.
 * The verified state, system, prepared Fock, geometry and their authenticated
 * owner must outlive this synchronous consumer. Calls may not overlap/reenter.
 */
class HostKsNuclearResponse {
 public:
  HostKsNuclearResponse(const VerifiedKsFinalState& state, const core::System& system,
                        const scf::PreparedFockPlan& fock, HostGridGeometry& geometry,
                        std::string_view method_identity, HostKsNuclearLimits limits,
                        HostKsResponse::ValidateCurrent validate_current);
  ~HostKsNuclearResponse();
  HostKsNuclearResponse(const HostKsNuclearResponse&) = delete;
  HostKsNuclearResponse& operator=(const HostKsNuclearResponse&) = delete;

  /** Atom/xyz input; symmetric AO S' and spin/AO F' publish together. */
  void source(std::span<const double> direction, std::span<double> overlap_direction,
              std::span<double> frozen_fock);
  /** Spin/AO D' and full energy-weighted-density W' publish only on convergence.
   * Outputs may alias external direction, never one another or borrowed state.
   */
  HostKsPerturbationDiagnostic solve(std::span<const double> direction,
                                     std::span<double> density_direction,
                                     std::span<double> weighted_density_direction,
                                     const response::GmresOptions& options);
  /** Independent nuclear sources and compiler-owned D/W maps; only iterative
   * actions share execution. All direction/source/output panels and simultaneous
   * solves fit the original composed ceiling before any source runs. Outputs
   * publish together only when every fresh physical residual passes.
   */
  HostKsNuclearBatchResult solve_batch(std::span<const HostKsNuclearRequest> requests,
                                       const response::GmresOptions& options);
  /** Allocate only, before any source/solve. False selects the original bounded
   * schedule for insufficient reserved storage, allocation failure or density
   * range. Numerical preparation failures are never converted to fallback.
   */
  bool admit_coordinate_sources(const response::GmresOptions& options);
  /** Allocate exact XC partials, then a whole-tile native AO prefix from the
   * remaining composed cap, reserving actual GMRES first. Returns all newly
   * owned numeric bytes. Optional allocation failures keep streamed fallbacks;
   * no source preparation runs and neither cache replaces frame authentication.
   */
  std::size_t admit_response_cache(const response::GmresOptions& options,
                                   std::size_t directions = 1);
  /** Perform the admitted single source sweep; failures retain actual work. */
  void prepare_coordinate_sources();
  /** Prepared integral first-source panel only; supplied CPKS D'/W' contract
   * into the physical-coordinate moving-density contribution, without new
   * primitive work. Owner/input/alias checks still precede publication.
   */
  void integral_density_response(std::span<const double> density_right,
                                 std::span<const double> weighted_density_right,
                                 std::span<double> output);
  [[nodiscard]] HostKsNuclearResources resources(const response::GmresOptions& options,
                                                 std::size_t directions = 1) const;
  [[nodiscard]] HostKsNuclearWork work() const noexcept;
  /** Authenticate the full stationary frame without another source/solve. */
  void validate_current() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft
