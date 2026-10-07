#ifndef GENERATIVEQC_SCF_TYPES_HPP
#define GENERATIVEQC_SCF_TYPES_HPP

#include <memory>
#include <optional>
#include <vector>

#include "dft/density_source.hpp"
#include "dft/scf_diagnostic.hpp"
#include "generativeqc/generativeqc.h"
#include "hf/reference.hpp"
#include "scf/fock_build.hpp"
#include "scf/initial_guess/preliminary_types.hpp"
#include "scf/precision_work.hpp"

namespace generativeqc::scf {

struct ScfHooks;

/** Diagnostics for the opt-in exact incremental Direct-J/K controller.
 *
 * The controller changes only the density presented to the same exact provider:
 * after one full anchor build, later accepted SCF iterates evaluate J/K on
 * delta-D and add that result to the retained anchor matrices. Trial/proposal
 * and post-SCF physical validation builds deliberately bypass the anchor.
 */
struct IncrementalDirectJkDiagnostic {
  std::uint32_t policy_version{1};
  bool requested{};
  bool active{};
  /** Full builds that establish/refresh the accepted-iterate anchor. */
  std::uint64_t anchor_full_builds{};
  /** Exact provider applications to delta-D for accepted SCF iterates. */
  std::uint64_t delta_builds{};
  /** Periodic accepted-iterate rebuilds after an existing anchor. */
  std::uint64_t periodic_rebuilds{};
  /** Full proposal/audit builds that never mutate the accepted anchor. */
  std::uint64_t bypass_full_builds{};
  /** Strict full physical builds performed by ordinary finalization. */
  std::uint64_t post_scf_full_builds{};
  /** Number of times a successful delta build became the next anchor. */
  std::uint64_t anchor_updates{};
  /** Largest absolute alpha/beta delta-density element observed. */
  double max_abs_delta_density{};
  /** Fixed-topology compaction counters below are complete for this item. */
  bool quartet_work_counters_valid{};
  /** Candidate shell quartets visited by full anchor/refresh builds. */
  std::uint64_t full_candidate_shell_quartets{};
  /** Candidate shell quartets rejected before Direct-J/K consumption. */
  std::uint64_t full_rejected_shell_quartets{};
  /** Shell quartets admitted by the physical screening predicate. */
  std::uint64_t full_admitted_shell_quartets{};
  /** AO quartet tiles materialized from admitted full-density shell quartets. */
  std::uint64_t full_admitted_quartet_tiles{};
  /** Candidate shell quartets visited by incremental delta-density builds. */
  std::uint64_t delta_candidate_shell_quartets{};
  /** Delta shell quartets rejected before Direct-J/K consumption. */
  std::uint64_t delta_rejected_shell_quartets{};
  /** Delta shell quartets admitted by the physical screening predicate. */
  std::uint64_t delta_admitted_shell_quartets{};
  /** AO quartet tiles materialized from admitted delta-density shell quartets. */
  std::uint64_t delta_admitted_quartet_tiles{};
};

/** How the requested floating-point precision policy actually resolved. */
struct PrecisionProvenance {
  uint32_t policy_version{1};
  /** Requested mode (\p generativeqc_precision_mode). */
  int32_t requested_mode{GENERATIVEQC_PRECISION_FP64};
  /** Effective Fock bits: 64 for FP64, 32 when a mixed route is active. */
  uint32_t effective_bits{64};
  /** Tile threshold for \p auto; zero when the mixed route is not active. */
  double mixed_precision_fock_threshold{0.0};
  /** The strict FP64 target refinement ran at the end of the run. */
  bool strict_refinement_applied{false};
  /** Accumulated FP32 Fock rounding the admission budget certified; 0 if none. */
  double mixed_precision_reserved_error{0.0};
  /** FP64 target-precision iterations run after the mixed iterative stage. */
  uint32_t refinement_iterations{0};
  /** Complete mixed-stage Fock/operator applications for this item. */
  uint64_t mixed_stage_fock_builds{0};
  /** Complete strict-FP64 SCF-stage Fock/operator applications for this item. */
  uint64_t strict_stage_fock_builds{0};
  /** Additional strict physical-Fock builds performed after SCF convergence. */
  uint64_t post_scf_fock_builds{0};
  /** Whole-execution provider retries before this successful/returned attempt. */
  uint64_t execution_retries{0};
  /** Certified per-item mixed-capable census used by the admission budget. */
  uint64_t mixed_admission_census{0};
  /** Exact final physical-residual audits executed for this item. */
  uint64_t final_residual_audits{0};
  /** Final-Fock operator applications skipped by retained-state reuse. */
  uint64_t skipped_final_fock_builds{0};
  /** Nonzero only when the operator-work counters above are fully instrumented.
   * Numerical failures can leave partially executed stages uncounted; their
   * counters are not certified by this flag. */
  uint32_t operator_work_counters_valid{0};
};

/** Numerical controls shared by the implemented mean-field solvers. */
struct ScfOptions {
  unsigned max_iterations{100};
  unsigned diis_history{8};
  double energy_tolerance{1.0e-10};
  double density_tolerance{1.0e-8};
  double screening_tolerance{1.0e-12};
  /** Experimental #990 exact incremental Direct-J/K controller. Off by
   * default. It is admitted only when every requested J/K term is Exact.
   * The final physical state still uses ordinary full provider builds. */
  bool incremental_direct_jk{};
  /** Accepted delta updates before refreshing the full anchor. Zero disables
   * periodic refresh; strict post-SCF full rebuilds are never disabled. */
  unsigned incremental_direct_jk_rebuild_interval{8};
  /** For screened CUDA lowers, require the previous accepted density RMS to be
   * at or below this positive threshold before using a ΔD build. Zero preserves
   * the cadence-only controller. */
  double incremental_direct_jk_density_rms_threshold{};
  /** Select the DF solver; direct four-center remains the default. */
  generativeqc_density_fitting_mode density_fitting_mode{GENERATIVEQC_DENSITY_FITTING_NONE};
  /** Relative cutoff used when factoring the auxiliary Coulomb metric. */
  double density_fitting_relative_threshold{1.0e-10};
  /** Byte budget for bounded DF plan/integral work; zero means implementation default. */
  std::size_t density_fitting_memory_budget_bytes{};
  /** Correlated energy consumers require values-only, bounded direct RHF. */
  bool export_physical_reference{false};
  std::size_t reference_memory_budget_bytes{};
  /**
   * Requested floating-point execution policy. \p std::nullopt (absent) preserves
   * the legacy GENERATIVEQC_MIXED_PRECISION_FOCK_THRESHOLD diagnostic switch; an
   * explicit \p GENERATIVEQC_PRECISION_FP64 keeps the pure double path; an explicit
   * \p GENERATIVEQC_PRECISION_AUTO derives the mixed Fock threshold from the
   * tolerances and finishes with a strict FP64 target refinement.
   */
  std::optional<generativeqc_precision_mode> precision_mode{};
  /** Resolved once; old internal callers may leave this unset for direct HF. */
  std::optional<ResolvedFockBuild> resolved_fock_build;
  /** Explicit synchronous CPU proposal/trace opt-in; null has no snapshot work. */
  ScfHooks* hooks{};
  /** Diagnostic proposal bridge rejects malformed seeds instead of normalizing them. */
  bool strict_initial_density{};
  /** False for an explicit energy-only endpoint. Backends must then omit
   * derivative evaluation and return an empty force vector. */
  bool compute_forces{true};
  /** Internal native RKS candidate override; public/default execution stays D. */
  dft::XcDensityRoute xc_density_route{dft::XcDensityRoute::DensityMatrix};
  /** Bounded AO/XC tile schedule; does not alter the grid or functional. */
  std::size_t xc_tile_points{256};
  /** #237 experimental CPU PBE-RKS anchor/update path. Default-off and not
   * exposed by the public descriptor until complete-solve benefit is proven. */
  bool experimental_incremental_xc{};
  /** Maximum exact anchor-relative updates before a transactional full rebuild. */
  std::size_t incremental_xc_max_updates{4};
  /** Rebuild when RMS(D-D0) exceeds this run-local anchor drift bound. */
  double incremental_xc_max_density_rms{5.0e-2};
  /** Below this nonzero anchor-relative RMS, prefer a full build rather than
   * subtracting nearly identical matrices. Zero disables the noise trigger. */
  double incremental_xc_noise_density_rms{1.0e-14};
  /** Consecutive non-improving physical-residual observations before forcing
   * a full accepted-state rebuild. Zero disables the stagnation trigger. */
  std::size_t incremental_xc_stagnation_iterations{4};
  enum class XcExecutionSchedule : std::uint32_t { DeviceFused = 0, HostUnfused = 1 };
  /** Placement-only semilocal XC schedule; scientific identity is unchanged. */
  XcExecutionSchedule xc_execution_schedule{XcExecutionSchedule::DeviceFused};
  /** Resolved semilocal component scales; exact exchange lives only in the
   * common FockBuildSpec. Unit defaults preserve legacy LDA/PBE callers. */
  double semilocal_exchange_scale{1.0};
  double semilocal_correlation_scale{1.0};
  /** Retain the already evaluated CPU RKS F[D] for an explicit snapshot read.
   * No extra Fock build, canonicalization or W is performed by energy-only SCF. */
  bool retain_ks_state{};
  /** Explicit bounded cold-start preparation; absent preserves the core guess. */
  std::optional<initial_guess::PreliminaryOptions> preliminary_guess;
};

/** Lower-specific facts consumed by the shared incremental Direct-J/K policy.
 *
 * These capabilities describe execution semantics, not backend identity. CPU
 * and CUDA therefore resolve the same accepted-iterate policy before lowering,
 * while a lower that performs density-weighted screening can request the
 * stricter refresh cadence needed to bound omitted delta contributions.
 */
struct IncrementalDirectJkCapabilities {
  bool provider_eligible{};
  bool density_weighted_screening{};
  bool conflicting_precision_policy{};
};

/** Backend-neutral accepted-iterate policy for exact incremental Direct-J/K. */
struct IncrementalDirectJkPolicy {
  bool requested{};
  bool active{};
  unsigned requested_rebuild_interval{};
  unsigned effective_rebuild_interval{};
  double density_rms_threshold{};
};

inline bool direct_jk_incremental_exact_eligible(const ResolvedFockBuild& strategy) noexcept {
  const auto exact = [](const FockTermSpec& term) {
    return !term.present || term.approximation == FockApproximation::Exact;
  };
  return (strategy.spec.coulomb.present || strategy.spec.exchange.present) &&
         exact(strategy.spec.coulomb) && exact(strategy.spec.exchange);
}

/** Resolve one policy before CPU/CUDA lowering.
 *
 * A density-weighted screened lower is allowed at most one accepted delta build
 * before a full-density refresh. This keeps screening omissions from accumulating
 * through an arbitrarily long anchor chain. Exact-linear lowers retain the
 * caller's requested cadence, including zero for no periodic refresh.
 */
inline IncrementalDirectJkPolicy resolve_incremental_direct_jk_policy(
    const ScfOptions& options, IncrementalDirectJkCapabilities capabilities) noexcept {
  IncrementalDirectJkPolicy policy;
  policy.requested = options.incremental_direct_jk;
  policy.requested_rebuild_interval = options.incremental_direct_jk_rebuild_interval;
  policy.density_rms_threshold = options.incremental_direct_jk_density_rms_threshold;
  policy.effective_rebuild_interval =
      capabilities.density_weighted_screening && options.screening_tolerance != 0.0
          ? 1U
          : options.incremental_direct_jk_rebuild_interval;
  policy.active = policy.requested && capabilities.provider_eligible &&
                  !capabilities.conflicting_precision_policy;
  return policy;
}

inline bool direct_jk_incremental_requires_full_build(
    bool anchored, unsigned delta_updates_since_full,
    const IncrementalDirectJkPolicy& policy) noexcept {
  return !anchored || (policy.effective_rebuild_interval != 0U &&
                       delta_updates_since_full >= policy.effective_rebuild_interval);
}

/** Internal mean-field result, including state retained for warm starts. */
/** Transitional compatibility alias. New HF/post-HF code should use hf::PhysicalReference. */
using PhysicalReference = hf::PhysicalReference;

struct ScfResult {
  double energy{};
  std::vector<double> forces;
  // RHF stores one N x N AO density; UHF stores alpha then beta matrices.
  // The state is explicit and remains private to prepared execution plans.
  std::vector<double> density;
  /** Present only for a successful CPU RKS/UKS solve requesting state retention. */
  std::vector<double> ks_physical_fock;
  unsigned iterations{};
  double energy_change{};
  double density_rms{};
  /** Physical commutator RMS at the retained density, over all spin entries.
   * KS separately gates the maximum per-spin value in dft_diagnostic.
   * Zero means the backend does not report this field separately. */
  double physical_residual_rms{};
  bool converged{};
  bool initial_density_used{};
  /** CPU physical operator evaluations, counting a joint UHF J/K as one build. */
  std::size_t fock_builds{};
  /** Exact incremental Direct-J/K execution provenance, when requested. */
  IncrementalDirectJkDiagnostic incremental_direct_jk{};
  /**
   * How the requested precision policy resolved. Set by the backend that can
   * report it (the CUDA mixed-precision route); the strict FP64 default
   * leaves the FP64 defaults in place.
   */
  PrecisionProvenance precision{};
  /** Execution-owned detailed precision timeline/operator census. It may remain
   * partial even when the aggregate provenance above is available. */
  PrecisionWork precision_work{};
  std::shared_ptr<const PhysicalReference> reference;
  /** Current immutable RKS factor when explicitly requested, including on a
   * nonconverged return. Its witness matches the returned density exactly. */
  std::shared_ptr<const OccupiedDensityFactor> xc_density_factor;
  dft::RksDensityDiagnostic xc_density_diagnostic;
  /** Populated by KS solvers; HF diagnostics and stopping rules are unchanged. */
  dft::ScfDiagnostic dft_diagnostic;
  initial_guess::PreliminaryDiagnostic preliminary_guess;
};

}  // namespace generativeqc::scf

#endif
