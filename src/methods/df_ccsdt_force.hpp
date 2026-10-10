#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cc/df_source_response.hpp"
#include "cc/df_triples.hpp"
#include "cc/lambda_response.hpp"
#include "hf/rhf_frame_response.hpp"
#include "methods/df_hf_guess.hpp"
#include "methods/rccsd_method.hpp"

namespace generativeqc::methods::detail {
/** Optional density/control handoff and benchmark-only exact-reference diagnostic.
 * No DF Fock/orbitals enter correlation or response. With no explicit density,
 * the complete owner may prepare a density in its qualified default domain.
 * The exact reference solver starts fresh DIIS and retains its original gates.
 * A refused/unconverged seed retries the ordinary cold Direct reference.
 * The detached seed must outlive the call. Clear the output reference before
 * reusing this diagnostic object so a prior frame is not an uncharged owner. */
struct DFCCSDTReferenceExperiment {
  const std::vector<double>* initial_density{};
  bool disable_preconvergence{};
  bool seed_fallback{};
  std::shared_ptr<const hf::PhysicalReference> reference;
  scf::PrecisionProvenance work;
};

/** Complete native endpoint on an unchanged conventional RHF reference with
 * a DF correlation Hamiltonian. The public DF-RCCSD(T) selector reuses this
 * owner for both energy and force requests. */
struct DFCCSDTResult {
  DFHFGuess reference_guess;
  // If true, total_seconds includes a resource-refused precursor whose work
  // counters are unavailable; successful-attempt counters are not endpoint totals.
  bool recycling_discarded_primal_attempt{};
  double energy{}, reference_energy{}, correlation_energy{}, triples_energy{};
  // Final RHF convergence diagnostics from the original cold reference, without replay.
  double reference_energy_change{}, reference_density_rms{};
  int reference_iterations{};
  std::vector<double> forces;
  Result method_result;
  generativeqc_correlation_diagnostic correlation{};
  std::size_t numeric_capacity_bytes{}, source_weight_values{}, metric_weight_values{};
  double total_seconds{}, triples_seconds{}, lambda_seconds{}, source_response_seconds{},
      orbital_seconds{};
  CcPerformanceDiagnostic primal;
  cc::SolverDiagnostic solver;
  cc::LambdaDiagnostic lambda;
  hf::RHFFrameResponseResult orbital;
  cc::triples::DFCudaResult triples;
  cc::triples::DFGapReductionDiagnostic triples_gap;
  cc::triples::DFScalarFusionDiagnostic triples_scalar_fusion;
  cc::triples::DFCudaFockResult triples_fock;
};

/** Cold normalized geometry -> exact native CUDA RHF -> DF source -> CCSD(T)
 * -> complete nuclear forces. Retains the exact source/frame only when forces
 * are requested, and consumes it before either input geometry can change.
 * Energy-only calls share the same scientific Hamiltonian and solver gates.
 * The optional CCSD-only mode omits triples and is an independent closure gate.
 * Disabling df_auxiliary_reduction retains expanded Lambda actions for matched
 * endpoint validation; it changes only the response schedule, not the method.
 * Disabling df_matrix_gemm selects the scalar DF residual for matched energy
 * and force endpoint comparisons with the same compiled library.
 * All phase bounds charge simultaneously live owners; no CPU integral/CC
 * reference fallback or four-index full MO Hamiltonian is used.
 * request_triples_gap_cotangents is a matched-schedule control: full-Fock
 * response replaces these diagonal sources, so their optional computation
 * does not contribute to molecular forces. Fixed-canonical consumers still
 * request them from the lower-level triples response API.
 * The complete molecular-force owner defaults to the bounded parallel gap
 * reduction and omits diagonal gap cotangents because full-Fock response
 * replaces them. Serial/all-output execution remains available explicitly for
 * matched validation; neither selector weakens numerical acceptance gates.
 * fused_triples_scalar_response is an opt-in generated primal/W/V region for
 * gap-free force response. Optional seed storage is admitted before allocation;
 * a tight budget retains the original schedule without changing output demand.
 * admitted_triples_w is an internal experimental forward-W admission only;
 * RHF, CCSD, reverse contractions, Lambda and orbital/nuclear response remain
 * FP64. It never changes the public precision mode or numerical gates.
 * lambda_true_residual_interval controls only periodic GMRES residual replay.
 * Predicted convergence, restarts and final acceptance still require a fresh
 * FP64 residual and an independent physical-equation audit. The complete
 * DF-CCSD(T) owner defaults to 30; explicitly selecting 1 retains the
 * historical per-iteration baseline. Standalone Lambda and shared GMRES
 * defaults remain unchanged; the interval must be positive.
 * lambda_core_reuse retains compiler-proven primal intermediates in one
 * immutable Lambda owner. Its complete storage is optional; disabling it or
 * refusing its budget/allocation preserves the original FP64 matrix actions.
 * lambda_audit_matrix lowers the original independent expanded equations
 * through bounded FP64 GEMM; resource refusal retains their scalar schedule.
 * Lambda defaults request batch 32 and original-graph matrix fresh replay.
 * Complete host/device budgets and currently free VRAM admit actual storage;
 * smaller batches and the original scalar replay remain bounded fallbacks.
 * ccsd_batch_limit=0 selects the endpoint default: 32 for energy-only calls,
 * eight for forces. Positive limits remain explicit overrides. The solver
 * admits the actual tile under its existing full budget and dimension checks;
 * this does not change standalone CCSD or any response precision policy.
 */
DFCCSDTResult run_df_ccsdt_native(
    runtime::ExecutionContext&, const core::System& orbital, const core::System& auxiliary,
    const generativeqc_method_descriptor&, bool forces = true, bool with_triples = true,
    bool df_auxiliary_reduction = true, bool df_matrix_gemm = true, bool lambda_matrix_gemm = true,
    std::size_t lambda_batch_limit = 32, std::size_t ccsd_batch_limit = 0,
    const hf::RHFFrameResponseOptions& frame_options = {}, bool derived_denominators = true,
    bool packed_diis = false, bool parallel_gap_reduction = true,
    bool request_triples_gap_cotangents = false, bool fused_triples_scalar_response = false,
    runtime::PrecisionDirective admitted_triples_w = {},
    std::size_t lambda_true_residual_interval = 30, bool lambda_core_reuse = true,
    bool lambda_audit_matrix = true, bool lambda_primal_matrix = true,
    DFCCSDTReferenceExperiment* reference_experiment = nullptr);

/** Ordered existing host boundaries for diagnostic bit-pattern comparisons.
 * Empty payloads remain distinguishable through their explicit element counts.
 * These identities are not numerical acceptance tests or cache keys. */
inline constexpr std::array<const char*, 35> df_gap_fingerprint_names{
    "triples_bov",      "triples_bvv",
    "triples_ovoo",     "triples_ovov",
    "triples_fov",      "triples_t1",
    "triples_t2",       "triples_foo",
    "triples_fvv",      "lambda1",
    "lambda2",          "parameter_foo",
    "parameter_fov",    "parameter_fvv",
    "parameter_ovov",   "parameter_ovvo",
    "parameter_oovv",   "parameter_ovoo",
    "parameter_oooo",   "parameter_bov",
    "parameter_bvv",    "factor_boo",
    "factor_bov",       "factor_bvv",
    "fock_source",      "coefficient_source",
    "df_gradient",      "orbital_rhs",
    "orbital_solution", "hcore_weights",
    "overlap_weights",  "fock_ao_weights",
    "orbital_gradient", "stationarity",
    "complete_forces"};

/** Fixed scalar metadata only: no numeric owner, extra download or retained
 * intermediate. Fingerprinting time is included in diagnostic response times. */
struct DFGapResponseFingerprints {
  std::array<std::uint64_t, df_gap_fingerprint_names.size()> identities{};
  std::array<std::size_t, df_gap_fingerprint_names.size()> elements{};
  std::size_t value_reads{};
  double seconds{};
};

/** One complete response composition on a shared native primal. Timings exclude
 * the common cold solve, so they must not be presented as cold endpoint times. */
struct DFGapForceSnapshot {
  std::vector<double> forces;
  double energy{}, triples_energy{}, orbital_residual{}, maximum_stationarity{};
  double clone_seconds{}, response_seconds{}, triples_seconds{}, lambda_seconds{},
      source_response_seconds{}, orbital_seconds{};
  std::size_t numeric_capacity_bytes{}, source_weight_values{}, metric_weight_values{};
  std::size_t orbital_iterations{}, orbital_actions{}, fock_response_work{};
  cc::LambdaDiagnostic lambda;
  cc::triples::DFCudaResult triples;
  cc::triples::DFGapReductionDiagnostic gap;
  DFGapResponseFingerprints fingerprints;
};

/** Diagnostic-only serial/all, parallel/all, omitted and repeated serial compositions.
 * The source token and bitwise nine-input identity are common to all cases.
 * Retained physical/reference owners are shared, not rebuilt or copied. */
struct DFGapForceComparison {
  std::array<DFGapForceSnapshot, 4> cases;
  std::size_t nocc{}, nvir{}, naux{};
  std::uint64_t source_identity{}, denominator_identity{}, primal_identity{};
  std::size_t retained_primal_host_bytes{}, retained_df_source_bytes{},
      retained_exact_source_bytes{}, output_bytes{}, clone_admission_bytes{};
  double total_seconds{}, primal_seconds{}, reference_energy{}, reference_energy_change{},
      reference_density_rms{};
  int reference_iterations{};
  CcPerformanceDiagnostic primal;
  cc::SolverDiagnostic solver;
};

/** Solve native CUDA RHF/DF-CCSD once, then consume independently owned host
 * copies through the very same complete force implementation. This isolates
 * response schedules from cold-reference variability; it is not a substitute
 * for cold endpoint qualification. Requires an explicit positive numeric budget
 * and refuses recycled Z guesses. Fixed output buffers are charged during the
 * cold solve; original host copies and shared sources remain charged until the
 * entire diagnostic returns. Any failure publishes no partial comparison. */
DFGapForceComparison diagnose_df_ccsdt_gap_schedules(
    runtime::ExecutionContext&, const core::System& orbital, const core::System& auxiliary,
    const generativeqc_method_descriptor&, const hf::RHFFrameResponseOptions& frame_options = {},
    std::size_t lambda_batch_limit = 8, std::size_t ccsd_batch_limit = 8,
    bool derived_denominators = true, bool packed_diis = false);

/** A physical-branch replay, not another triples/Lambda composition. Host
 * fingerprints retain no weight matrices; streamed device weights are copied
 * through one explicitly admitted reusable buffer for the diagnostic census. */
struct DFPhysicalResponseSnapshot {
  std::vector<double> df_gradient, orbital_gradient;
  DFGapResponseFingerprints fingerprints;
  std::array<std::uint64_t, 2> weight_identities{};
  std::array<std::size_t, 2> weight_elements{};
  std::size_t weight_transfer_bytes{}, weight_buffer_bytes{}, numeric_capacity_bytes{};
  std::size_t orbital_iterations{}, orbital_actions{};
  double source_seconds{}, orbital_seconds{}, weight_census_seconds{};
  double orbital_residual{}, maximum_stationarity{};
};

/** Four native physical responses on one immutable reference and factor seed,
 * followed by one complete force publication. All orbital replays use the
 * final coefficient source, not the independently computed DF gradients.
 * No scientific equations, defaults, or acceptance tolerances are changed. */
struct DFPhysicalResponseComparison {
  std::array<DFPhysicalResponseSnapshot, 4> cases;
  std::vector<double> forces;
  std::uint64_t source_identity{}, reference_identity{}, factor_seed_identity{},
      orbital_seed_identity{};
  std::size_t nocc{}, nvir{}, naux{}, output_bytes{}, numeric_capacity_bytes{};
  int reference_iterations{};
  double energy{}, total_seconds{}, triples_seconds{}, lambda_seconds{};
  CcPerformanceDiagnostic primal;
  cc::LambdaDiagnostic lambda;
};

/** Diagnostic-only late-phase replay through the existing complete force owner.
 * Requires CUDA, a positive explicit budget, and ordinary unrecycled diagonal
 * orbital response. Output storage is reserved before the cold solve; scratch,
 * device-to-host weight transfers and synchronization are explicit and included
 * in phase/endpoint timing. This does not qualify a cold endpoint speedup. */
DFPhysicalResponseComparison diagnose_df_ccsdt_physical_responses(
    runtime::ExecutionContext&, const core::System& orbital, const core::System& auxiliary,
    const generativeqc_method_descriptor&, const hf::RHFFrameResponseOptions& frame_options = {});
}  // namespace generativeqc::methods::detail
