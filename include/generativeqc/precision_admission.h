#ifndef GENERATIVEQC_PRECISION_ADMISSION_H
#define GENERATIVEQC_PRECISION_ADMISSION_H

#include "generativeqc/generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Context for the versioned application-precision evidence policy.
 * Boolean fields are 0/1. census_presence is 0 for missing metadata, 1 for
 * explicitly available counters, and 2 for explicitly unavailable counters.
 * requested_controls and standard_controls are energy/density/screening
 * tolerances in that order. Zero denotes unavailable controls or allowances;
 * zero force_rms denotes an unspecified RMS budget. All doubles are finite
 * and nonnegative. Model/gradient qualification is supplied by its owning
 * capability; this query cannot qualify a new method or arithmetic route.
 */
typedef struct generativeqc_precision_admission_context_v1 {
  uint32_t struct_size, abi_version;
  uint32_t ks_model, full_range_exchange, range_exchange, nonlocal_correlation;
  uint32_t hartree_fock, density_fitted, warm_start_used, warm_start_fallback;
  uint32_t gradient_present, gradient_fp64, gradient_qualified, census_presence;
  uint32_t converged, cuda_execution, fp64_candidate, reserved;
  double energy_abs, force_max_abs, force_rms;
  double requested_controls[3], standard_controls[3];
} generativeqc_precision_admission_context_v1;

/** Policy decisions are not per-call observable error estimates. Missing or
 * incomplete evidence yields false decisions. A strict warm decision requires
 * strict-Fock and final-audit events at the exact returned owner/epoch/generation.
 */
typedef struct generativeqc_precision_admission_v1 {
  uint32_t struct_size, abi_version, policy_version;
  uint32_t budget_fock, strict_warm, standard_fallback;
} generativeqc_precision_admission_v1;

/** @native-contract generativeqc_precision_admission_query_v1
 * @behavior Evaluate shared budget-Fock, strict warm and Standard fallback policy.
 * @inputs Nonnull versioned context, provenance, detail and output; event/operator arrays
 * contain exactly the counts in detail. Missing mixed-build metadata uses UINT64_MAX;
 * unavailable effective bits use zero. Supplied evidence and model traits must describe
 * the same completed result. No model or numerical qualification is inferred here.
 * @outputs Transactionally writes policy version 1 and three boolean decisions.
 * @errors INVALID_ARGUMENT for invalid flags, nonfinite/negative controls or inconsistent
 * array/count pairs; ABI_MISMATCH for invalid descriptors; NOT_IMPLEMENTED for an unknown
 * detail version. Output is unchanged on failure.
 * @lifetime All input/output storage is borrowed only during the call; no pointer is retained.
 * @execution Synchronous allocation-free host query; no context, GPU, compiler or solve is
 * needed. Explicit observable targets still require their independent endpoint audit.
 * @units Budgets use Eh and Eh/bohr; solver controls retain their existing native units;
 * counts denote actual work, never inferred iterations or timings.
 */
GENERATIVEQC_API generativeqc_status generativeqc_precision_admission_query_v1(
    const generativeqc_precision_admission_context_v1* context,
    const generativeqc_precision_provenance* provenance,
    const generativeqc_precision_work_detail* detail,
    const generativeqc_precision_work_event* events,
    const generativeqc_precision_operator_record* operators,
    generativeqc_precision_admission_v1* output);

enum {
  GENERATIVEQC_PRECISION_CANDIDATE_AUDIT = 0,
  GENERATIVEQC_PRECISION_CANDIDATE_STRICT = 1,
  GENERATIVEQC_PRECISION_CANDIDATE_FOCK = 2,
  GENERATIVEQC_PRECISION_CANDIDATE_WARM_FOCK = 3,
  GENERATIVEQC_PRECISION_CANDIDATE_MIXED_GRADIENT = 4,
  GENERATIVEQC_PRECISION_CANDIDATE_STRICT_GRADIENT = 5,
  GENERATIVEQC_PRECISION_CANDIDATE_STRICT_XC = 6,
  GENERATIVEQC_PRECISION_CANDIDATE_STANDARD = 7
};
enum {
  GENERATIVEQC_PRECISION_CANDIDATE_GRADIENT_QUALIFIED = 1,
  GENERATIVEQC_PRECISION_CANDIDATE_GEOMETRY_FALLBACK = 2,
  GENERATIVEQC_PRECISION_CANDIDATE_CORRECTION_UNQUALIFIED = 4,
  GENERATIVEQC_PRECISION_CANDIDATE_GRADIENT_RETRIED = 8,
  GENERATIVEQC_PRECISION_CANDIDATE_GRID_UNAVAILABLE = 16,
  GENERATIVEQC_PRECISION_CANDIDATE_EXECUTION_RETRIED = 32,
  GENERATIVEQC_PRECISION_CANDIDATE_STANDARD_FALLBACK = 64,
  GENERATIVEQC_PRECISION_CANDIDATE_BUDGET_AUDIT = 128
};

/** Detached facts for completed candidate classification, not new qualification.
 * Boolean fields are 0/1. budget_requested identifies an explicit arithmetic
 * allowance, not a frontend profile; budget_native/budget_fock describe actual
 * admitted execution. Equal targets and work receive equal decisions regardless
 * of preset names. Gradient owner: 0 unknown, 1 stationary CUDA,
 * 2 composite stationary CUDA. Returned mode: 0 unknown, 1 mixed, 2 FP64.
 * correction_result_flags bits identify the typed complete correction, its
 * success, D4 success, gCP success, cuda+cpu backend and present gradient.
 * Strings must be NUL-terminated within their inline arrays. Missing gradient
 * qualification is distinct from an explicitly unknown name. Counts describe
 * actual attempted work; missing mixed Fock count is explicitly marked.
 */
typedef struct generativeqc_precision_candidate_context_v1 {
  uint32_t struct_size, abi_version;
  uint32_t budget_requested, explicit_target, budget_native, budget_fock, forces_requested;
  uint32_t gradient_present, gradient_schema_version, gradient_complete;
  uint32_t gradient_owner, gradient_returned_mode, qualification_present;
  uint32_t gradient_capability_fallback, weighted_task_counters_collected;
  uint32_t correction_present, correction_result_flags, mixed_fock_count_present, reserved;
  uint64_t geometry_points, grid_points, weighted_tasks, primitive_sources;
  uint64_t gradient_retries, execution_retries;
  char qualification[96], correction_model_identity[65];
} generativeqc_precision_candidate_context_v1;

/** One native route and ordered presentation reasons, never an error estimate. */
typedef struct generativeqc_precision_candidate_v1 {
  uint32_t struct_size, abi_version, policy_version, disposition;
  uint32_t qualified_gradient, reasons;
  char gradient_admission[96];
} generativeqc_precision_candidate_v1;

/** @native-contract generativeqc_precision_candidate_query_v1
 * @behavior Classify completed precision candidates using shared native work
 * predicates, incumbent gradient receipt domains and complete correction binding.
 * Explicit observable targets always require an independent audit. No new
 * method, derivative, provider or FP32 domain is qualified by this query.
 * @inputs Nonnull versioned candidate/admission contexts, provenance, detail
 * and output. Event/operator arrays contain the counts declared by detail.
 * Candidate facts override the admission context's gradient interpretation.
 * Correction identity is compared to the selected library's canonical native
 * r2SCAN-3c catalog entry; display names are not model identity.
 * @outputs Publish policy version 1, one disposition, reason bits and complete
 * gradient admission name. AUDIT requires the existing independent observable
 * endpoint; it does not mean a numerical failure or an unsupported protocol.
 * @errors INVALID_ARGUMENT for malformed facts, flags, strings or work records;
 * ABI_MISMATCH for incompatible descriptors; NOT_IMPLEMENTED for unsupported
 * work-detail versions. Failure leaves output unchanged.
 * @lifetime Borrow all inputs synchronously, retain nothing. Complete reads
 * before publishing output, permitting input/output overlap.
 * @execution CPU metadata only, allocation-free and reentrant; no electronic
 * solve, compiler, Python interpreter, resource allocation or fallback occurs.
 * @units Counts are semantic diagnostic work, not timings or inferred zeros.
 */
GENERATIVEQC_API generativeqc_status generativeqc_precision_candidate_query_v1(
    const generativeqc_precision_candidate_context_v1* candidate,
    const generativeqc_precision_admission_context_v1* context,
    const generativeqc_precision_provenance* provenance,
    const generativeqc_precision_work_detail* detail,
    const generativeqc_precision_work_event* events,
    const generativeqc_precision_operator_record* operators,
    generativeqc_precision_candidate_v1* output);

/** Inputs for the incumbent budget-Fock SCF control policy. Boolean fields are
 * 0/1. modules_available is evidence from the selected Direct capability or the
 * fitted model's qualification owner; this query does not establish it. An
 * override_mask bit selects energy, density or screening in that order. Unset
 * controls are zero. Zero force_rms means no RMS target was requested. */
typedef struct generativeqc_precision_controls_context_v1 {
  uint32_t struct_size, abi_version;
  uint32_t modules_available, ks_model, restricted_spin, self_consistent, device_fused;
  uint32_t override_mask, reserved;
  double energy_abs, force_max_abs, force_rms, requested_controls[3];
} generativeqc_precision_controls_context_v1;

/** selected=0 preserves the caller's ordinary strict controls; all returned
 * controls are then zero. selected=1 supplies bounded energy/density/screening
 * tolerances, each no looser than its explicitly supplied caller override. */
typedef struct generativeqc_precision_controls_v1 {
  uint32_t struct_size, abi_version, policy_version, selected;
  double controls[3];
} generativeqc_precision_controls_v1;

/** @native-contract generativeqc_precision_controls_query_v1
 * @behavior Select the shared budget-Fock SCF convergence controls.
 * @inputs Versioned model/capability facts, positive energy and force allowances,
 * optional nonnegative RMS allowance, and explicitly selected positive overrides.
 * The supplied qualification belongs to the same selected native library/model.
 * @outputs Transactionally writes policy version 1, selection and three controls.
 * Unsupported traits or tighter budgets select no override, preserving strict SCF.
 * @errors INVALID_ARGUMENT for null inputs, invalid flags/mask, nonzero reserved,
 * nonfinite/invalid allowances or overrides; ABI_MISMATCH for descriptor headers.
 * Output is unchanged on failure. Missing qualification is a successful nonselection.
 * @lifetime Storage is borrowed only for this call; no pointer is retained.
 * @execution Allocation-free synchronous host policy; no device, compiler or solve.
 * This neither admits an arithmetic provider nor supplies an observable-error proof.
 * @units Energy allowance is Eh; force allowances are Eh/bohr; controls use existing
 * native SCF units. Explicit tighter caller controls remain unchanged.
 */
GENERATIVEQC_API generativeqc_status
generativeqc_precision_controls_query_v1(const generativeqc_precision_controls_context_v1* context,
                                         generativeqc_precision_controls_v1* output);

/** Execution events are operations, not inferred scientific work counts. */
enum {
  GENERATIVEQC_PRECISION_EXECUTE_CANDIDATE = 0,
  GENERATIVEQC_PRECISION_EXECUTE_FP64_FALLBACK = 1,
  GENERATIVEQC_PRECISION_CLASSIFY_CANDIDATE = 2,
  GENERATIVEQC_PRECISION_RELEASE_CANDIDATE = 3,
  GENERATIVEQC_PRECISION_EXECUTE_AUDIT = 4,
  GENERATIVEQC_PRECISION_VERIFY_FALLBACK = 5
};

enum {
  GENERATIVEQC_PRECISION_CALLBACK_SUCCESS = 0,
  GENERATIVEQC_PRECISION_CALLBACK_RETRY = 1,
  GENERATIVEQC_PRECISION_CALLBACK_ERROR = 2
};

enum {
  GENERATIVEQC_PRECISION_EXECUTION_QUALIFIED = 0,
  GENERATIVEQC_PRECISION_EXECUTION_AUDITED = 1,
  GENERATIVEQC_PRECISION_EXECUTION_VERIFIED = 2,
  GENERATIVEQC_PRECISION_EXECUTION_REJECTED = 3,
  GENERATIVEQC_PRECISION_EXECUTION_FAILED = 4
};

/** retry_candidate is 0/1. Only an explicitly retryable candidate callback
 * failure may select one FP64 replacement. All reserved fields are zero. */
typedef struct generativeqc_precision_execution_context_v1 {
  uint32_t struct_size, abi_version, retry_candidate, reserved;
} generativeqc_precision_execution_context_v1;

/** Versioned execution receipt. Attempted and completed event masks distinguish
 * a failed candidate from its replacement. REJECTED means the independently
 * verified replacement still misses its target; it cannot publish a Result. */
typedef struct generativeqc_precision_execution_receipt_v1 {
  uint32_t struct_size, abi_version, policy_version;
  uint32_t attempted_events, completed_events, outcome;
} generativeqc_precision_execution_receipt_v1;

/** @native-contract generativeqc_precision_execution_callback_v1
 * @behavior Execute one synchronous operation requested by the native schedule.
 * @inputs state is the caller's opaque owner. event is one of the six precision
 * execution events; the native owner supplies nonnull decision storage.
 * @outputs Return CALLBACK_SUCCESS and set decision=0 for execution/release;
 * CLASSIFY returns 1 for an admitted no-audit publication or 0 to require an
 * independent audit. AUDIT/VERIFY return 1 only when every declared observable
 * gate meets its allowance. Return CALLBACK_RETRY only for a candidate failure
 * eligible under the caller's unchanged numerical retry policy; other failures
 * return CALLBACK_ERROR. No exception may cross this callback boundary.
 * @errors Callback failures stop the schedule, except the single explicitly
 * admitted candidate retry. Neither audit nor verification errors are retried.
 * @lifetime The callback owns preparation, identity validation and cleanup,
 * including failed preparation/solve. It releases transient auditors before
 * returning and candidate residency before allocating an independent auditor.
 * Retained candidate fleets, detached results and independent seed histories
 * stay owned by state; native code retains no resource or pointer after return.
 * @execution Invoked synchronously on the calling thread. Numerical kernels,
 * model/qualification admission, observable reduction and resource leases keep
 * their existing owners. Native stage order is not an observable certificate.
 * @units decision is boolean; masks/events do not denote iterations or work.
 */
typedef int (*generativeqc_precision_execution_callback_v1)(void* state, uint32_t event,
                                                            uint32_t* decision);

/** @native-contract generativeqc_precision_execution_run_v1
 * @behavior Own bounded application-precision candidate/auditor/fallback order.
 * Run candidate, at most one admitted FP64 retry, then classification. An audit
 * request releases candidate residency before one independent FP64 audit. Only
 * an unmet audit selects one tighter verification of the FP64 replacement.
 * @inputs Nonnull versioned context, callback and receipt; state may be null.
 * The callback must preserve the same scientific model and independent histories
 * and must not present numerical/qualification failure as unsupported protocol.
 * @outputs Publish policy version 1, attempted/completed event masks and final
 * outcome. Callback failure publishes a FAILED progress receipt, not a partial
 * observable. REJECTED is a completed schedule, not a successful budget gate.
 * Malformed descriptor errors leave receipt unchanged and invoke no callback.
 * @errors INVALID_ARGUMENT for null pointers, invalid flags/reserved fields or
 * invalid callback decisions; ABI_MISMATCH for incompatible descriptor headers;
 * INTERNAL_ERROR for callback errors. No C++ exception escapes this boundary.
 * @lifetime Borrow all storage synchronously; retain no callback, state, result
 * or endpoint owner. Context and receipt may overlap; inputs are copied first.
 * @execution Constant-storage host control; electronic work occurs only through
 * explicit callbacks. No device, context, library discovery, compiler, implicit
 * Python fallback or retry after native protocol refusal is performed here.
 * @units Event masks and outcomes are dimensionless; numerical allowances and
 * measured observable errors retain their separately declared physical units.
 */
GENERATIVEQC_API generativeqc_status generativeqc_precision_execution_run_v1(
    const generativeqc_precision_execution_context_v1* context,
    generativeqc_precision_execution_callback_v1 callback, void* state,
    generativeqc_precision_execution_receipt_v1* output);

/** Ordered slots/bits for energy absolute, force maximum and force RMS norms. */
enum {
  GENERATIVEQC_OBSERVABLE_ENERGY_ABSOLUTE = 1,
  GENERATIVEQC_OBSERVABLE_FORCE_MAXIMUM = 2,
  GENERATIVEQC_OBSERVABLE_FORCE_RMS = 4
};

/** One to three unique requested norms. Allowances are absolute + relative *
 * reference_norm, with nonnegative finite coefficients and at least one positive
 * coefficient per selected slot. Unselected coefficients/energies are zero.
 * Forces, when requested, contain exactly atom_count*3 Cartesian components. */
typedef struct generativeqc_observable_comparison_context_v1 {
  uint32_t struct_size, abi_version, requested_mask, reserved;
  uint64_t atom_count;
  double energy, reference_energy, absolute[3], relative[3];
} generativeqc_observable_comparison_context_v1;

/** Unselected slots are zero. met_mask is a subset of measured_mask, with no
 * implicit tolerance or slack added to the inclusive difference <= allowance. */
typedef struct generativeqc_observable_comparison_v1 {
  uint32_t struct_size, abi_version, policy_version, measured_mask, met_mask, reserved;
  double differences[3], reference_norms[3], allowances[3];
} generativeqc_observable_comparison_v1;

/** @native-contract generativeqc_observable_comparison_query_v1
 * @behavior Measure completed energy/force discrepancies and resolve allowances.
 * Energy uses absolute scalar differences; forces use maximum absolute component
 * or scaled RMS over all 3N components. RMS uses compensated normalized squares
 * to avoid overflow and unbounded summation error, not an unscaled sum of squares.
 * @inputs Nonnull versioned context/output. A nonzero requested_mask selects the
 * three documented slots. Selected force norms require nonnull force/reference
 * arrays and positive atom_count; otherwise both pointers and atom_count are zero.
 * Selected observables and all coefficients are finite FP64. Values must belong
 * to the same model and declared scope; convergence/identity admission belongs to
 * their endpoint owner, not this comparison. Arrays may not change during the call.
 * @outputs Transactionally publish version 1, measured/met masks, differences,
 * reference norms and absolute-plus-relative allowances. Coefficient multiplication
 * is rounded before addition. Every failure leaves output unchanged.
 * @errors INVALID_ARGUMENT for null/mismatched arrays, invalid masks/reserved
 * fields, invalid/overflowing dimensions, nonfinite values, overflowing differences
 * or allowances, or invalid coefficients; ABI_MISMATCH for descriptor headers.
 * @lifetime Borrow inputs/output only during the call. Copy the descriptor and
 * complete all reads before publication, allowing input/output storage overlap.
 * @execution Synchronous allocation-free host comparison, at most two passes
 * over each force array. No context, compiler, device, solve, discovery or implicit
 * oracle/fallback work. A met receipt is observed agreement, not a certificate
 * of reference uncertainty, model accuracy, derivative qualification or convergence.
 * @units Energy is Eh; forces are Eh/bohr; relative coefficients are dimensionless.
 */
GENERATIVEQC_API generativeqc_status generativeqc_observable_comparison_query_v1(
    const generativeqc_observable_comparison_context_v1* context, const double* forces,
    const double* reference_forces, generativeqc_observable_comparison_v1* output);

#ifdef __cplusplus
}
#endif

#endif
