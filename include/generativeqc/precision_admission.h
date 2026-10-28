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

#ifdef __cplusplus
}
#endif

#endif
