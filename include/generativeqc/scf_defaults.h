#ifndef GENERATIVEQC_SCF_DEFAULTS_H
#define GENERATIVEQC_SCF_DEFAULTS_H

#include "generativeqc/generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Model/backend context for default solver controls, separate from arithmetic.
 * has_ecp is boolean. MethodIR identity identifies the complete electronic model;
 * a familiar name or ingredient cannot qualify a custom composition. */
typedef struct generativeqc_scf_defaults_context_v1 {
  uint32_t struct_size, abi_version;
  generativeqc_backend backend;
  generativeqc_method method;
  uint32_t has_ecp, reserved;
} generativeqc_scf_defaults_context_v1;

/** Default controls only: callers preserve every explicit user override. These
 * tolerances do not replace the native owner's physical final-state checks. */
typedef struct generativeqc_scf_defaults_v1 {
  uint32_t struct_size, abi_version, policy_version, qualified_defaults;
  double energy_tolerance, density_tolerance, screening_tolerance;
} generativeqc_scf_defaults_v1;

/** @native-contract generativeqc_scf_defaults_query_v1
 * @behavior Query shared versioned default SCF convergence controls.
 * @inputs Nonnull versioned context and output; optional KS descriptor and complete
 * lowercase 64-byte MethodIR identity. A missing identity cannot qualify KS defaults.
 * KS grid fields, dense element radii and XC schedule are inspected; other descriptor
 * fields are outside this metadata query. A null KS descriptor denotes a non-KS model.
 * @outputs Transactionally writes policy version 1, three tolerances, and whether the
 * qualified CUDA default policy applied. CPU/ECP/other models retain strict defaults.
 * @errors INVALID_ARGUMENT for invalid backend/flags, malformed identity or grid data;
 * ABI_MISMATCH for invalid descriptor headers. Output is unchanged on failure.
 * @lifetime All pointers are borrowed for this call only; no pointer is retained.
 * @execution Synchronous allocation-free host metadata query; no provider, GPU,
 * quadrature, compiler or scientific solve. Unknown model identities keep strict defaults.
 * @units Energy tolerance is Eh; density and screening retain their native SCF units.
 */
GENERATIVEQC_API generativeqc_status generativeqc_scf_defaults_query_v1(
    const generativeqc_scf_defaults_context_v1* context, const generativeqc_ks_options* ks_options,
    const char* model_identity, size_t model_identity_size, generativeqc_scf_defaults_v1* output);

#ifdef __cplusplus
}
#endif

#endif
