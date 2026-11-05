#ifndef GENERATIVEQC_SCF_DEFAULTS_H
#define GENERATIVEQC_SCF_DEFAULTS_H

#include "generativeqc/generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Validated calculation facts, separate from arithmetic and method admission.
 * has_ecp is boolean. SCF defaults do not depend on backend, spin, ECP, model or grid;
 * supplying these facts cannot grant a physical-final-state capability. */
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
 * lowercase 64-byte MethodIR identity. Optional KS grid data is validated but does
 * not select tolerances. A null KS descriptor denotes a non-KS model.
 * @outputs Transactionally writes policy version 2, energy 1e-9, density 1e-5 and
 * screening 1e-12 for SCF. SCC retains energy 1e-10/density 1e-8; qualified_defaults
 * is zero for SCC and one otherwise. Explicit user controls and stricter method-owned
 * reference/final-state gates remain authoritative.
 * @errors INVALID_ARGUMENT for invalid backend/flags, malformed identity or grid data;
 * ABI_MISMATCH for invalid descriptor headers. Output is unchanged on failure.
 * @lifetime All pointers are borrowed for this call only; no pointer is retained.
 * @execution Synchronous allocation-free host metadata query; no provider, GPU,
 * quadrature, compiler or scientific solve. Unknown model identities use the same defaults.
 * @units Energy tolerance is Eh, matching GPU4PySCF's default conv_tol. Density and
 * screening retain native SCF units; density is not GPU4PySCF's MO-gradient norm.
 */
GENERATIVEQC_API generativeqc_status generativeqc_scf_defaults_query_v1(
    const generativeqc_scf_defaults_context_v1* context, const generativeqc_ks_options* ks_options,
    const char* model_identity, size_t model_identity_size, generativeqc_scf_defaults_v1* output);

#ifdef __cplusplus
}
#endif

#endif
