#ifndef GENERATIVEQC_FOCK_H
#define GENERATIVEQC_FOCK_H

#include "generativeqc/generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct generativeqc_fock_plan generativeqc_fock_plan;
typedef int32_t generativeqc_fock_spin;
enum { GENERATIVEQC_FOCK_RESTRICTED = 0, GENERATIVEQC_FOCK_UNRESTRICTED = 1 };
typedef int32_t generativeqc_fock_operator;
enum {
  GENERATIVEQC_FOCK_FULL_RANGE = 0,
  GENERATIVEQC_FOCK_SHORT_RANGE = 1,
  GENERATIVEQC_FOCK_LONG_RANGE = 2
};
typedef int32_t generativeqc_fock_approximation;
enum { GENERATIVEQC_FOCK_EXACT = 0, GENERATIVEQC_FOCK_DENSITY_FITTED = 1 };

/** Raw J/K coefficients are applied once by native Fock/energy assembly.
 * present is 0 or 1. A present term with coefficient zero still returns its
 * raw matrix. Reserved SR/LR operators currently fail explicitly. */
typedef struct generativeqc_fock_term {
  int32_t present;
  double coefficient;
  generativeqc_fock_operator op;
  double omega;
  generativeqc_fock_approximation approximation;
} generativeqc_fock_term;

/** Mathematical identity, independent of backend and memory placement.
 * Restricted D includes double occupation; unrestricted Da/Db have unit
 * occupation. Standard HF uses cJ=1, cK=-0.5 (restricted) or -1 (unrestricted).
 */
typedef struct generativeqc_fock_spec {
  uint32_t struct_size, abi_version, spec_version;
  generativeqc_fock_spin spin;
  uint32_t derivative_order;
  generativeqc_fock_term coulomb, exchange;
} generativeqc_fock_spec;

/** NULL controls use screening=1e-12, metric cutoff=1e-10 and the default
 * buffer allowance. An explicit screening zero means unscreened execution.
 * Screening must be finite and nonnegative. Metric cutoffs must be finite
 * and in [0, 1), including when no fitted term is requested.
 * A zero metric cutoff selects 1e-10; zero device bytes selects 256 MiB for
 * the independent CUDA source. These controls never authorize a change from
 * exact to fitted mathematics. */
typedef struct generativeqc_fock_controls {
  uint32_t struct_size, abi_version;
  double screening_tolerance, metric_relative_threshold;
  uint64_t device_budget_bytes;
} generativeqc_fock_controls;

/** Construct owned normalized source snapshots. system/auxiliary/context may
 * be destroyed after success. A null auxiliary uses the orbital basis for
 * fitted terms; unused auxiliary data does not affect exact-only identity.
 * Failure leaves *output NULL. Independent CUDA execution uses host SCF
 * control and CUDA integral consumers; ordinary HF APIs retain fused solvers.
 * A handle and its error/diagnostic state are not concurrently reentrant;
 * serialize every call on a handle and its destruction.
 */
GENERATIVEQC_API generativeqc_status generativeqc_fock_plan_create(
    generativeqc_context* context, const generativeqc_system* system,
    const generativeqc_system* auxiliary, const generativeqc_fock_spec* spec,
    const generativeqc_fock_controls* controls, generativeqc_fock_plan** output);
GENERATIVEQC_API void generativeqc_fock_plan_destroy(generativeqc_fock_plan* plan);
/** Valid until the next call on this plan; null handles return a fixed message. */
GENERATIVEQC_API const char* generativeqc_fock_plan_last_error(const generativeqc_fock_plan* plan);

/** Caller-owned optional outputs. All matrix buffers use matrix_count=nbf^2
 * and row-major public AO order. Requested output buffers must be disjoint.
 * Absent raw terms leave their buffers untouched. Fock matrices include Hcore.
 * gradient is the fixed-density TWO-ELECTRON energy derivative only, with
 * gradient_count=3*Natom; it excludes one-electron, nuclear and Pulay terms.
 * Unrequested gradients use NULL and count zero. All outputs change only on
 * success. This is a fixed-density operation, with no density normalization.
 */
typedef struct generativeqc_fock_result {
  uint32_t struct_size, abi_version;
  uint64_t matrix_count, gradient_count;
  double* coulomb;
  double* exchange_alpha;
  double* exchange_beta;
  double* fock_alpha;
  double* fock_beta;
  double* gradient;
  double energy_one_electron, energy_two_electron, nuclear_repulsion;
} generativeqc_fock_result;

GENERATIVEQC_API generativeqc_status generativeqc_fock_plan_evaluate(
    generativeqc_fock_plan* plan, const double* density, uint64_t density_count, const double* beta,
    uint64_t beta_count, generativeqc_fock_result* result);

/** Convergence controls only; NULL selects 100 iterations, DIIS history 8,
 * energy tolerance 1e-10 Hartree and density RMS tolerance 1e-8. Explicit
 * counts must be positive and tolerances finite and positive. */
typedef struct generativeqc_fock_scf_controls {
  uint32_t struct_size, abi_version;
  uint32_t max_iterations, diis_history;
  double energy_tolerance, density_tolerance;
} generativeqc_fock_scf_controls;

/** Optional caller-owned SCF outputs in public AO/atom order. Restricted
 * density_count is nbf^2; unrestricted is 2*nbf^2, alpha followed by beta.
 * NULL density with count zero omits its copy. NULL forces with count zero
 * requests energy-only execution; otherwise force_count must be 3*Natom and
 * the plan must support first derivatives. Forces are COMPLETE negative
 * energy derivatives, including one-electron, nuclear and overlap/Pulay terms.
 * Buffers and this descriptor must be disjoint. All outputs remain unchanged
 * on failure, including NOT_CONVERGED. The plan retains sources, not iterates.
 */
typedef struct generativeqc_fock_scf_result {
  uint32_t struct_size, abi_version;
  double* density;
  double* forces;
  uint64_t density_count, force_count;
  double energy, energy_change, density_rms;
  uint32_t iterations;
  int32_t initial_density_used;
  uint64_t fock_builds;
} generativeqc_fock_scf_result;

/** Solve the declared native J/K energy using the shared host SCF driver.
 * No XC is added. CUDA plans use CUDA integrals/J/K/response and host
 * DIIS/eigensolves. Optional initial_density uses the output density layout;
 * it must satisfy the existing Hermiticity, overlap-metric occupation and
 * electron/spin trace guards. NULL/zero requests the core-Hamiltonian guess.
 * Input density is copied before execution and may alias an output buffer.
 * Handles are not concurrently reentrant; serialize calls and destruction.
 */
GENERATIVEQC_API generativeqc_status generativeqc_fock_plan_solve(
    generativeqc_fock_plan* plan, const generativeqc_fock_scf_controls* controls,
    const double* initial_density, uint64_t initial_density_count,
    generativeqc_fock_scf_result* result);

/** Stable v1 schedule identifiers for prepared source execution. */
enum {
  GENERATIVEQC_FOCK_CPU_REFERENCE = 0,
  GENERATIVEQC_FOCK_CUDA_FUSED = 1,
  GENERATIVEQC_FOCK_STANDARD_DF = 2,
  GENERATIVEQC_FOCK_CPU_INDEPENDENT = 3,
  GENERATIVEQC_FOCK_CUDA_INDEPENDENT = 4
};

/** Requested and canonical semantics plus actual prepared source diagnostics.
 * The public independent plan always uses the common provider boundary even
 * when semantic resolution identifies a standard-HF preferred schedule.
 * source_schedule records the actual plan route and resolved_schedule records
 * that preference. CUDA module/driver/library-private storage is excluded.
 */
typedef struct generativeqc_fock_diagnostic {
  uint32_t struct_size, abi_version;
  generativeqc_fock_spec requested, resolved;
  generativeqc_backend backend;
  int32_t resolved_schedule, source_schedule;
  uint64_t nbf, coordinate_count, device_bytes, device_budget_bytes;
  uint64_t auxiliary_rank, auxiliary_tile;
  double screening_tolerance, metric_relative_threshold;
  int32_t df_streamed;
  char direct_schedule[96], df_value_backend[48], df_value_mapping[32], df_response_mapping[32];
  char one_electron_value_backend[32], one_electron_value_mapping[32];
  char one_electron_response_mapping[32];
} generativeqc_fock_diagnostic;
GENERATIVEQC_API generativeqc_status generativeqc_fock_plan_diagnostic(
    const generativeqc_fock_plan* plan, generativeqc_fock_diagnostic* diagnostic);

/** Tools-only resident RHF response/Krylov vector owner.
 *
 * The owner borrows an exact CUDA Fock plan and therefore must be destroyed
 * before its parent generativeqc_fock_plan. Coefficients are copied during creation.
 * Vector slots remain on the parent's CUDA device. Small scalar dot/norm/status
 * values are the only success-path Krylov transfers; final vector download is
 * explicit. This ABI is additive and is not a public Calculator capability.
 */
typedef struct generativeqc_rhf_response_resident generativeqc_rhf_response_resident;

typedef struct generativeqc_rhf_response_resident_diagnostic {
  uint32_t struct_size, abi_version;
  uint64_t nbf, nocc, nvirt, dimension, vector_slots;
  uint64_t owned_device_bytes;
  uint64_t h2d_bytes, d2h_bytes, synchronizations;
  uint64_t operator_actions, blas_calls;
  int32_t device_id;
} generativeqc_rhf_response_resident_diagnostic;

GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_create(
    generativeqc_fock_plan* plan, const double* coefficients, uint64_t coefficient_count,
    const double* orbital_energies, uint64_t energy_count, uint32_t nocc, uint32_t vector_slots,
    uint64_t device_budget_bytes, generativeqc_rhf_response_resident** output);
GENERATIVEQC_API void generativeqc_rhf_response_resident_destroy(
    generativeqc_rhf_response_resident* owner);
GENERATIVEQC_API const char* generativeqc_rhf_response_resident_last_error(
    const generativeqc_rhf_response_resident* owner);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_get_diagnostic(
    const generativeqc_rhf_response_resident* owner,
    generativeqc_rhf_response_resident_diagnostic* diagnostic);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_upload(
    generativeqc_rhf_response_resident* owner, uint32_t slot, const double* values, uint64_t count);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_download(
    generativeqc_rhf_response_resident* owner, uint32_t slot, double* values, uint64_t count);
GENERATIVEQC_API generativeqc_status
generativeqc_rhf_response_resident_zero(generativeqc_rhf_response_resident* owner, uint32_t slot);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_copy(
    generativeqc_rhf_response_resident* owner, uint32_t destination, uint32_t source);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_scale(
    generativeqc_rhf_response_resident* owner, uint32_t slot, double alpha);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_axpy(
    generativeqc_rhf_response_resident* owner, uint32_t destination, double alpha, uint32_t source);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_dot(
    generativeqc_rhf_response_resident* owner, uint32_t left, uint32_t right, double* value);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_norm(
    generativeqc_rhf_response_resident* owner, uint32_t slot, double* value);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_apply(
    generativeqc_rhf_response_resident* owner, uint32_t destination, uint32_t source);
/** Reconstruct final RHF nuclear response matrices from one solved resident
 * rotation vector. frozen_mo/overlap_mo are row-major (nbf,nbf) host matrices.
 * On success D1 and W1 remain device-resident and contiguous until the next
 * resident operator action or reconstruction. This is tools-only plumbing for
 * Hessian/HVP consumers, not a public Calculator API. */
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_reconstruct_v1(
    generativeqc_rhf_response_resident* owner, uint32_t solution_slot, const double* frozen_mo,
    uint64_t frozen_count, const double* overlap_mo, uint64_t overlap_count);
GENERATIVEQC_API const double* generativeqc_rhf_response_resident_reconstructed_weights_device_v1(
    const generativeqc_rhf_response_resident* owner);
GENERATIVEQC_API generativeqc_status generativeqc_rhf_response_resident_download_reconstruction_v1(
    generativeqc_rhf_response_resident* owner, double* density_derivative, uint64_t density_count,
    double* energy_weighted_density_derivative, uint64_t energy_count);

/** Tools-only resident unrestricted-HF response/Krylov vector owner.
 *
 * The owner borrows an exact unscreened unrestricted CUDA Fock plan. Alpha and
 * beta occupied/virtual rotations share one device-resident slot arena; raw
 * spin J/K actions and the coupled UHF Jacobian are enqueued on the plan's
 * stream without a host response-vector round trip. This additive ABI is
 * intentionally tools-only and does not advertise a public Calculator API.
 */
typedef struct generativeqc_uhf_response_resident generativeqc_uhf_response_resident;

typedef struct generativeqc_uhf_response_resident_diagnostic {
  uint32_t struct_size, abi_version;
  uint64_t nbf, nocc_alpha, nvirt_alpha, nocc_beta, nvirt_beta;
  uint64_t dimension, vector_slots, owned_device_bytes;
  uint64_t h2d_bytes, d2h_bytes, synchronizations;
  uint64_t operator_actions, blas_calls;
  int32_t device_id;
} generativeqc_uhf_response_resident_diagnostic;

GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_create(
    generativeqc_fock_plan* plan, const double* coefficients_alpha,
    uint64_t coefficients_alpha_count, const double* orbital_energies_alpha,
    uint64_t orbital_energies_alpha_count, uint32_t nocc_alpha, const double* coefficients_beta,
    uint64_t coefficients_beta_count, const double* orbital_energies_beta,
    uint64_t orbital_energies_beta_count, uint32_t nocc_beta, uint32_t vector_slots,
    uint64_t device_budget_bytes, generativeqc_uhf_response_resident** output);
GENERATIVEQC_API void generativeqc_uhf_response_resident_destroy(
    generativeqc_uhf_response_resident* owner);
GENERATIVEQC_API const char* generativeqc_uhf_response_resident_last_error(
    const generativeqc_uhf_response_resident* owner);
GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_get_diagnostic(
    const generativeqc_uhf_response_resident* owner,
    generativeqc_uhf_response_resident_diagnostic* diagnostic);
GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_upload(
    generativeqc_uhf_response_resident* owner, uint32_t slot, const double* values, uint64_t count);
GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_download(
    generativeqc_uhf_response_resident* owner, uint32_t slot, double* values, uint64_t count);
GENERATIVEQC_API generativeqc_status
generativeqc_uhf_response_resident_zero(generativeqc_uhf_response_resident* owner, uint32_t slot);
GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_copy(
    generativeqc_uhf_response_resident* owner, uint32_t destination, uint32_t source);
GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_scale(
    generativeqc_uhf_response_resident* owner, uint32_t slot, double alpha);
GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_axpy(
    generativeqc_uhf_response_resident* owner, uint32_t destination, double alpha, uint32_t source);
GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_dot(
    generativeqc_uhf_response_resident* owner, uint32_t left, uint32_t right, double* value);
GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_norm(
    generativeqc_uhf_response_resident* owner, uint32_t slot, double* value);
GENERATIVEQC_API generativeqc_status generativeqc_uhf_response_resident_apply(
    generativeqc_uhf_response_resident* owner, uint32_t destination, uint32_t source);

#ifdef __cplusplus
}
#endif
#endif
