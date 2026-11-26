#ifndef GENERATIVEQC_KS_HESSIAN_H
#define GENERATIVEQC_KS_HESSIAN_H

#include "generativeqc/generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Independent response limits; SCF allocations remain charged to the batch. */
typedef struct generativeqc_ks_hessian_options_v2 {
  size_t struct_size;
  uint32_t abi_version;
  size_t device_bytes, integral_bytes, grid_bytes, response_bytes, output_bytes;
  uint64_t max_nonlocal_pairs;
  size_t restart, max_iterations;
  unsigned int reorthogonalizations;
  double relative_tolerance, absolute_tolerance;
  /** Dense only: 0 uses FP64; 1 uses FP32 proposals with FP64 physical correction. */
  uint32_t response_fp32_proposal;
} generativeqc_ks_hessian_options_v2;

/** Detached work receipt; residual fields refer to the last completed direction. */
typedef struct generativeqc_ks_hessian_receipt_v2 {
  size_t struct_size;
  uint32_t abi_version;
  uint64_t device_bytes, completed_directions, induced_fock_actions;
  uint64_t exact_primitive_components, one_electron_primitive_components;
  /** Counts across the admitted fitted radial sources; exact LR is not included. */
  uint64_t df_value_passes, df_contraction_passes;
  uint64_t grid_right_directions, grid_mixed_contractions, vv10_reference_pairs;
  uint64_t response_iterations, response_workspace_bytes;
  uint64_t ecp_workspace_bytes, df_workspace_bytes;
  double response_rhs_norm, response_residual_norm, response_tolerance;
  double execution_seconds;
  uint64_t cpks_operator_bytes, cpks_physical_checks, cpks_cached_actions;
  uint64_t cpks_proposal_iterations, cpks_proposal_seeds, cpks_strict_iterations;
} generativeqc_ks_hessian_receipt_v2;

/** @native-contract generativeqc_ks_hessian_cuda_version_v2
 * @behavior Discover the linked CUDA Hessian ABI, independent of device initialization.
 * @outputs Returns 2 if linked, 0 otherwise. Not a per-method or per-device capability.
 * @execution Allocation-free host metadata query; no context needed.
 */
GENERATIVEQC_API uint32_t generativeqc_ks_hessian_cuda_version_v2(void);

/** @native-contract generativeqc_ks_hessian_default_options_v2
 * @behavior Initialize current bounded CUDA Hessian/HVP options, including size and ABI version.
 * @inputs Pointer to writable options storage.
 * @outputs Overwrites the descriptor. Fields can be changed before execution.
 * @errors A null pointer returns INVALID_ARGUMENT.
 * @execution Allocation-free host metadata query; works without the CUDA provider.
 * @units Capacities in bytes; response tolerances use the native FP64 residual convention.
 */
GENERATIVEQC_API generativeqc_status
generativeqc_ks_hessian_default_options_v2(generativeqc_ks_hessian_options_v2* options);

/** @native-contract generativeqc_ks_hessian_cuda_v2
 * @behavior Differentiate an accepted physical CUDA KS state for one prepared batch item.
 * @inputs Live batch, original input index, initialized options and receipt (set struct_size
 * and abi_version). For N atoms, supply NULL/0 direction and exactly (3N)^2 outputs, or a
 * finite 3N-entry direction and exactly 3N outputs. The four host records/panels are disjoint.
 * @outputs Raw row-major Hessian or atom-major Cartesian HVP, plus a detached work receipt.
 * @lifetime No pointers are retained. Serialize against other batch/context operations.
 * @errors Malformed requests, unqualified/stale states, failed convergence and resource
 * exhaustion fail without changing either output. Missing linked providers never fall back.
 * @execution Synchronous FP64 CUDA response on the accepted SCF provider. Checks the batch's
 * shared device ledger if present; downloads only the terminal derivative, not SCF panels.
 * @units Coordinates/direction in Bohr coordinate conventions; Hessian in Hartree/Bohr^2;
 * HVP is the linear action in the supplied direction. Capacities in bytes.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_hessian_cuda_v2(
    generativeqc_batch* batch, size_t index, const generativeqc_ks_hessian_options_v2* options,
    const double* direction, size_t direction_count, double* output, size_t output_count,
    generativeqc_ks_hessian_receipt_v2* receipt);

/** CPU derivative limits. The expected system identifies the accepted geometry,
 * including any coordinate update, not necessarily the original preparation.
 * The numerical reservation excludes the borrowed primal SCF/Fock/basis/grid,
 * its final-state export scratch, BLAS scratch and process RSS. Work ceilings
 * apply to the whole request, not to each column independently.
 */
typedef struct generativeqc_ks_host_hessian_options_v1 {
  size_t struct_size;
  uint32_t abi_version;
  const generativeqc_atom* expected_atoms;
  size_t expected_atom_count;
  int32_t expected_charge, expected_multiplicity;
  size_t numeric_bytes, integral_bytes, grid_bytes;
  size_t response_bytes, ecp_bytes, dispersion_bytes, output_bytes;
  size_t max_eri_primitives, max_map_terms, max_grid_visits, max_d3_pairs;
  size_t restart, max_iterations, gmres_bytes, true_residual_every, stagnation_window;
  unsigned int reorthogonalizations;
  double relative_tolerance, absolute_tolerance;
  double breakdown_tolerance, stagnation_tolerance;
  double symmetry_absolute_tolerance, symmetry_relative_tolerance;
} generativeqc_ks_host_hessian_options_v1;

/** Detached CPU receipt. Work counters sum over completed directions and the
 * residual fields report maxima. Reservations are not measured peak RSS.
 */
typedef struct generativeqc_ks_host_hessian_receipt_v1 {
  size_t struct_size;
  uint32_t abi_version;
  uint64_t numeric_reservation_bytes, frame_bytes, geometry_bytes, output_bytes;
  uint64_t fixed_bound_bytes, gmres_workspace_bytes, completed_directions;
  uint64_t response_iterations, induced_fock_actions;
  uint64_t first_eri_primitives, curvature_eri_primitives;
  uint64_t first_one_electron_primitives, curvature_one_electron_primitives;
  uint64_t source_xc_point_visits, curvature_xc_point_visits, response_xc_point_visits;
  uint64_t ecp_kernel_blocks, ecp_radial_layers, ecp_ao_primitive_points;
  uint64_t ecp_contraction_terms, d3_pair_visits, d3_reference_terms;
  double response_residual_norm, response_relative_residual;
  double electronic_energy, total_energy, d3_energy, source_energy_error;
  double raw_symmetry_error, execution_seconds;
} generativeqc_ks_host_hessian_receipt_v1;

/** @native-contract generativeqc_ks_hessian_cpu_version_v1
 * @behavior Discover the linked CPU Hessian ABI without initializing a context.
 * @outputs Returns 1 if linked, 0 otherwise; not a per-system capability.
 * @execution Allocation-free host metadata query.
 */
GENERATIVEQC_API uint32_t generativeqc_ks_hessian_cpu_version_v1(void);

/** @native-contract generativeqc_ks_hessian_cpu_method_v1
 * @behavior Query the linked CPU derivative catalog by exact MethodIR identity.
 * @inputs NUL-terminated native electronic or total MethodIR identity.
 * @outputs Returns 1 if linked for the identity, 0 for null, missing or unknown identity.
 * @execution Host metadata query; prepared state and capacity admission are separate.
 */
GENERATIVEQC_API uint32_t generativeqc_ks_hessian_cpu_method_v1(const char* identity);

enum {
  GENERATIVEQC_KS_HESSIAN_CPU_MODEL_UNLINKED = 0,
  GENERATIVEQC_KS_HESSIAN_CPU_MODEL_ELECTRONIC_ONLY = 1,
  GENERATIVEQC_KS_HESSIAN_CPU_MODEL_MATCHED = 2,
};

/** @native-contract generativeqc_ks_hessian_cpu_model_match_v1
 * @behavior Authenticate a linked bare electronic Hessian closure and a total
 * method that uses that same closure, with any correction resolved natively.
 * @inputs Two NUL-terminated MethodIR identities; use the same value for a
 * bare method. The first identity must not carry a correction.
 * @outputs UNLINKED if the library cannot bind the electronic identity;
 * ELECTRONIC_ONLY if it binds but the total method is missing or uses another
 * closure; MATCHED if both identities select the same linked electronic owner.
 * @execution Host metadata only; no SCF, Hessian, capacity or system admission.
 */
GENERATIVEQC_API uint32_t generativeqc_ks_hessian_cpu_model_match_v1(
    const char* electronic_identity, const char* total_identity);

/** @native-contract generativeqc_ks_hessian_cpu_default_options_v1
 * @behavior Initialize bounded CPU derivative defaults, size and ABI version.
 * @inputs Pointer to writable options storage.
 * @outputs Overwrites the descriptor; the caller must set expected system fields.
 * @errors A null pointer returns INVALID_ARGUMENT.
 * @execution Does not require a context or linked CPU Hessian provider.
 * @units Capacities in bytes; tolerances apply to the physical response residual.
 */
GENERATIVEQC_API generativeqc_status
generativeqc_ks_hessian_cpu_default_options_v1(generativeqc_ks_host_hessian_options_v1* options);

/** @native-contract generativeqc_ks_hessian_cpu_minimum_bytes_v1
 * @behavior Return the mandatory numerical reservation used by CPU Hessian execution.
 * @inputs Live context and normalized orbital system, exact total MethodIR identity or
 * a native named KS selector (including any correction), dense=1 for a matrix or 0
 * for HVP, and initialized CPU v1 options. The query ignores numeric_bytes, expected
 * geometry and solver settings.
 * @outputs Numerical byte floor, including the configured grid and response capacities.
 * @lifetime Borrows inputs synchronously; serialize with context operations. Inputs and
 * output must not overlap, and no pointers are retained.
 * @errors Unsupported program/system, invalid inputs or modeled sub-cap violations
 * leave the result unchanged. Meeting the floor does not guarantee later grid, solver,
 * work-cap, optional-schedule or final-state admission.
 * @execution Shape and correction sizing only; no SCF, grid construction or ERI work census.
 * @units Bytes; this is a modeled derivative reservation, not a measure of process RSS.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_hessian_cpu_minimum_bytes_v1(
    generativeqc_context* context, const generativeqc_system* system, const char* method,
    uint32_t dense, const generativeqc_ks_host_hessian_options_v1* options, size_t* minimum_bytes);

/** @native-contract generativeqc_ks_hessian_cpu_v1
 * @behavior Differentiate one accepted CPU KS batch item, including any prepared native D3.
 * @inputs Live batch, original input index, initialized options and receipt (set struct_size
 * and abi_version). Expected elements/coordinates/charge/spin must match the live state.
 * For N atoms, supply NULL/0 direction and exactly (3N)^2 outputs, or a finite 3N-entry
 * direction and exactly 3N outputs. All caller and borrowed source buffers must be disjoint.
 * @outputs Raw row-major matrix or atom-major Cartesian HVP and a detached work receipt.
 * @lifetime No pointers retained; serialize against other batch/context operations.
 * @errors Malformed, stale, unsupported, nonconverged, or over-budget requests fail
 * without changing either output. No provider, Python, or compiler fallback.
 * @execution Synchronous FP64 CPU derivative of the current verified physical SCF state.
 * @units Geometry in Bohr; Hessian in Hartree/Bohr^2; HVP acts on the supplied direction.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_hessian_cpu_v1(
    generativeqc_batch* batch, size_t index, const generativeqc_ks_host_hessian_options_v1* options,
    const double* direction, size_t direction_count, double* output, size_t output_count,
    generativeqc_ks_host_hessian_receipt_v1* receipt);

/** @native-contract generativeqc_ks_hessian_cpu_recycled_v1
 * @behavior Differentiate a verified Direct FP64 LDA/PBE RKS batch item with Cartesian or
 * real-spherical AOs; PBE may include linked D3(BJ).
 * @inputs Same batch, expected system options and detached publication as CPU v1 dense;
 * block_size is two through four, not exceeding the number of Cartesian coordinates.
 * @outputs Raw row-major matrix and complete derivative work receipt.
 * @lifetime Synchronous; the recycle space resets at every group and borrows no caller storage.
 * @errors Invalid, unsupported, stale, nonconverged and over-budget requests publish nothing.
 * A requested group that does not fit is not replaced by an independent schedule.
 * @execution The same native stationary owner with sequential, recycled GMRES initial guesses.
 * Every response still passes the physical residual test.
 * @units Block size in RHS directions; geometry in Bohr; Hessian in Hartree/Bohr^2.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_hessian_cpu_recycled_v1(
    generativeqc_batch* batch, size_t index, const generativeqc_ks_host_hessian_options_v1* options,
    size_t block_size, double* output, size_t output_count,
    generativeqc_ks_host_hessian_receipt_v1* receipt);

/** @native-contract generativeqc_ks_hessian_cpu_sequential_v1
 * @behavior Differentiate a verified Direct FP64 LDA/PBE RKS item with Cartesian or
 * real-spherical AOs; PBE may include linked D3(BJ).
 * @inputs Same live batch, expected system, options and detached buffers as CPU v1 dense.
 * @outputs Raw row-major matrix and complete derivative work receipt.
 * @lifetime Synchronous; retains no caller buffers.
 * @errors Invalid, unsupported, stale, nonconverged and over-budget requests publish nothing.
 * @execution Stream one physical CPKS solve at a time, without recycled initial guesses or
 * concurrent RHS actions. Geometry-only source reuse remains available when it fits.
 * @units Geometry in Bohr; Hessian in Hartree/Bohr^2.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_hessian_cpu_sequential_v1(
    generativeqc_batch* batch, size_t index, const generativeqc_ks_host_hessian_options_v1* options,
    double* output, size_t output_count, generativeqc_ks_host_hessian_receipt_v1* receipt);

/** @native-contract generativeqc_ks_hvp_cpu_batch_v1
 * @behavior Apply the stationary CPU KS Hessian to multiple directions of one accepted item,
 * without constructing the dense molecular Hessian.
 * @inputs Same state and options as generativeqc_ks_hessian_cpu_v1. For N atoms, provide a
 * nonempty contiguous array of complete 3N-entry finite Cartesian directions. Counts are in
 * doubles; output_count equals direction_count. All caller/source buffers must be disjoint.
 * @outputs Actions in input order with the same layout, and one detached aggregate receipt.
 * @lifetime Synchronous; retains no caller buffers. Serialize batch/context operations.
 * @errors Invalid, stale, unsupported, over-budget or failed response leaves both destinations
 * unchanged. Output, integral, grid, and applicable correction work gates cover the whole batch.
 * @execution Independent FP64 CPKS solves on the same state. Concurrent RHS actions use bounded
 * groups when admitted; otherwise columns stream without changing the response equations.
 * @units Geometry in Bohr; Hartree/Bohr^2 action on the supplied direction.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_hvp_cpu_batch_v1(
    generativeqc_batch* batch, size_t index, const generativeqc_ks_host_hessian_options_v1* options,
    const double* directions, size_t direction_count, double* output, size_t output_count,
    generativeqc_ks_host_hessian_receipt_v1* receipt);

/** @native-contract generativeqc_ks_hvp_cpu_batch_minimum_bytes_v1
 * @behavior Size the mandatory CPU action-panel reservation before preparing the SCF state.
 * @inputs Same normalized system, exact method and options as the CPU Hessian byte query;
 * directions is a positive count of complete Cartesian perturbations, not a count of doubles.
 * @outputs Detached width-one numerical floor including the whole action publication and snapshot.
 * @lifetime Synchronous; borrows no buffers on return. Serialize context operations.
 * @errors Invalid shape, unsupported system or insufficient component capacity leaves output
 * unchanged. Success does not guarantee numerical convergence or execution-time work admission.
 * @execution Shape-only native planning; does not solve SCF or allocate response panels.
 * @units Capacity in bytes; direction count is dimensionless.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_hvp_cpu_batch_minimum_bytes_v1(
    generativeqc_context* context, const generativeqc_system* system, const char* method,
    size_t directions, const generativeqc_ks_host_hessian_options_v1* options,
    size_t* minimum_bytes);

/** @native-contract generativeqc_ks_hvp_cpu_prepared_minimum_bytes_v1
 * @behavior Quote the mandatory CPU action-panel bytes from the accepted batch item.
 * @inputs Live batch, original input index, positive number of Cartesian directions,
 * and initialized host options with the expected geometry, charge and multiplicity.
 * @outputs Detached width-one numerical floor, using the live normalized sources.
 * @lifetime Synchronous; serialize operations on the batch and its context.
 * @errors Missing, stale, unsupported or over-budget sources leave the output unchanged.
 * @execution Does not normalize a new system, copy a response frame, or run SCF.
 * @units Capacity in bytes; direction count is dimensionless.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_hvp_cpu_prepared_minimum_bytes_v1(
    generativeqc_batch* batch, size_t index, size_t directions,
    const generativeqc_ks_host_hessian_options_v1* options, size_t* minimum_bytes);

#ifdef __cplusplus
}
#endif
#endif
