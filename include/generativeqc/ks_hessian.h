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

#ifdef __cplusplus
}
#endif
#endif
