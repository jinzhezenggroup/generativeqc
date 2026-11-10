#ifndef GENERATIVEQC_SADDLE_H
#define GENERATIVEQC_SADDLE_H

#include "generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Termination is separate from C ABI argument/execution status. */
typedef enum generativeqc_saddle_status_v1 {
  GENERATIVEQC_SADDLE_FIRST_ORDER = 0,
  GENERATIVEQC_SADDLE_OTHER_INDEX = 1,
  GENERATIVEQC_SADDLE_AMBIGUOUS_CURVATURE = 2,
  GENERATIVEQC_SADDLE_ATTEMPT_LIMIT = 3,
  GENERATIVEQC_SADDLE_TARGET_NOT_NEGATIVE = 4,
  GENERATIVEQC_SADDLE_MODE_TRACKING_LOST = 5,
  GENERATIVEQC_SADDLE_TRUST_EXHAUSTED = 6,
  GENERATIVEQC_SADDLE_EVALUATION_FAILED = 7,
  GENERATIVEQC_SADDLE_HESSIAN_FAILED = 8,
  GENERATIVEQC_SADDLE_CANCELLED = 9
} generativeqc_saddle_status_v1;

/** All controls are explicit; no method selection or preoptimization occurs. */
typedef struct generativeqc_saddle_options_v1 {
  size_t struct_size;
  uint32_t abi_version;
  uint32_t project_rigid_motions;
  uint64_t max_attempts;
  uint64_t max_coordinates;
  uint64_t maximum_numeric_bytes;
  double energy_change;
  double gradient_rms;
  double gradient_max;
  double displacement_rms_angstrom;
  double displacement_max_angstrom;
  double initial_radius;
  double minimum_radius;
  double maximum_radius;
  double curvature_tolerance;
  double raw_symmetry_tolerance;
  double minimum_mode_overlap;
  double acceptance_ratio;
} generativeqc_saddle_options_v1;

/** Last fully accepted state, including fresh-final-verification status. */
typedef struct generativeqc_saddle_receipt_v1 {
  size_t struct_size;
  uint32_t abi_version;
  uint32_t status;
  uint32_t has_valid_state;
  uint32_t stationary_verified;
  uint64_t attempts;
  uint64_t accepted_steps;
  uint64_t energy_gradient_calls;
  uint64_t hessian_calls;
  uint64_t negative_curvatures;
  uint64_t near_zero_curvatures;
  uint64_t rigid_modes;
  uint64_t active_modes;
  uint64_t numeric_capacity_bytes;
  double energy;
  double energy_change;
  double gradient_rms;
  double gradient_max;
  double displacement_rms_angstrom;
  double displacement_max_angstrom;
  double trust_radius;
  double target_curvature;
  double target_overlap;
} generativeqc_saddle_receipt_v1;

/** @native-contract generativeqc_saddle_evaluator_v1
 * @behavior Supply one requested derivative stage of a fixed model.
 * @inputs user is the caller's borrowed context; coordinates has count finite Bohr
 * values. stage=0 requests energy/gradient and stage=1 requests a raw Hessian.
 * @outputs Only the requested output pointers are nonnull. Write one energy and
 * count gradient values for stage=0, or count*count row-major Hessian values for
 * stage=1. Return zero only when the requested work is complete and valid.
 * @lifetime All buffers are borrowed until callback return; retain none of them.
 * @errors Nonzero terminates with an explicit derivative-stage failure receipt.
 * Never throw through a C callback or consume an unconverged electronic state.
 * @execution Synchronous CPU callback; model/state and callback memory are owned
 * by the caller and excluded from the optimizer's numerical capacity bound.
 * @units Hartree, Bohr, Hartree/Bohr and Hartree/Bohr^2; gradients are not forces.
 */
typedef int (*generativeqc_saddle_evaluator_v1)(void* user, uint32_t stage,
                                                const double* coordinates, size_t count,
                                                double* energy, double* gradient, double* hessian);
/** @native-contract generativeqc_saddle_progress_v1
 * @behavior Observe initial, accepted/rejected and fresh-final states.
 * @inputs user is the caller's context; receipt and count coordinates describe
 * the last fully accepted state, not an unaccepted trial geometry.
 * @outputs Return zero to continue or nonzero to request cancellation.
 * @lifetime Borrowed immutable observation, valid only until callback return.
 * @errors Do not throw. Nonzero cancels before further derivative work and clears
 * stationary verification, including when observing the final state.
 * @execution Synchronous on the calling thread; observations do not themselves
 * consume an optimizer attempt or a derivative evaluation.
 * @units Coordinates Bohr; receipt units match generativeqc_saddle_correct_v1.
 */
typedef int (*generativeqc_saddle_progress_v1)(void* user,
                                               const generativeqc_saddle_receipt_v1* receipt,
                                               const double* coordinates, size_t count);
/** @native-contract generativeqc_saddle_cancellation_v1
 * @behavior Check whether the caller requests cancellation.
 * @inputs user is the caller's borrowed opaque context, which may be NULL.
 * @outputs Return nonzero to cancel or zero to continue.
 * @lifetime No context is retained after the synchronous optimizer call.
 * @errors Do not throw; cancellation is not successful convergence.
 * @execution Checked before derivative work and after progress observations.
 * @units No physical units.
 */
typedef int (*generativeqc_saddle_cancellation_v1)(void* user);

/** @native-contract generativeqc_saddle_correct_v1
 * @behavior Execute the native local first-order-saddle corrector with exact initial,
 * accepted-step and final Hessians. Every accepted/rejected trial consumes one attempt
 * from max_attempts<=100; initial preparation and final verification are separate calls.
 * A nonstationary input requires an already negative target mode, not minimum escape.
 * @inputs Initialize options and receipt with sizeof(type) and GENERATIVEQC_ABI_VERSION.
 * All controls are explicit, finite and positive; coordinate triples are finite Bohr.
 * project_rigid_motions is zero or one. The optional initial_mode has count values.
 * All numerical output pointers and detail are required and must not overlap inputs
 * or one another. coordinate_count<=max_coordinates<=300. detail_capacity>=1.
 * evaluator is required; progress and cancelled may be NULL. Callbacks bind one fixed
 * physical model/state policy; this mathematical owner does not establish their chemistry.
 * @outputs Receipt distinguishes all termination reasons and first-order index, preserving
 * the last fully accepted geometry, gradient and raw Hessian on algorithm failure. Modes
 * and curvatures refer to the explicitly projected vibrational space when requested;
 * no small vibrational eigenvalue is deleted. Only active_modes curvatures are written.
 * A successful C call does not itself mean optimizer convergence; inspect receipt.status
 * and stationary_verified. detail is NUL terminated. Initial failure has no valid state.
 * @lifetime All input/output and callback buffers are borrowed for this synchronous call.
 * No pointer is retained. Callbacks must not retain temporary coordinates or receipts.
 * @errors Invalid controls, dimensions or ABI reject before derivative callbacks and
 * preserve numerical outputs/receipt. Capacity refusal returns RESOURCE_LIMIT. Algorithm
 * failures return SUCCESS with an explicit failure receipt and last-valid-state outputs.
 * @execution Synchronous CPU-only native optimization and shared symmetric eigensolver.
 * No SCF, oracle, finite differences, external optimizer or implicit lower-level stage is
 * launched. maximum_numeric_bytes bounds this owner, excluding callback/provider memory.
 * RMS denominators are 3N, with no implicit rigid alignment. Cancellation has no success
 * semantics. Serialize calls sharing callback resources.
 * @units Energy Hartree, coordinates/trust radii Bohr, gradients Hartree/Bohr, raw Hessians
 * Hartree/Bohr^2; displacement convergence controls and receipts are Angstrom. Curvatures
 * are unweighted Cartesian vibrational curvatures, not mass-weighted frequencies.
 */
GENERATIVEQC_API generativeqc_status generativeqc_saddle_correct_v1(
    const generativeqc_saddle_options_v1* options, const double* initial_coordinates,
    size_t coordinate_count, const double* initial_mode, generativeqc_saddle_evaluator_v1 evaluator,
    generativeqc_saddle_progress_v1 progress, generativeqc_saddle_cancellation_v1 cancelled,
    void* user, generativeqc_saddle_receipt_v1* receipt, double* coordinates, double* gradient,
    double* hessian, double* mode, double* curvatures, char* detail, size_t detail_capacity);

#ifdef __cplusplus
}
#endif

#endif
