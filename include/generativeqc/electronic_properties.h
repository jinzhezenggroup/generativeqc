#ifndef GENERATIVEQC_ELECTRONIC_PROPERTIES_H
#define GENERATIVEQC_ELECTRONIC_PROPERTIES_H

#include "generativeqc/generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct generativeqc_electronic_properties generativeqc_electronic_properties;

/** @native-contract generativeqc_electronic_properties_cuda_create_v1
 * @behavior Create a detached CUDA FP64 property/sampling owner, without SCF.
 * @inputs CUDA context, normalized all-electron through-f system, finite symmetric
 * row-major restricted total or alpha then beta density, exact density_count of
 * nAO*nAO or 2*nAO*nAO, positive point_capacity <= INT_MAX, non-NULL output.
 * @outputs Owned handle; output is NULL on failure. Density and basis are copied.
 * @errors INVALID_ARGUMENT for malformed inputs; NOT_IMPLEMENTED without CUDA or
 * for CPU contexts/ECP systems; allocation and CUDA errors propagate. No CPU fallback.
 * @lifetime Owner survives input system/context destruction. Destroy explicitly.
 * @execution Synchronous upload on context device; serialize owner operations.
 * @units Geometry Bohr; density in normalized public AO basis, electrons.
 */
GENERATIVEQC_API generativeqc_status generativeqc_electronic_properties_cuda_create_v1(
    generativeqc_context* context, const generativeqc_system* system, const double* density,
    size_t density_count, size_t point_capacity, generativeqc_electronic_properties** output);

/** @native-contract generativeqc_electronic_properties_cuda_analyze_v1
 * @behavior Compute dipole and optional Mulliken/Lowdin charges entirely on CUDA.
 * @inputs Live owner and CUDA context on its device; optional dipole[3], and either
 * both Mulliken/Lowdin[natom] outputs with exact atom_count or neither with zero.
 * At least one property must be requested.
 * @outputs Owned host output buffers, published synchronously after device work.
 * @errors INVALID_ARGUMENT for shapes/device mismatch; NUMERICAL_FAILURE for
 * nonpositive overlap eigenvalues/nonfinite results; CUDA/allocation errors.
 * Output contents on failure are unspecified; a later call may retry.
 * @lifetime Caller owns outputs; no borrowed pointers escape.
 * @execution FP64 overlap/position kernels, cuSOLVER and cuBLAS; no host analysis.
 * Serialize with evaluation/destruction. Restores the caller's current device.
 * @units Dipole e*Bohr including nuclei; charges in elementary-charge units.
 */
GENERATIVEQC_API generativeqc_status generativeqc_electronic_properties_cuda_analyze_v1(
    generativeqc_context* context, generativeqc_electronic_properties* properties, double* dipole,
    double* mulliken, double* lowdin, size_t atom_count);

/** @native-contract generativeqc_electronic_properties_cuda_evaluate_v1
 * @behavior Sample total or alpha-minus-beta density using generated CUDA AOs.
 * @inputs Live owner and same-device CUDA context; finite row-major points[n,3],
 * spin equal to 0 or 1, output[n] with output_count == point_count. Empty calls
 * permit NULL points/output. Nonempty calls require both buffers.
 * @outputs Host density values; native point tiles respect owner's point_capacity.
 * @errors INVALID_ARGUMENT for bad inputs; CUDA/allocation/numerical errors.
 * No host AO or density-contraction fallback. Failure may leave partial output.
 * @lifetime Caller owns all buffers. Owner retains only its detached state.
 * @execution Synchronous FP64 generated AO/contraction path; serialize owner use.
 * Restores the caller's current device, including on failure.
 * @units Points Bohr; output electrons/Bohr**3, signed for spin density.
 */
GENERATIVEQC_API generativeqc_status generativeqc_electronic_properties_cuda_evaluate_v1(
    generativeqc_context* context, generativeqc_electronic_properties* properties,
    const double* points, size_t point_count, int spin, double* output, size_t output_count);

/** @native-contract generativeqc_electronic_properties_destroy_v1
 * @behavior Destroy a detached electronic-property owner; NULL is a no-op.
 * @inputs Owner or NULL; externally serialize against all owner operations.
 * @outputs None; outstanding device work is drained before resources retire.
 * @errors No exceptions cross the ABI; destruction does not report CUDA errors.
 * @lifetime Invalidates owner and releases all host/device storage.
 * @execution Synchronous teardown on its device; restores caller device.
 * @units Not applicable.
 */
GENERATIVEQC_API void generativeqc_electronic_properties_destroy_v1(
    generativeqc_electronic_properties* properties);

/** @native-contract generativeqc_calculation_execute_density_v1
 * @behavior Execute one mean-field calculation and detach its converged AO density.
 * No second SCF, reference solver, or correlated-density approximation is used.
 * @inputs Live RHF/UHF/RKS/UKS calculation; ordinary versioned result descriptor;
 * non-NULL density buffer with exactly nAO*nAO (restricted) or 2*nAO*nAO doubles.
 * @outputs Ordinary energy/optional forces and row-major density. Restricted
 * density includes double occupation; unrestricted stores alpha followed by beta.
 * @errors Ordinary execution errors; NOT_IMPLEMENTED for non-mean-field methods,
 * NOT_CONVERGED for an unaccepted state, INVALID_ARGUMENT for wrong capacity.
 * A rejected capture revokes publication; output contents on failure are unspecified.
 * @lifetime Caller owns all output buffers; copies survive calculation destruction.
 * @execution Synchronous on the prepared backend, with explicit host density export.
 * Serialize against all other operations on the calculation/context.
 * @units Energy Hartree, geometry Bohr, density in the normalized public AO basis.
 */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_execute_density_v1(
    generativeqc_calculation* calculation, generativeqc_result_descriptor* output, double* density,
    size_t density_count);

/** @native-contract generativeqc_ks_batch_copy_density_v1
 * @behavior Detach the accepted density of one completed KS batch item without SCF replay.
 * @inputs Live batch, zero-based item index, non-NULL output of exactly one
 * restricted or two unrestricted square AO matrices, as for execute_density_v1.
 * @outputs Row-major restricted total density or alpha then beta matrices.
 * @errors INVALID_ARGUMENT for bad pointers/count/index; NOT_IMPLEMENTED for
 * non-KS owners; final-state validation rejects stale or unsuccessful executions.
 * @lifetime Caller owns the detached output; no borrowed device pointers escape.
 * @execution Synchronous token-checked native final-state export. Externally serialize
 * with batch execute, coordinate changes, reset and destruction.
 * @units Normalized public AO basis; restricted density includes double occupation.
 */
GENERATIVEQC_API generativeqc_status generativeqc_ks_batch_copy_density_v1(
    generativeqc_batch* batch, size_t index, double* density, size_t density_count);

/** @native-contract generativeqc_system_position_integrals_cpu_v1
 * @behavior Evaluate the three normalized AO position-operator matrices.
 * @inputs Live context/system; non-NULL output of exactly 3*nAO*nAO doubles.
 * @outputs Axis-major row-major <mu|r_axis|nu>, with origin at (0,0,0).
 * @errors INVALID_ARGUMENT for NULL inputs or wrong capacity; native failures propagate.
 * @lifetime Caller owns the output, which outlives the context/system.
 * @execution Explicit synchronous CPU FP64 Gaussian analysis; no SCF or ERIs.
 * @units Bohr; public Cartesian or real-spherical AO ordering and normalization.
 */
GENERATIVEQC_API generativeqc_status generativeqc_system_position_integrals_cpu_v1(
    generativeqc_context* context, const generativeqc_system* system, double* output,
    size_t output_count);

#ifdef __cplusplus
}
#endif

#endif
