#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "scf/cuda_direct_jk.hpp"

namespace generativeqc::scf {

/** Borrow the provider's ordinary nonblocking stream. Density producers,
 * XC and matrix consumers can enqueue on it without an intervening host
 * transfer or fence. The plan must outlive all borrowers; destruction drains
 * the stream before releasing immutable integral metadata. Null is invalid. */
cudaStream_t cuda_direct_jk_stream(const CudaDirectJkPlan* plan);
/** Device ordinal in the current process's visibility namespace; null returns -1. */
int cuda_direct_jk_device(const CudaDirectJkPlan* plan) noexcept;

/** Prepare the ordinary exact provider on a non-null caller-owned stream.
 * Preparation completes its uploads before returning. Destruction drains but
 * never destroys this stream. The caller must retire captured borrowers before
 * destroying the plan, and keep the stream alive until destruction completes.
 */
generativeqc_status create_cuda_direct_jk_plan_on_stream(
    int device_id, const std::vector<core::System>& systems, unsigned derivative_order,
    double screening_tolerance, std::size_t device_budget_bytes, cudaStream_t stream,
    CudaDirectJkPlan** output, CudaDirectJkDiagnostic& diagnostic, std::string& detail);

/** Prepare a zero-order phase source with the same borrowed-stream contract.
 * Reserve canonical storage before any optional operator leases: generated
 * J/K caches, MD and range sources are not used by resident replay. Ordinary
 * providers retain their existing preparation priority and bounded fallback.
 */
generativeqc_status create_cuda_direct_jk_resident_plan_on_stream(
    int device_id, const std::vector<core::System>& systems, double screening_tolerance,
    std::size_t device_budget_bytes, cudaStream_t stream, CudaDirectJkPlan** output,
    CudaDirectJkDiagnostic& diagnostic, std::string& detail);

/** Bound optional Schwarz row/sort storage for a Cartesian source shape.
 * Queries CUB workspace on the current CUDA device without allocating or
 * evaluating ERIs. Include this in admission before constructing the source.
 */
generativeqc_status cuda_direct_jk_resident_layout_device_bytes(std::size_t batch,
                                                                std::size_t direct_nbf,
                                                                std::size_t& bytes,
                                                                std::string& detail);

/** Freeze compact extents of the already prepared geometry-only Schwarz rows.
 * Positive masks require admitted screened-row metadata; no ERI is evaluated.
 * Download all bucket tails once on the provider stream, never during replay.
 */
generativeqc_status prepare_cuda_direct_jk_resident_layout(CudaDirectJkPlan* plan,
                                                           std::string& detail);

/** Inventory for an optional full-range canonical source lease.
 * A positive preparation threshold needs a previously frozen compact layout.
 * Zero denotes an unavailable domain. This query neither evaluates integrals
 * nor initializes CUDA; the immutable Direct plan remains its scientific owner.
 */
std::size_t cuda_direct_jk_resident_value_bytes(const CudaDirectJkPlan* plan);

/** Evaluate each canonical source value once on the owning stream and retain
 * it under maximum_bytes. Preparation fences and finite-audits the complete
 * source before publication. Insufficient capacity returns OUT_OF_MEMORY with
 * the original exact route intact; driver/numerical failures must propagate.
 * Values are released with the plan. Repeated preparation is idempotent.
 */
generativeqc_status prepare_cuda_direct_jk_resident_values(CudaDirectJkPlan* plan,
                                                           std::size_t maximum_bytes,
                                                           std::string& detail);

/** Cache FP32 recurrence values in FP32 storage, never strict ERIs.
 * Restricted through-d sources share full/SR/LR identity and bounded angular
 * selection with strict sources. Unselected buckets evaluate mixed recurrences
 * on replay. Publication follows finite validation; strict callers cannot borrow
 * this precision lease or change its immutable operator, omega or bucket mask.
 */
generativeqc_status prepare_cuda_direct_jk_mixed_resident_values(
    CudaDirectJkPlan* plan, std::size_t maximum_bytes, std::string& detail,
    FockOperator source_operator = FockOperator::FullRange, double omega = 0.0,
    std::uint32_t bucket_mask = (1U << 28) - 1);

/** Retain exact values of one immutable radial operator, with bounded partial
 * residency. SR/LR values can serve only matching pure-exchange requests; full
 * J/K and a different omega keep their ordinary exact provider. A successful
 * lease never changes operator identity, even when preparation is repeated.
 */
generativeqc_status prepare_cuda_direct_jk_range_resident_values(CudaDirectJkPlan* plan,
                                                                 std::size_t maximum_bytes,
                                                                 std::uint32_t bucket_mask,
                                                                 FockOperator source_operator,
                                                                 double omega, std::string& detail);

/** Select whole immutable angular buckets, numbered first*(first+1)/2+second.
 * The mask is shared across batch items. Unselected buckets retain the native
 * canonical evaluator, not a second whole-source action. Selection affects
 * storage/scheduling only; the full-range FP64 equation and Schwarz predicate
 * are unchanged. Screened inventories require the frozen resident layout.
 * A zero mask retains Direct without allocation. Published leases cannot be
 * replaced in place, since captured consumers may borrow their pointers. */
std::size_t cuda_direct_jk_selected_resident_value_bytes(const CudaDirectJkPlan* plan,
                                                         std::uint32_t bucket_mask);
generativeqc_status prepare_cuda_direct_jk_selected_resident_values(CudaDirectJkPlan* plan,
                                                                    std::size_t maximum_bytes,
                                                                    std::uint32_t bucket_mask,
                                                                    std::string& detail);

/** Add bounded pageable FP64 backing to omitted buckets of a strict LR lease.
 * maximum_bytes includes pageable values and both pinned tiles; device_budget
 * includes both device tiles. A refusal publishes nothing and preserves the
 * existing GPU source. Only one immutable system and eager replay are supported.
 */
generativeqc_status prepare_cuda_direct_jk_host_values(
    CudaDirectJkPlan* plan, std::size_t maximum_bytes, std::size_t device_budget,
    std::size_t tile_bytes, std::uint32_t bucket_mask, std::string& detail);
std::size_t cuda_direct_jk_host_value_bytes(const CudaDirectJkPlan* plan) noexcept;
std::size_t cuda_direct_jk_host_capacity_bytes(const CudaDirectJkPlan* plan) noexcept;

/** Enqueue one exact, unscreened full-range AO ERI tile into caller-owned
 * device storage on caller_stream. The tile is row-major [i,j,k,l] with the
 * last axis fastest and uses the prepared Direct plan's public AO basis.
 *
 * This borrows immutable geometry/basis metadata only. It does not use the
 * HF screening tolerance, allocate storage, transfer through host memory, or
 * synchronize on success. The plan must outlive completion on caller_stream.
 */
generativeqc_status enqueue_cuda_direct_eri_tile(CudaDirectJkPlan* plan, std::size_t item,
                                                 const std::array<std::size_t, 4>& begin,
                                                 const std::array<std::size_t, 4>& count,
                                                 double* output, std::size_t elements,
                                                 cudaStream_t caller_stream, std::string& detail);

/** Enqueue raw, unscaled value J/K against caller-owned device matrices.
 *
 * Arrays are row-major [item,AO,AO] over the complete homogeneous plan;
 * matrix_elements equals batch_size*nbf*nbf. Beta is required only for UKS.
 * Requested outputs and the error integer must be non-null and disjoint
 * from inputs and one another; absent outputs must be null. Inputs may be
 * nonsymmetric, preserving the existing common-provider convention.
 *
 * Uses cuda_direct_jk_stream(plan), consumes no host density and performs no
 * allocation, D2H/H2D or success-path synchronization. SUCCESS means enqueue
 * succeeded: numerical_error is set to zero, then nonfinite input/output sets
 * it to one on the stream. The caller checks it with its scalar diagnostics.
 * All borrowed buffers must remain alive until work completes. A new enqueue
 * resets only the supplied error slot, allowing independent item owners.
 * This value-only seam does not advertise device force/derivative support.
 */
generativeqc_status enqueue_cuda_direct_jk_device(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                                  const double* density, const double* beta,
                                                  std::size_t matrix_elements, double* coulomb,
                                                  double* alpha_exchange, double* beta_exchange,
                                                  int* numerical_error, std::string& detail,
                                                  std::uint64_t* census = nullptr);

/** Enqueue an explicitly provisional, coarser screened full-range RKS J/K.
 * Only existing MD-J and generated K owners are admitted. Unsupported consumers
 * return NOT_IMPLEMENTED before submission, so the caller can use ordinary J/K.
 * Screening may only be coarser than the immutable prepared source. This is not
 * a physical Fock certificate: the solver must close with its original operator.
 */
generativeqc_status enqueue_cuda_direct_jk_proposal_device(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const double* density, std::size_t matrix_elements,
    double* coulomb, double* exchange, int* numerical_error, double screening, std::string& detail);

/** Whether an ordinary value action has complete canonical census coverage.
 * Generated/generic channels do not silently report zero as measured work. */
bool cuda_direct_jk_value_census_available(const CudaDirectJkPlan* plan,
                                           FockBuildSpec spec) noexcept;

/** Whether J can use the generated strict shell consumer for this operator.
 * Used by the KS proposal scheduler to prefer profitable FP64 work over the
 * generic mixed recurrence. This query does not change K or allocate storage. */
bool cuda_direct_jk_generated_coulomb_available(const CudaDirectJkPlan* plan,
                                                FockBuildSpec spec) noexcept;

/** Whether the immutable canonical quartet source can supply a density-independent
 * linear action. This optional schedule retains no four-index integral tensor. */
bool cuda_direct_jk_linear_available(const CudaDirectJkPlan* plan) noexcept;

/** Fixed geometry-only Schwarz mask for response, including signed densities.
 * Uses the canonical quartet's complete permutation orbit for both J and K;
 * never enters density-dependent shell screening. threshold must be finite,
 * nonnegative and at least the plan's preparation threshold. An unscreened
 * plan therefore supports both approximate actions and exact residual audits
 * without changing metadata, row storage or stream ownership.
 * Optional device census[2] counts visited canonical quartets and evaluated
 * ERI values, not FLOPs. It is reset per call and must not alias any buffer.
 * NOT_IMPLEMENTED means the optional canonical storage was not admitted. */
generativeqc_status enqueue_cuda_direct_jk_linear_device(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const double* density, std::size_t matrix_elements,
    double* coulomb, double* exchange, int* numerical_error, double threshold,
    std::uint64_t* census, std::string& detail);

/** Required caller-owned elements for the J/K canonical correction
 * planes. The canonical Cartesian dimension may exceed the public AO count.
 * Zero denotes unavailable canonical storage; no CUDA work or allocation. */
std::size_t cuda_direct_jk_compensation_elements(const CudaDirectJkPlan* plan, unsigned spins = 1);

/** Accuracy-qualified full-range fixed-mask action. Same stream,
 * lifetime, census and finite-error contract as the linear action above.
 * correction must have exactly cuda_direct_jk_compensation_elements(plan, spins)
 * doubles, disjoint from all inputs, outputs and diagnostic slots. It is reset
 * per action and folded before canonical-to-public projection. A compatible mask
 * may reuse admitted resident values unless allow_resident is false;
 * no allocation or success-path fence.
 * Prepared whole-shell recurrence preserves the same AO Schwarz mask and
 * compensation planes. Disable allow_materialized for an independent
 * per-component replay, also disabling allow_resident for an independent audit.
 * Missing prepared storage retains that bounded replay.
 * Unavailable canonical storage returns NOT_IMPLEMENTED without enqueueing.
 */
generativeqc_status enqueue_cuda_direct_jk_compensated_device(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const double* density, std::size_t matrix_elements,
    double* coulomb, double* exchange, double* correction, std::size_t correction_elements,
    int* numerical_error, double threshold, std::uint64_t* census, std::string& detail,
    bool allow_resident = true, const double* beta = nullptr, double* beta_exchange = nullptr,
    bool allow_materialized = true);

/** Replay an authenticated full-range FP32 source with FP64 products/scatter.
 * Requires complete mixed residency; no recurrence or strict-source fallback
 * can masquerade as this action. Correction/output lifetime matches FP64.
 */
generativeqc_status enqueue_cuda_direct_jk_mixed_resident_device(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const double* density, std::size_t matrix_elements,
    double* coulomb, double* exchange, double* correction, std::size_t correction_elements,
    int* numerical_error, double threshold, std::uint64_t* census, std::string& detail);

/** Eligibility for the default reuse-preserving shell bilinear derivative.
 * Canonical residency alone does not admit this schedule. No device work is
 * submitted; unavailable shell storage preserves bounded polarization. */
bool cuda_direct_jk_bilinear_preferred(const CudaDirectJkPlan* plan) noexcept;

/** One unscreened canonical traversal of P:(J'(D)-K'(D)/2).
 * Restricted, single-system, retained first-derivative plan only. Both input
 * matrices use the public AO frame and may be signed. The optional device
 * census[2] records canonical quartet visits and evaluated three-axis center
 * derivative jets; these counts are not FLOPs. Returns host electronic gradient
 * coordinates, after finite audit and stream completion. Optional canonical
 * storage refusal returns NOT_IMPLEMENTED for the caller's bounded fallback. */
generativeqc_status execute_cuda_direct_bilinear_derivative_device(
    CudaDirectJkPlan* plan, const double* density, const double* seed, std::size_t matrix_elements,
    std::vector<double>& gradient, std::uint64_t* census, std::string& detail);

/** Enqueue primary full-range J/K and one exact SR/LR K correction after one
 * Direct shell-density preparation. This seam is available only when the
 * retained bounded shell owner covers the range value request; callers keep
 * the ordinary two-call fallback for other domains. */
generativeqc_status enqueue_cuda_direct_rsh_values_device(
    CudaDirectJkPlan* plan, FockBuildSpec primary, FockBuildSpec correction, const double* density,
    const double* beta, std::size_t matrix_elements, double* coulomb, double* full_alpha_exchange,
    double* full_beta_exchange, double* range_alpha_exchange, double* range_beta_exchange,
    int* primary_error, int* range_error, std::string& detail);

/** Experimental value-only variant: evaluate Coulomb ERI recurrences in FP32
 * while retaining FP64 density reads, screening, accumulation and output.
 * Exchange remains FP64 unless mixed_exchange selects restricted full/SR/LR
 * J/K sharing the same FP32 recurrence. The optional device counter is cleared
 * on the provider stream and counts only AO-ERI values actually evaluated by
 * the mixed recurrence after screening and zero-density rejection. The caller
 * must perform a strict FP64 target audit before publishing a converged state. */
generativeqc_status enqueue_cuda_direct_jk_device_mixed_j(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const double* density, const double* beta,
    std::size_t matrix_elements, double* coulomb, double* alpha_exchange, double* beta_exchange,
    int* numerical_error, std::string& detail, std::uint64_t* mixed_coulomb_work_count = nullptr,
    bool mixed_exchange = false);
}  // namespace generativeqc::scf
