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
                                                  int* numerical_error, std::string& detail);

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

/** Prefer the bilinear canonical derivative where the fast shell derivative
 * lease is absent (notably through-f). Other domains retain their bounded
 * three-pass shell consumer until a measured crossover is qualified. */
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
 * Exchange remains FP64. The optional caller-owned device counter is cleared
 * on the provider stream and counts only AO-ERI values actually evaluated by
 * the mixed recurrence after screening and zero-density rejection. The caller
 * must perform a strict FP64 target audit before publishing a converged state. */
generativeqc_status enqueue_cuda_direct_jk_device_mixed_j(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const double* density, const double* beta,
    std::size_t matrix_elements, double* coulomb, double* alpha_exchange, double* beta_exchange,
    int* numerical_error, std::string& detail, std::uint64_t* mixed_coulomb_work_count = nullptr);
}  // namespace generativeqc::scf
