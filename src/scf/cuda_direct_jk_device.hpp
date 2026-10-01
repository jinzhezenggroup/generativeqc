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
generativeqc_status enqueue_cuda_direct_eri_tile(
    CudaDirectJkPlan* plan, std::size_t item,
    const std::array<std::size_t, 4>& begin,
    const std::array<std::size_t, 4>& count,
    double* output, std::size_t elements, cudaStream_t caller_stream,
    std::string& detail);

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
