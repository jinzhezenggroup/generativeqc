#pragma once

#include <cstdint>

namespace generativeqc::solver::cuda {

/** Query FP64 Cholesky scratch for one column-major lower triangle, lda=n.
 * The caller validates dimensions/storage and borrows an already stream-bound
 * solver handle. Workspace is reported in double elements, not bytes; bounded
 * admission and checked conversion belong to the resource/method owner.
 * No handle creation, allocation, synchronization or stream mutation occurs.
 * The return is the unmodified provider status, not numerical factorization info.
 */
std::uint32_t query_lower_cholesky(void* solver, int n, double* matrix,
                                   int& workspace_elements) noexcept;

/** Overwrite the lower triangle with L for A=L L^T on the borrowed stream.
 * The caller supplies the queried device workspace and device info slot and
 * retains their lifetime until stream completion. Numerical info is asynchronous:
 * interpreting non-positive-definiteness and selecting a fallback stay with the
 * method owner. This provider submits exactly one factorization.
 */
std::uint32_t launch_lower_cholesky(void* solver, int n, double* matrix, double* workspace,
                                    int workspace_elements, int* info) noexcept;

}  // namespace generativeqc::solver::cuda
