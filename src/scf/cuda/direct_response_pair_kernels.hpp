#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace generativeqc::scf::cuda_execution {

/** Reindex Coulomb rows into the even-under-density-transpose exchange part. */
cudaError_t prepare_exchange_pair_rows(cudaStream_t stream, const double* coulomb_rows,
                                       double* exchange_rows, std::size_t stride,
                                       std::size_t nao) noexcept;

/** Apply raw J and K in the public AO basis. The scratch holds two packed
 * density orientations. The odd exchange part borrows the same Coulomb rows
 * when needed, without allocating another matrix. */
cudaError_t enqueue_hybrid_pair_rows(cudaStream_t stream, const double* coulomb_rows,
                                     const double* exchange_rows, std::size_t stride,
                                     std::size_t nao, const double* density, double* scratch,
                                     double* coulomb, double* exchange) noexcept;

}  // namespace generativeqc::scf::cuda_execution
