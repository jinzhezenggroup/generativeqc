#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace generativeqc::scf::cuda_df {

/** Initialize a column-major identity in already admitted setup scratch. */
void launch_metric_identity(cudaStream_t stream, std::size_t dimension, double* matrix);

/** Launch on the caller's stream, compacting the logical work domain when legal. */
void launch_symmetrize_metrics_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::size_t dimension, double* metrics);

/** Launch on the caller's stream, remapping dense metric columns when legal. */
void launch_scale_eigenvectors_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::size_t matrix_elements,
                                      std::size_t dimension, const double* eigenvectors,
                                      const double* scales, double* scaled_eigenvectors);

/** Scale column-major [eigendirection,pair] projections in place.
 * The caller must establish full metric rank; no cutoff is changed here.
 * square_root selects whitening rather than the full inverse used for J.
 */
void launch_scale_metric_projection(cudaStream_t stream, std::size_t dimension, std::size_t pairs,
                                    const double* eigenvalues, bool square_root, double* projected);

/** Scale column-major [eigendirection,pair] projections into disjoint storage.
 * This preserves the same elementwise division while allowing callers to fuse
 * a required retention copy into the scaling pass.
 */
void launch_scale_metric_projection_to(cudaStream_t stream, std::size_t dimension,
                                       std::size_t pairs, const double* eigenvalues,
                                       bool square_root, const double* projected, double* scaled);

/** Form column-major R=lambda^-1/2 Q^T for a full-rank spectral forward factor.
 * Response borrows its otherwise unused inverse-matrix workspace for R. The
 * output must be disjoint from Q/eigenvalues; no allocation or cutoff occurs.
 */
void launch_spectral_response_root(cudaStream_t stream, std::size_t dimension,
                                   const double* eigenvectors, const double* eigenvalues,
                                   double* root);

}  // namespace generativeqc::scf::cuda_df
