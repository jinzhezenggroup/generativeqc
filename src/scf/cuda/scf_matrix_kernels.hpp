#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace generativeqc::scf::cuda_execution {

/** Preserve launch geometry, stream and per-item state routing. */
void launch_copy_matrix_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                               std::size_t elements, const double* source, double* destination);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_copy_selected_matrices_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          std::int32_t matrices_per_system, std::int32_t nbf,
                                          const std::uint8_t* selected, const double* source,
                                          double* destination);

/** Extract matrix diagonals for selected systems. */
void launch_extract_matrix_diagonals_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                            cudaStream_t stream, std::int32_t batch_size,
                                            std::int32_t matrices_per_system, std::int32_t nbf,
                                            const std::uint8_t* selected, const double* matrices,
                                            double* diagonals);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_build_orthogonalizer_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                        cudaStream_t stream, std::int32_t batch_size,
                                        std::int32_t nbf, const double* eigenvectors,
                                        const double* eigenvalues, const std::uint8_t* active,
                                        double* orthogonalizer, std::uint8_t* failed);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_matrix_product_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                  const double* left, bool transpose_left, const double* right,
                                  const std::uint8_t* active, double* output, double scale = 1.0);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_broadcast_spin_matrix_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::int32_t batch_size,
                                         std::int32_t spin_count, std::int32_t nbf,
                                         const double* physical_matrices,
                                         const std::uint8_t* active, double* spin_matrices);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_spin_matrix_product_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                       cudaStream_t stream, std::int32_t batch_size,
                                       std::int32_t spin_count, std::int32_t nbf,
                                       const double* left, bool left_is_spin, bool transpose_left,
                                       const double* right, bool right_is_spin,
                                       const std::uint8_t* active, double* output);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_clear_active_matrices_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::int32_t batch_size,
                                         std::int32_t matrices_per_system, std::int32_t nbf,
                                         const std::uint8_t* active, double* matrices);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_subtract_matrix_batches_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                           cudaStream_t stream, std::int32_t batch_size,
                                           std::int32_t matrices_per_system, std::int32_t nbf,
                                           const double* subtract, const std::uint8_t* active,
                                           double* minuend);

/** Prepare #990's device-resident ΔD build and periodic exact refresh independently per item. */
void launch_prepare_incremental_direct_jk_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::int32_t batch_size, std::int32_t spin_count, std::int32_t nbf,
    std::uint32_t rebuild_interval, const double* density, const double* hcore,
    const std::uint8_t* active, double* anchor_density, double* anchor_fock,
    double* delta_density, std::uint32_t* delta_updates, std::uint8_t* full_build,
    double* max_abs_delta_density);

/** Combine h+G(ΔD) with the retained full-Fock anchor, then advance the exact anchor. */
void launch_finalize_incremental_direct_jk_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::int32_t batch_size, std::int32_t spin_count, std::int32_t nbf,
    const double* density, const double* hcore, const std::uint8_t* active,
    double* anchor_density, double* anchor_fock, double* fock,
    std::uint32_t* delta_updates, const std::uint8_t* full_build);

}  // namespace generativeqc::scf::cuda_execution
