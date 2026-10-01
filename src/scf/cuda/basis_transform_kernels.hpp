#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace generativeqc::scf::cuda_execution {

/** Preserve launch geometry, stream and per-item state routing. */
void launch_initialize_direct_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          std::int32_t matrices_per_system, std::int32_t nbf,
                                          const double* hcore, const std::uint8_t* active,
                                          double* fock);

/** Optional shell-local projection spans, packed as (begin,end) pairs.
 * Public rows for all systems precede source columns for all systems. Spans
 * must contain every nonzero transform coefficient; null retains dense HF
 * projection. This reduces through-f projection work to O(NAO^2 * shell size).
 */
void launch_transform_density_to_direct_right_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t spin_count, std::int32_t nbf, std::int32_t direct_nbf, const double* transform,
    const double* density, const std::uint8_t* active, double* temporary,
    const std::int32_t* shell_spans = nullptr);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_transform_density_to_direct_left_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t spin_count, std::int32_t nbf, std::int32_t direct_nbf, const double* transform,
    const double* temporary, const std::uint8_t* active, double* direct_density,
    const std::int32_t* shell_spans = nullptr);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_transform_direct_fock_left_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                              cudaStream_t stream, std::int32_t batch_size,
                                              std::int32_t spin_count, std::int32_t nbf,
                                              std::int32_t direct_nbf, const double* transform,
                                              const double* direct_fock, const std::uint8_t* active,
                                              double* temporary,
                                              const std::int32_t* shell_spans = nullptr);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_transform_direct_fock_right_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                               cudaStream_t stream, std::int32_t batch_size,
                                               std::int32_t spin_count, std::int32_t nbf,
                                               std::int32_t direct_nbf, const double* transform,
                                               const double* temporary, const double* hcore,
                                               const std::uint8_t* active, double* fock,
                                               const std::int32_t* shell_spans = nullptr);

}  // namespace generativeqc::scf::cuda_execution
