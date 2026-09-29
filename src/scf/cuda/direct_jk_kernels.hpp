#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

struct ShellPairDensityBounds;

// Shared one-output-owner reduction width, unchanged from the direct source.
constexpr unsigned kIndependentJkThreads = 32;

/** Device-only validity reduction for the allocation-free provider seam. */
void launch_independent_jk_finite_kernel(cudaStream_t stream, const double* values,
                                         std::size_t count, int* failure);

/** Preserve the exact public-AO consumer launch and borrowed allocations. */
void launch_independent_jk_bounds_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, DeviceBatch batch, double* bounds,
                                         int* failure);

/** Preserve the exact public-AO consumer launch and borrowed allocations. */
void launch_independent_jk_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, DeviceBatch batch, std::size_t system_begin,
                                  bool want_j, bool want_k, bool unrestricted, bool mixed_j,
                                  DirectCoulombRange exchange_range, double exchange_omega,
                                  double screening, const double* bounds, const double* density,
                                  const double* beta, double* j_out, double* ka_out,
                                  double* kb_out);

/** Preserve the exact public-AO consumer launch and borrowed allocations. */
void launch_independent_jk_derivative_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t coordinates_per_item, std::size_t system_begin, double cj, double ck,
    bool unrestricted, DirectCoulombRange exchange_range, double exchange_omega, double screening,
    const double* bounds, const double* density, const double* beta, double* out);

/** Fuse Coulomb and split-range exchange derivatives in one quartet traversal.
 * Output is source-major [J, short-range K, long-range K].
 */
void launch_independent_rsh_derivative_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t coordinates_per_item, std::size_t system_begin, std::size_t source_stride,
    double cj, double short_ck, double long_ck, bool unrestricted, double omega, double screening,
    const double* bounds, const double* density, const double* beta, double* out);

/** Provider-facing shell derivative seam. Queue/numerical ownership remains in
 * the Direct consumer layer; host source owners borrow only this launch ABI. */
void launch_bounded_shell_energy_derivative(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    double coulomb_coefficient, double exchange_coefficient,
    DirectCoulombRange radial_range = DirectCoulombRange::Full, double radial_omega = 0.0);

}  // namespace generativeqc::scf::cuda_execution
