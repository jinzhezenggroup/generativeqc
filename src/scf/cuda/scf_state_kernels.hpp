#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace generativeqc::scf::cuda_execution {

/** Count completed physical Fock work under its pre-convergence active mask.
 * Captured calls count only when executed. The owner resets once per solve. */
void launch_count_fock_application_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          const std::uint8_t* active, std::uint64_t* builds);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_initialize_state_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    bool reuse_previous_energy, const double* energy, std::uint8_t* active, std::uint8_t* converged,
    std::uint8_t* failed, std::uint32_t* iterations, double* previous_energy, double* energy_change,
    double* density_rms, std::uint32_t* diis_count, std::uint32_t* diis_head);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_inspect_solver_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, const int* info,
                                  std::uint8_t* active, std::uint8_t* failed,
                                  std::uint8_t* converged);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_expand_spin_active_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::int32_t batch_size,
                                      std::int32_t spin_count, const std::uint8_t* active,
                                      std::uint8_t* spin_active);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_inspect_spin_solver_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                       cudaStream_t stream, std::int32_t batch_size,
                                       std::int32_t spin_count, const int* info,
                                       std::uint8_t* active, std::uint8_t* failed,
                                       std::uint8_t* converged);

}  // namespace generativeqc::scf::cuda_execution
