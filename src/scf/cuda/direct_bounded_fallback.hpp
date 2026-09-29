#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

enum class DirectRangeOperator : std::uint32_t { Full = 0, Long = 1, Short = 2 };

/** Force-output fallback; purpose selects screening semantics, not the scientific output. */
/** Method-neutral force variant. Coefficients multiply the Coulomb and exchange
 * density contractions without changing topology, screening, or recurrence. */
void launch_bounded_direct_shell_quartet_kernel_scaled(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile,
    double coulomb_coefficient, double exchange_coefficient);

/** Range-separated exchange derivative on the same bounded shell scheduler.
 * Full-range Schwarz bounds remain a conservative gate for SR/LR operators. */
void launch_bounded_direct_range_exchange_force_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DirectRangeOperator radial_operator, double omega,
    double exchange_coefficient);

void launch_bounded_direct_shell_quartet_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile);

/** Consume only Fock registry gaps through the bounded hierarchical dispatcher. */
void launch_bounded_direct_fock_shell_quartet_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock,
    unsigned long long* global_cursor);

}  // namespace generativeqc::scf::cuda_execution
