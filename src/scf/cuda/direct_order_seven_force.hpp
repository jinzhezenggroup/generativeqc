#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

/** Borrow the same plan's immutable class-major topology for the order-seven
 * cooperative consumer. The caller proves max-L=2, resident primitive pairs,
 * compatible recurrence and a 256-lane schedule; no device inventory is copied
 * back to the host. Density bounds must belong to the force owner, not to the
 * Coulomb owner stored in topology. Cursor resets remain caller-owned. */
cudaError_t launch_direct_order_seven_force(
    bool unrestricted, dim3 grid, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* topology, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient, double exchange_coefficient,
    bool separate_sources);

}  // namespace generativeqc::scf::cuda_execution
