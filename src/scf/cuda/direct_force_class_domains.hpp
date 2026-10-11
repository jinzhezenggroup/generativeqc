#pragma once

#include <cuda_runtime.h>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/direct_block_domain.hpp"

namespace generativeqc::scf::cuda_execution {

/** Replace only the incumbent's order-zero-through-six force drain.
 * Orders zero through five keep their scalar consumers; order six retains
 * its Cartesian warp algebra with all tiles/subtiles. The caller owns cursor
 * initialization and still submits the existing seven/eight consumers.
 * Spatial reuse caches at most 64 primitive-local double values, not quartets. */
cudaError_t launch_direct_force_spatial_spd_low(
    bool unrestricted, dim3 grid, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient, double exchange_coefficient,
    bool separate_sources, detail::BoundedDirectBlockDomain block_domain);

/** Execute only the proved s/p/d subdomains of a mixed f basis.
 * The caller supplies compatible resident primitive pairs, both retained
 * derivative schedules and this plan's immutable class-major topology. The
 * residual bounded consumer owns every f-containing class. This owner resets
 * the borrowed cursor between domains and propagates every submission error. */
cudaError_t launch_direct_force_class_domains(
    bool unrestricted, dim3 grid, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* topology, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* generated_overflow, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DeviceShellClassProfileEntry* profile, double coulomb_coefficient, double exchange_coefficient,
    bool separate_sources);

}  // namespace generativeqc::scf::cuda_execution
