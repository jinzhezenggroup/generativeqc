#ifndef GENERATIVEQC_SCF_SHARED_RYS_WEIGHTED_Z_HPP
#define GENERATIVEQC_SCF_SHARED_RYS_WEIGHTED_Z_HPP

#include <cuda_runtime_api.h>

#include <cstdint>

#include "runtime/compensated_output.hpp"

namespace generativeqc::scf::generated {

/** Private additive companion; availability is independent of shared XY. */
std::uint64_t enabled_shared_rys_weighted_z_fock_shell_class_mask() noexcept;
cudaError_t launch_shell_class_shared_rys_weighted_z_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted, unsigned worker_blocks,
    const void* tasks, const std::uint32_t* task_offset, const std::int64_t* primitive_pair_offsets,
    const void* primitive_pairs, const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    runtime::CompensatedOutput output, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

}  // namespace generativeqc::scf::generated

#endif
