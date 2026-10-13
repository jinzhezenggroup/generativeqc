#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

/** Consume one complete shell class using a CTA-shared primitive recurrence.
 * Borrows the bounded class-major topology and existing pair cache; no quartet
 * arena or resident ERIs are required. Orders 3--12, full-range FP64 J/K only.
 * The caller excludes precisely this class from the generic fallback.
 */
cudaError_t launch_bounded_materialized_fock(
    bool unrestricted, unsigned shell_class, cudaStream_t stream, unsigned worker_blocks,
    DeviceBatch batch, const GeneratedShellPairStream* topology, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    unsigned long long* work_count = nullptr);

/** Resolve host spin/precision while retaining compile-time angular dispatch. */
void dispatch_angular_fock_quartets(
    bool unrestricted, bool mixed_precision, cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask);

}  // namespace generativeqc::scf::cuda_execution
