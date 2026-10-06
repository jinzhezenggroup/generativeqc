#pragma once

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

/** Compiler-owned exact J inventory and fixed Hermite stride; no GPU probe. */
std::uint64_t direct_md_coulomb_shell_class_mask() noexcept;
unsigned direct_md_coulomb_hermite_width() noexcept;

/** Pair transforms borrow prepared geometry and resident total-spin density.
 * Both Hermite arrays contain width*primitive_pair_count FP64 values. Calls
 * are ordered on the owner's stream; projection adds to existing exact J. */
cudaError_t prepare_direct_md_coulomb_density(cudaStream_t stream, DeviceBatch batch,
                                              const double* density, double* density_hermite,
                                              double* potential_hermite,
                                              const std::uint8_t* active);
cudaError_t project_direct_md_coulomb(cudaStream_t stream, DeviceBatch batch,
                                      double* potential_hermite, double* coulomb,
                                      const std::uint8_t* active);

/** Execute one admitted class without materializing shell-quartet products.
 * work_count optionally records admitted shell quartets, never primitive work.
 * Unsupported classes return NotSupported before any output mutation. */
cudaError_t enqueue_direct_md_coulomb_class(unsigned shell_class, unsigned workers,
                                            cudaStream_t stream,
                                            const GeneratedShellPairStream* topology,
                                            DeviceBatch batch, const double* density_hermite,
                                            double* potential_hermite, double screening,
                                            std::uint32_t* head, unsigned long long* work_count);

}  // namespace generativeqc::scf::cuda_execution
