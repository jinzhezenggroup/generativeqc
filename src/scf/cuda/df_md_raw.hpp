#pragma once

#include "scf/cuda/df_md_owner.hpp"

namespace generativeqc::scf::cuda_execution {
struct CudaDfMdRaw;

/** Optional immutable FP64 E transforms for packed raw values. Preparation
 * owns and drains its private stream. Replay borrows the caller's stream and
 * output; freeing the synchronous CUDA allocations retires outstanding reads.
 * The caller must charge the complete reported storage to its enclosing cap. */
generativeqc_status create_cuda_df_md_raw(int device, const core::System& orbital,
                                          const core::System& auxiliary, std::size_t device_budget,
                                          std::size_t host_budget, CudaDfMdRaw** owner,
                                          CudaDfMdResources& resources, std::string& detail);
void destroy_cuda_df_md_raw(CudaDfMdRaw* owner) noexcept;

/** Overwrite a contiguous packed-lower pair / public-auxiliary rectangle.
 * No screening, density, metric, allocation, atomics or pair multiplicity.
 * Each output sums all ordered primitive products exactly once. */
cudaError_t cuda_df_md_raw_tile(const CudaDfMdRaw* owner, cudaStream_t stream,
                                std::size_t pair_begin, std::size_t pair_count,
                                std::size_t aux_begin, std::size_t aux_count, double* output);
}  // namespace generativeqc::scf::cuda_execution
