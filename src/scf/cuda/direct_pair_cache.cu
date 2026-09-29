#include "generated_direct_pair_cache.cuh"
#include "scf/cuda/direct_pair_cache.hpp"

namespace generativeqc::scf::cuda_execution {

void launch_build_shell_primitive_pair_cache_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                    cudaStream_t stream, DeviceBatch batch,
                                                    PrimitivePairData* shell_primitive_pairs) {
  build_shell_primitive_pair_cache_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, shell_primitive_pairs);
}

}  // namespace generativeqc::scf::cuda_execution
