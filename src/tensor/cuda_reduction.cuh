#pragma once

// CuMetal does not provide CUB. Callers use their original source-major serial
// reduction when this capability is absent; do not substitute a new tree or
// weaken the provider's strict-FP64 arithmetic contract.
#if defined(GENERATIVEQC_CUDA_PROVIDER_CUMETAL) && GENERATIVEQC_CUDA_PROVIDER_CUMETAL
#define GENERATIVEQC_TENSOR_HAS_STRICT_FP64_BLOCK_REDUCE 0
#else
#define GENERATIVEQC_TENSOR_HAS_STRICT_FP64_BLOCK_REDUCE 1

#include <cub/block/block_reduce.cuh>

namespace generativeqc::tensor {

struct StrictFp64Add {
  __device__ __forceinline__ double operator()(double a, double b) const { return __dadd_rn(a, b); }
};

template <int Threads>
struct StrictFp64BlockReduce {
  using Implementation = cub::BlockReduce<double, Threads, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
  using TempStorage = typename Implementation::TempStorage;

  __device__ __forceinline__ static double sum(double value, TempStorage& storage) {
    return Implementation(storage).Reduce(value, StrictFp64Add{});
  }
};

}  // namespace generativeqc::tensor

#endif
