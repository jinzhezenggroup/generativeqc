#pragma once

#include <cub/block/block_reduce.cuh>

namespace generativeqc::tensor {

struct StrictFp64Add {
  __device__ __forceinline__ double operator()(double a, double b) const {
    return __dadd_rn(a, b);
  }
};

template <int Threads>
struct StrictFp64BlockReduce {
  using Implementation =
      cub::BlockReduce<double, Threads, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
  using TempStorage = typename Implementation::TempStorage;

  __device__ __forceinline__ static double sum(double value, TempStorage& storage) {
    return Implementation(storage).Reduce(value, StrictFp64Add{});
  }
};

}  // namespace generativeqc::tensor
