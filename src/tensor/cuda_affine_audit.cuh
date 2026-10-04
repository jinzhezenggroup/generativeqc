#pragma once

#include "tensor/cuda_runtime.cuh"
#include "tensor/native_contraction.hpp"

namespace generativeqc::tensor {

// Plain arrays keep the device audit independent of host std::array methods.
struct AffineAuditView {
  std::size_t rank{};
  std::size_t shape[ContractionOperand::kMaximumRank]{};
  std::size_t strides[ContractionOperand::kMaximumRank]{};
};

/** Audit logical output elements only; padding is borrowed storage, not data.
 * atomicCAS preserves an earlier error from any operation on this stream. */
template <class T>
static __global__ void audit_affine_contraction(const T* output, AffineAuditView view,
                                                std::size_t count, int* error) {
  for (std::size_t flat = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; flat < count;
       flat += std::size_t(blockDim.x) * gridDim.x) {
    std::size_t logical = flat, offset = 0;
    for (auto axis = view.rank; axis != 0; --axis) {
      offset += logical % view.shape[axis - 1] * view.strides[axis - 1];
      logical /= view.shape[axis - 1];
    }
    if (!isfinite(output[offset])) atomicCAS(error, 0, 1);
  }
}

}  // namespace generativeqc::tensor
