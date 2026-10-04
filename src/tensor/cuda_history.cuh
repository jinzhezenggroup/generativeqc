#pragma once

#include "cuda_runtime.cuh"

namespace generativeqc_tensor {

/** Physical row of a bounded chronological window; all extents are preflighted. */
__device__ inline int history_row(int first, int logical, int capacity) {
  const int row = first + logical;
  return row < capacity ? row : row - capacity;
}

/** Update only a newly inserted row/column of a resident physical Gram matrix.
 *
 * One 256-thread block owns each live dot. The per-thread serial order, explicit
 * FP64 multiply/add rounding and binary reduction tree match the full-Gram
 * fallback. No old-old dot is evaluated, and unused/retired slots are unread.
 * The caller queues this after insertion on the same stream. No workspace,
 * atomics, provider discovery or history-size tensor movement is required.
 */
static __global__ void history_gram_row(const double* errors, I elements, int capacity, int first,
                                        int count, int inserted, double* gram) {
  if (blockIdx.x >= static_cast<unsigned>(count)) return;
  const int row = history_row(first, static_cast<int>(blockIdx.x), capacity);
  __shared__ double partial[256];
  double sum = 0.0;
  for (I i = threadIdx.x; i < elements; i += 256)
    sum = __dadd_rn(sum,
                    __dmul_rn(errors[I(row) * elements + i], errors[I(inserted) * elements + i]));
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] = __dadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
    __syncthreads();
  }
  if (!threadIdx.x) {
    gram[I(row) * capacity + inserted] = partial[0];
    if (row != inserted) gram[I(inserted) * capacity + row] = partial[0];
  }
}

/** Strict-order weighted sum over a logical slice of chronological history rows.
 *
 * The operation is provider-neutral: physical ring layout and explicit ordered
 * FP64 arithmetic select this bounded tensor reduction, not a vendor
 * choice in the scientific owner. Coefficients are chronological; vectors are
 * physical. A failed small solve leaves the destination untouched.
 */
static __global__ void diis_combine_slice(const double* vectors, const double* coefficients,
                                          I stride, I offset, I elements, int count,
                                          const int* status, double* result, int* arithmetic_error,
                                          int capacity = 0, int first = 0) {
  if (*status) return;
  if (!capacity) capacity = count;
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
       i += I(blockDim.x) * gridDim.x) {
    double value = 0.0;
    for (int logical = 0; logical < count; ++logical) {
      const int row = history_row(first, logical, capacity);
      value =
          __dadd_rn(value, __dmul_rn(coefficients[logical], vectors[I(row) * stride + offset + i]));
    }
    result[i] = finite(value, arithmetic_error, 0);
  }
}

}  // namespace generativeqc_tensor
