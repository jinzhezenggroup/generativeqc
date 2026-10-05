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
                                        int count, int inserted, double* gram,
                                        const unsigned char* metric_weights = nullptr) {
  if (blockIdx.x >= static_cast<unsigned>(count)) return;
  const int row = history_row(first, static_cast<int>(blockIdx.x), capacity);
  __shared__ double partial[256];
  double sum = 0.0;
  for (I i = threadIdx.x; i < elements; i += 256) {
    auto product = __dmul_rn(errors[I(row) * elements + i], errors[I(inserted) * elements + i]);
    if (metric_weights) product = __dmul_rn(product, double(metric_weights[i]));
    sum = __dadd_rn(sum, product);
  }
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

/** Insert a singleton prefix and generic size-one/two permutation orbits.
 * The compiler-owned map supplies coordinates, weights and projection. Read
 * both orbit members before admitting rounding-only asymmetry; never silently
 * assume an arbitrary dense vector lies in the symmetric subspace. Scratch is
 * one byte per stored coordinate plus caller-owned scalar audit state.
 */
template <class OrbitMap>
static __global__ void history_insert_orbits(const double* singles, const double* doubles,
                                             const double* singles_error,
                                             const double* doubles_error, I n1, I n2,
                                             double* vector, double* error, unsigned char* weights,
                                             OrbitMap map, double tolerance, int* refused,
                                             double* maximum_asymmetry, int* arithmetic_error) {
  __shared__ double block_maximum[256];
  double local_maximum = 0.;
  for (I flat = I(blockIdx.x) * blockDim.x + threadIdx.x; flat < n1 + n2;
       flat += I(blockDim.x) * gridDim.x) {
    if (flat < n1) {
      vector[flat] = finite(singles[flat], arithmetic_error, 0);
      error[flat] = finite(singles_error[flat], arithmetic_error, 0);
      weights[flat] = 1;
      continue;
    }
    const auto k = flat - n1;
    const auto orbit = map(static_cast<std::size_t>(k));
    if (static_cast<std::size_t>(k) > orbit.partner) continue;
    const auto destination = n1 + static_cast<I>(orbit.slot);
    const double a = doubles[k], b = doubles[orbit.partner];
    const double r = doubles_error[k], s = doubles_error[orbit.partner];
    const double delta = fmax(fabs(a - b), fabs(r - s));
    local_maximum = fmax(local_maximum, delta);
    if (!isfinite(delta) || fabs(a - b) > tolerance * (1 + fmax(fabs(a), fabs(b))) ||
        fabs(r - s) > tolerance * (1 + fmax(fabs(r), fabs(s))))
      atomicExch(refused, 1);
    vector[destination] = finite(map.project(a, b), arithmetic_error, 0);
    error[destination] = finite(map.project(r, s), arithmetic_error, 0);
    weights[destination] = orbit.weight;
  }
  block_maximum[threadIdx.x] = local_maximum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      block_maximum[threadIdx.x] =
          fmax(block_maximum[threadIdx.x], block_maximum[threadIdx.x + stride]);
    __syncthreads();
  }
  if (!threadIdx.x)
    atomicMax(reinterpret_cast<unsigned long long*>(maximum_asymmetry),
              static_cast<unsigned long long>(__double_as_longlong(block_maximum[0])));
}

/** Combine an orbit-stored history directly into a full dense destination.
 * No complete unpacked history or conversion tensor is materialized. Partner
 * coordinates use the identical chronological sum, restoring bitwise symmetry.
 */
template <class OrbitMap>
static __global__ void diis_combine_orbits(const double* vectors, const double* coefficients,
                                           I stride, I offset, I full_elements, int count,
                                           const int* status, double* result, int* arithmetic_error,
                                           int capacity, int first, OrbitMap map) {
  if (*status) return;
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < full_elements;
       i += I(blockDim.x) * gridDim.x) {
    const auto column = offset + static_cast<I>(map(static_cast<std::size_t>(i)).slot);
    double value = 0.;
    for (int logical = 0; logical < count; ++logical) {
      const auto row = history_row(first, logical, capacity);
      value = __dadd_rn(value, __dmul_rn(coefficients[logical], vectors[I(row) * stride + column]));
    }
    result[i] = finite(value, arithmetic_error, 0);
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
