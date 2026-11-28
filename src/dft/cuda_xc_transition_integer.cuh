#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace generativeqc::dft::transition_integer {

constexpr int digits = 8;
constexpr int block_width = 512;

struct Workspace {
  int* product{};
  int* exponent{};
  signed char *left{}, *right{}, *packed_left{}, *packed_right{};
  bool disabled{};
};

inline unsigned blocks(std::size_t elements) { return (elements + 255) / 256; }

/** Convert one rotation column per block using a separate binary exponent.
 * Keeping the normalized magnitude below one half bounds every nearest
 * radix-128 digit by 64, including the signed remainder digits. */
static __global__ void split(const double* input, signed char* output, int* exponents,
                             int rotations, int rank, int* status) {
  const int column = blockIdx.x;
  const auto extent = std::size_t(rotations) * rank;
  const auto start = std::size_t(column) * rank;
  __shared__ double maxima[8];
  __shared__ double normalization;
  double maximum{};
  for (int row = threadIdx.x; row < rank; row += blockDim.x)
    maximum = fmax(maximum, fabs(input[start + row]));
  for (int step = 16; step > 0; step /= 2)
    maximum = fmax(maximum, __shfl_down_sync(0xffffffff, maximum, step));
  if (threadIdx.x % 32 == 0) maxima[threadIdx.x / 32] = maximum;
  __syncthreads();
  if (threadIdx.x == 0) {
    maximum = 0;
    for (double value : maxima) maximum = fmax(maximum, value);
    int exponent{};
    (void)frexp(maximum, &exponent);
    if (exponent > 300) atomicExch(status, 1);
    exponent = max(-300, min(300, exponent + 1));
    exponents[column] = exponent;
    normalization = ldexp(1.0, -exponent);
  }
  __syncthreads();
  for (int row = threadIdx.x; row < rank; row += blockDim.x) {
    double residual = input[start + row] * normalization;
    if (!(fabs(residual) <= 0.5)) {
      atomicExch(status, 1);
      residual = 0;
    }
    for (int plane = 0; plane < digits; ++plane) {
      residual *= 128;
      const int digit = __double2int_rn(residual);
      output[plane * extent + start + row] = static_cast<signed char>(digit);
      residual -= digit;
    }
  }
}

/** Concatenate digit pairs of the same binary weight along the GEMM rank.
 * Each caller bounds digits * rank * 64^2 before submitting the INT32 product. */
static __global__ void pack(const signed char* left, const signed char* right,
                            signed char* packed_left, signed char* packed_right, int rotations,
                            int rank, int diagonal) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride = (diagonal + 1) * rank;
  if (index >= std::size_t(rotations) * stride) return;
  const auto column = index / stride, point = index % rank;
  const auto digit = (index % stride) / rank;
  packed_left[index] = left[((digit * rotations + column) * rank) + point];
  packed_right[index] = right[(((diagonal - digit) * rotations + column) * rank) + point];
}

/** Restore the original powers of two and the common RKS rotation lift 4.
 * The final FP64 CPKS physical residual uses none of this representation. */
static __global__ void accumulate(const int* integers, double* output, const int* left_exp,
                                  const int* right_exp, int order, int height, int width,
                                  int diagonal, bool first) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= std::size_t(height) * width) return;
  const auto row = index % height, column = index / height;
  const auto target = column * order + row;
  const int exponent = left_exp[row] + right_exp[column] - 7 * (diagonal + 2) + 2;
  // The admitted exponent range fits normal binary64 for this exact scale.
  const auto scale = __longlong_as_double(static_cast<long long>(1023 + exponent) << 52);
  output[target] = __fma_rn(scale, integers[index], first ? 0.0 : output[target]);
}

template <class Submit>
cudaError_t triangle(int first, int count, const Submit& submit) {
  if (count <= block_width) return submit(first, first, count, count);
  const auto left = ((count / 2 + block_width / 2) / block_width) * block_width;
  const auto right = count - left;
  auto error = submit(first + left, first, right, left);
  if (error == cudaSuccess) error = triangle(first, left, submit);
  if (error == cudaSuccess) error = triangle(first + left, right, submit);
  return error;
}

static inline cudaError_t project(cublasHandle_t blas, const Workspace& workspace,
                                  const double* fields, const double* tangents, double* matrix,
                                  int order, int rank, bool initialize, int* status,
                                  cudaStream_t stream) {
  split<<<order, 256, 0, stream>>>(fields, workspace.left, workspace.exponent, order, rank, status);
  split<<<order, 256, 0, stream>>>(tangents, workspace.right, workspace.exponent + order, order,
                                   rank, status);
  auto error = cudaGetLastError();
  const int one = 1, zero = 0;
  for (int diagonal = 0; diagonal < digits && error == cudaSuccess; ++diagonal) {
    const int reduction = (diagonal + 1) * rank;
    pack<<<blocks(std::size_t(order) * reduction), 256, 0, stream>>>(
        workspace.left, workspace.right, workspace.packed_left, workspace.packed_right, order, rank,
        diagonal);
    if ((error = cudaGetLastError()) != cudaSuccess) break;
    error = triangle(0, order, [&](int row, int column, int height, int width) {
      if (cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, height, width, reduction, &one,
                       workspace.packed_left + std::size_t(row) * reduction, CUDA_R_8I, reduction,
                       workspace.packed_right + std::size_t(column) * reduction, CUDA_R_8I,
                       reduction, &zero, workspace.product, CUDA_R_32I, height, CUBLAS_COMPUTE_32I,
                       CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
        return cudaErrorNotSupported;
      accumulate<<<blocks(std::size_t(height) * width), 256, 0, stream>>>(
          workspace.product, matrix + std::size_t(column) * order + row, workspace.exponent + row,
          workspace.exponent + order + column, order, height, width, diagonal,
          initialize && diagonal == 0);
      return cudaGetLastError();
    });
  }
  return error;
}

}  // namespace generativeqc::dft::transition_integer
