"""Build-time CUDA schedules for generic restarted right-preconditioned GMRES.

This is generic FP64 linear algebra, not a second scientific derivative algebra.
The native response owner supplies the operator, storage, stream, finite limits
and checkpoint controller. Every iterative numerical scalar stays on the device.
No method, occupation, density, XC or integral policy belongs in this schedule.
"""


def emit_cuda_gmres() -> str:
    """Emit deterministic kernels; importing/generating needs no CUDA runtime.

    One block owns each reduction, so no atomics reorder floating reductions.
    Norms use a scaled sum of squares rather than dot(x,x), including for tiny
    RHSs. Two-pass MGS is selectable; every candidate gets a true residual test,
    so neither a projected residual nor an Arnoldi breakdown publishes a result.
    """
    return r"""#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "response/cuda_gmres.hpp"

namespace generativeqc::response::cuda_gmres_detail {
using Status = CudaGmresStatus;
constexpr unsigned threads = 256;
struct State {
  int code;
  int breakdown;
  double rhs_norm;
  double residual_norm;
  double tolerance;
  double beta;
  double norm_before;
  double norm_after;
  double projection;
};

__device__ void fail(State* state, Status status) {
  atomicCAS(&state->code, 0, static_cast<int>(status));
}

__device__ int status_code(State* state) {
  // Peer lanes/blocks can report failure while a vector kernel is entering.
  // Only the flag is polled here; stream/barrier ordering owns other state.
  return atomicCAS(&state->code, 0, 0);
}

__global__ void validate(const double* rhs, const double* diagonal, std::size_t count,
                         double floor, State* state) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += std::size_t(blockDim.x) * gridDim.x) {
    if (!isfinite(rhs[index])) fail(state, Status::nonfinite_input);
    if (diagonal && (!isfinite(diagonal[index]) || fabs(diagonal[index]) <= floor))
      fail(state, Status::invalid_preconditioner);
  }
}

__device__ void lassq(double value, double& scale, double& sum) {
  const double magnitude = fabs(value);
  if (magnitude == 0.0) return;
  if (scale < magnitude) {
    const double ratio = scale / magnitude;
    sum = 1.0 + sum * ratio * ratio;
    scale = magnitude;
  } else {
    const double ratio = magnitude / scale;
    sum += ratio * ratio;
  }
}

__global__ void norm(const double* values, std::size_t count, double* output,
                     State* state, Status failure) {
  __shared__ double scales[threads], sums[threads];
  __shared__ int active;
  if (!threadIdx.x) active = state->code == 0;
  __syncthreads();
  if (!active) return;
  double scale = 0.0, sum = 0.0;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    const double value = values[index];
    if (!isfinite(value)) fail(state, failure);
    else lassq(value, scale, sum);
  }
  scales[threadIdx.x] = scale;
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = threads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) {
      const double other = scales[threadIdx.x + stride];
      if (other != 0.0) {
        if (scales[threadIdx.x] < other) {
          const double ratio = scales[threadIdx.x] / other;
          sums[threadIdx.x] = sums[threadIdx.x + stride] + sums[threadIdx.x] * ratio * ratio;
          scales[threadIdx.x] = other;
        } else {
          const double ratio = other / scales[threadIdx.x];
          sums[threadIdx.x] += sums[threadIdx.x + stride] * ratio * ratio;
        }
      }
    }
    __syncthreads();
  }
  if (!threadIdx.x && !state->code) {
    const double result = scales[0] * sqrt(sums[0]);
    if (!isfinite(result)) fail(state, failure);
    else *output = result;
  }
}

__global__ void initial(State* state, double relative, double absolute) {
  if (state->code) return;
  state->residual_norm = state->rhs_norm;
  state->beta = state->rhs_norm;
  state->tolerance = fmax(absolute, relative * state->rhs_norm);
  if (state->rhs_norm <= state->tolerance) state->code = static_cast<int>(Status::converged);
}

__global__ void cycle(State* state, double* target) {
  if (state->code) return;
  state->beta = state->residual_norm;
  target[0] = state->beta;
}

__global__ void normalize(const double* input, double* output, std::size_t count,
                          const double* divisor, State* state, bool arnoldi) {
  if (status_code(state) || (arnoldi && state->breakdown)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += std::size_t(blockDim.x) * gridDim.x) {
    const double value = input[index] / *divisor;
    if (!isfinite(value)) fail(state, Status::numerical_failure);
    else output[index] = value;
  }
}

__global__ void precondition(const double* input, const double* diagonal, double* output,
                             std::size_t count, State* state) {
  if (status_code(state)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += std::size_t(blockDim.x) * gridDim.x) {
    const double value = diagonal ? input[index] / diagonal[index] : input[index];
    if (!isfinite(value)) fail(state, Status::invalid_preconditioner);
    else output[index] = value;
  }
}

__global__ void dot(const double* left, const double* right, std::size_t count, State* state) {
  __shared__ double partial[threads];
  __shared__ int active;
  if (!threadIdx.x) active = state->code == 0;
  __syncthreads();
  if (!active) return;
  double sum = 0.0, correction = 0.0;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    const double product = left[index] * right[index];
    const double adjusted = product - correction;
    const double next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = threads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) {
    if (!isfinite(partial[0])) fail(state, Status::numerical_failure);
    else state->projection = partial[0];
  }
}

__global__ void orthogonalize(const double* basis, double* work, std::size_t count,
                              State* state) {
  if (status_code(state)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += std::size_t(blockDim.x) * gridDim.x) {
    const double value = work[index] - state->projection * basis[index];
    if (!isfinite(value)) fail(state, Status::numerical_failure);
    else work[index] = value;
  }
}

__global__ void accumulate(double* entry, State* state) {
  if (state->code) return;
  const double value = *entry + state->projection;
  if (!isfinite(value)) fail(state, Status::numerical_failure);
  else *entry = value;
}

__global__ void arnoldi(double* entry, State* state, double relative) {
  if (state->code) return;
  *entry = state->norm_after;
  state->breakdown = state->norm_after <= relative * state->norm_before;
}

__global__ void rotate(double* hessenberg, double* cosine, double* sine, double* target,
                       double* coefficients, std::size_t leading, std::size_t column,
                       State* state) {
  if (state->code) return;
  double* current = hessenberg + column * leading;
  for (std::size_t row = 0; row < column; ++row) {
    const double upper = cosine[row] * current[row] + sine[row] * current[row + 1];
    current[row + 1] = -sine[row] * current[row] + cosine[row] * current[row + 1];
    current[row] = upper;
  }
  const double radius = hypot(current[column], current[column + 1]);
  if (!isfinite(radius)) { fail(state, Status::numerical_failure); return; }
  if (radius == 0.0) { fail(state, Status::breakdown); return; }
  cosine[column] = current[column] / radius;
  sine[column] = current[column + 1] / radius;
  current[column] = radius;
  current[column + 1] = 0.0;
  target[column + 1] = -sine[column] * target[column];
  target[column] *= cosine[column];
  for (std::size_t reverse = column + 1; reverse; --reverse) {
    const std::size_t row = reverse - 1;
    double value = target[row];
    for (std::size_t index = row + 1; index <= column; ++index)
      value -= hessenberg[index * leading + row] * coefficients[index];
    const double divisor = hessenberg[row * leading + row];
    if (divisor == 0.0) { fail(state, Status::breakdown); return; }
    value /= divisor;
    if (!isfinite(value)) { fail(state, Status::numerical_failure); return; }
    coefficients[row] = value;
  }
}

__global__ void candidate(const double* base, const double* directions,
                          const double* coefficients, double* output, std::size_t count,
                          std::size_t columns, State* state) {
  if (status_code(state)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += std::size_t(blockDim.x) * gridDim.x) {
    double value = base[index];
    for (std::size_t column = 0; column < columns; ++column)
      value += directions[column * count + index] * coefficients[column];
    if (!isfinite(value)) fail(state, Status::numerical_failure);
    else output[index] = value;
  }
}

__global__ void residual(const double* rhs, const double* image, double* output,
                         std::size_t count, State* state) {
  if (status_code(state)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += std::size_t(blockDim.x) * gridDim.x) {
    const double value = rhs[index] - image[index];
    if (!isfinite(value)) fail(state, Status::nonfinite_operator);
    else output[index] = value;
  }
}

__global__ void accept(State* state) {
  if (state->code) return;
  if (state->residual_norm <= state->tolerance)
    state->code = static_cast<int>(Status::converged);
  else if (state->breakdown) state->code = static_cast<int>(Status::breakdown);
}
}  // namespace generativeqc::response::cuda_gmres_detail
"""
