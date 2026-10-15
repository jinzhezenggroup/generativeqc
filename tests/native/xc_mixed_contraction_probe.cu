#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

using I = std::int64_t;

__device__ double finite(double value, int* error, int code) {
  if (!::isfinite(value)) atomicOr(error, code);
  return value;
}

#include "generated_xc_mixed_contractions.cuh"

template <class Arithmetic>
__global__ void density_probe(const double* density, const double* ao, double* output, I count,
                              I active, I full, I jets, const size_t* ids, int* error) {
  tiled_density_product_body<false, true, Arithmetic>(density, ao, active, count, jets, output,
                                                      error, ids, full, blockIdx.z);
}

template <class Arithmetic>
__global__ void potential_probe(const double* ao, const double* work, double* output, I count,
                                I active, I full, I jets, const size_t* ids, int* error) {
  tiled_potential_body<true, Arithmetic>(ao, work, active, count, jets, nullptr, output, nullptr,
                                         true, error, ids, full, 0);
}

/** Test the actual generated scientific bodies with no production library.
 * Independent host contractions supply the oracle. Error codes distinguish
 * arithmetic-domain rejection from CUDA setup/execution failure. */
extern "C" int mixed_contractions(const double* density, const double* ao, const double* work,
                                  double* products, double* potential, I count, I active, I full,
                                  I jets, const size_t* ids, int mixed) {
  const auto panel = count * active * jets;
  const auto matrix = full * full;
  double* allocation{};
  size_t* device_ids{};
  int* error{};
  int status = -1;
  if (cudaMalloc(&allocation, (2 * matrix + 3 * panel) * sizeof(double)) != cudaSuccess)
    return status;
  auto* device_ao = allocation + matrix;
  auto* device_work = device_ao + panel;
  auto* device_products = device_work + panel;
  auto* device_potential = device_products + panel;
  const bool setup =
      cudaMalloc(&device_ids, active * sizeof(size_t)) == cudaSuccess &&
      cudaMalloc(&error, sizeof(int)) == cudaSuccess &&
      cudaMemset(error, 0, sizeof(int)) == cudaSuccess &&
      cudaMemcpy(allocation, density, matrix * sizeof(double), cudaMemcpyHostToDevice) ==
          cudaSuccess &&
      cudaMemcpy(device_ao, ao, panel * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess &&
      cudaMemcpy(device_work, work, panel * sizeof(double), cudaMemcpyHostToDevice) ==
          cudaSuccess &&
      cudaMemcpy(device_potential, potential, matrix * sizeof(double), cudaMemcpyHostToDevice) ==
          cudaSuccess &&
      cudaMemcpy(device_ids, ids, active * sizeof(size_t), cudaMemcpyHostToDevice) == cudaSuccess;
  if (setup) {
    const dim3 density_grid((active + 15) / 16, (count + 15) / 16, jets);
    const I columns = (active + 15) / 16;
    const dim3 potential_grid(columns * (columns + 1) / 2);
    if (mixed) {
      density_probe<float><<<density_grid, dim3(16, 16)>>>(
          allocation, device_ao, device_products, count, active, full, jets, device_ids, error);
      potential_probe<float><<<potential_grid, dim3(16, 16)>>>(
          device_ao, device_work, device_potential, count, active, full, jets, device_ids, error);
    } else {
      density_probe<double><<<density_grid, dim3(16, 16)>>>(
          allocation, device_ao, device_products, count, active, full, jets, device_ids, error);
      potential_probe<double><<<potential_grid, dim3(16, 16)>>>(
          device_ao, device_work, device_potential, count, active, full, jets, device_ids, error);
    }
    if (cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(products, device_products, panel * sizeof(double), cudaMemcpyDeviceToHost) ==
            cudaSuccess &&
        cudaMemcpy(potential, device_potential, matrix * sizeof(double), cudaMemcpyDeviceToHost) ==
            cudaSuccess)
      if (cudaMemcpy(&status, error, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        status = -1;
  }
  if (cudaFree(allocation) != cudaSuccess) status = -1;
  if (cudaFree(device_ids) != cudaSuccess) status = -1;
  if (cudaFree(error) != cudaSuccess) status = -1;
  return status;
}
