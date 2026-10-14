#include <cuda_runtime.h>

#include "dft/xc_point_mixed.hpp"

__global__ void point_probe_kernel(const double* inputs, double* output, int mixed, int* status) {
  double rho[2]{inputs[0], inputs[4]};
  double gradient[2][3]{{inputs[1], inputs[2], inputs[3]}, {inputs[5], inputs[6], inputs[7]}};
  const auto candidate = mixed == 2 ? generativeqc::dft::point::evaluate_pbe_full_mixed(
                                          true, rho, gradient, inputs[8], inputs[9])
                                    : generativeqc::dft::point::evaluate_pbe_restricted_mixed(
                                          rho, gradient, inputs[8], inputs[9]);
  const auto value =
      mixed ? candidate.value
            : generativeqc::dft::point::evaluate(true, rho, gradient, inputs[8], inputs[9]);
  output[0] = value.energy;
  for (unsigned spin = 0; spin < 2; ++spin) {
    output[1 + spin * 4] = value.rho[spin];
    for (unsigned axis = 0; axis < 3; ++axis)
      output[2 + spin * 4 + axis] = value.gradient[spin][axis];
  }
  *status = !value.valid ? -1 : mixed && candidate.fp32_correlation ? 1 : 0;
}

extern "C" int mixed_point(const double* inputs, double* output, int mixed) {
  double* device_values{};
  int* device_status{};
  int status = -2;
  if (cudaMalloc(&device_values, 19 * sizeof(double)) != cudaSuccess) return status;
  if (cudaMalloc(&device_status, sizeof(int)) != cudaSuccess) {
    cudaFree(device_values);
    return status;
  }
  if (cudaMemcpy(device_values, inputs, 10 * sizeof(double), cudaMemcpyHostToDevice) ==
      cudaSuccess) {
    point_probe_kernel<<<1, 1>>>(device_values, device_values + 10, mixed, device_status);
    if (cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(output, device_values + 10, 9 * sizeof(double), cudaMemcpyDeviceToHost) ==
            cudaSuccess)
      if (cudaMemcpy(&status, device_status, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        status = -2;
  }
  if (cudaFree(device_values) != cudaSuccess) status = -2;
  if (cudaFree(device_status) != cudaSuccess) status = -2;
  return status;
}
