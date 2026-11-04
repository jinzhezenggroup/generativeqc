"""Emit a bounded actual-CUDA probe of stored-D versus occupied XC features.

This is a representation counterexample, not a molecular SCF calculation or a
performance benchmark. Both contractions and feature consumers come from the
checked-out production generators. Run the emitted executable only via Slurm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _cpp_source_support import cpp_function_definition, cpp_record_definition
from generativeqc_compiler.dft.ao_cuda import emit_native_xc_contraction_kernels
from generativeqc_compiler.dft.feature_policy import emit_feature_policy
from generativeqc_compiler.dft.xc_contraction_cuda import (
    DEFAULT_XC_MATRIX_SCHEDULE,
    _emit_occupied_features,
    _emit_tiled,
)

ROOT = Path(__file__).resolve().parents[3]

PREAMBLE = r"""
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>
using I = std::int64_t;
namespace generativeqc_tensor {
inline unsigned blocks(I count, unsigned width) { return (count + width - 1) / width; }
inline void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}
__device__ double finite(double value, int* error, int code) {
  if (!isfinite(value)) atomicExch(error, code);
  return value;
}
"""

DRIVER = r"""
__global__ void dense_products(const double* density, const double* ao,
                               double* work, int* error) {
  tiled_density_product_body<false, true>(density, ao, 16, 1, 4,
                                         work, error, nullptr, 16, blockIdx.z);
}
__global__ void dense_features(const double* ao, const double* work,
                               double* features, int* error) {
  density_features_body<true>(ao, work, 16, 1, 1, 4, 4, 5, 0, features, error);
}
template<class Scalar> struct DeviceBuffer {
  Scalar* value{};
  explicit DeviceBuffer(std::size_t count) {
    generativeqc_tensor::cuda_check(cudaMalloc(&value, count * sizeof(Scalar)));
  }
  ~DeviceBuffer() { if (value) cudaFree(value); }
};
void observe(cudaStream_t stream, bool underflow) {
  using generativeqc_tensor::cuda_check;
  std::array<double, 256> frame{}, density{};
  std::array<double, 64> ao{};
  for (std::size_t orbital = 0; orbital < 16; ++orbital)
    frame[orbital * 16 + orbital] = 1.0;
  if (underflow) {
    frame[0] = 1e-200;
    frame[1] = 1.0;
    frame[16] = 1.0;
    frame[17] = -1e-200;
  }
  ao[0] = underflow ? 1e200 : 0.25;
  for (std::size_t row = 0; row < 16; ++row)
    for (std::size_t column = 0; column < 16; ++column)
      density[row * 16 + column] = 2.0 * std::fma(frame[row], frame[column], 0.0);
  DeviceBuffer<double> device_frame(256), device_density(256), device_ao(64),
      dense_work(64), occupied_work(64), device_features(5);
  DeviceBuffer<int> error(1);
  cuda_check(cudaMemcpyAsync(device_frame.value, frame.data(), sizeof(frame),
                             cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemcpyAsync(device_density.value, density.data(), sizeof(density),
                             cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemcpyAsync(device_ao.value, ao.data(), sizeof(ao),
                             cudaMemcpyHostToDevice, stream));
  cuda_check(cudaMemsetAsync(error.value, 0, sizeof(int), stream));
  dense_products<<<dim3(1, 1, 4), dim3(16, 16), 0, stream>>>(
      device_density.value, device_ao.value, dense_work.value, error.value);
  cuda_check(cudaGetLastError());
  dense_features<<<1, 32, 0, stream>>>(device_ao.value, dense_work.value,
                                      device_features.value, error.value);
  cuda_check(cudaGetLastError());
  std::array<double, 5> dense{}, occupied{};
  cuda_check(cudaMemcpyAsync(dense.data(), device_features.value, sizeof(dense),
                             cudaMemcpyDeviceToHost, stream));
  CudaXcOccupiedDensity witness{device_density.value, device_frame.value, 16, 1, {1, 0}, 1};
  if (!witness.matches(device_density.value, 16, 1, 1))
    throw std::runtime_error("finite fixture rejected by structural witness");
  launch_occupied_density(stream, witness, device_ao.value, 16, 1,
                          occupied_work.value, device_features.value, 5,
                          error.value, nullptr);
  cuda_check(cudaMemcpyAsync(occupied.data(), device_features.value, sizeof(occupied),
                             cudaMemcpyDeviceToHost, stream));
  int failure = 0;
  cuda_check(cudaMemcpyAsync(&failure, error.value, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  if (failure) throw std::runtime_error("generated feature consumer reported an error");
  std::cout << "{\"fixture\":\"" << (underflow ? "stored-density-underflow" : "ordinary-control")
            << "\",\"stored_D00\":" << density[0]
            << ",\"witness_matches\":true,\"error\":" << failure << ",\"dense_features\":[";
  for (std::size_t term = 0; term < 5; ++term) {
    if (term) std::cout << ',';
    if (!std::isfinite(dense[term]) || !std::isfinite(occupied[term]))
      throw std::runtime_error("nonfinite output cannot be published as a numeric observation");
    std::cout << dense[term];
  }
  std::cout << "],\"occupied_features\":[";
  for (std::size_t term = 0; term < 5; ++term) {
    if (term) std::cout << ',';
    std::cout << occupied[term];
  }
  std::cout << "]}";
}
int main() {
  try {
    using generativeqc_tensor::cuda_check;
    int device = 0, driver = 0, runtime = 0;
    cuda_check(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device));
    cuda_check(cudaDriverGetVersion(&driver));
    cuda_check(cudaRuntimeGetVersion(&runtime));
    cudaStream_t stream{};
    cuda_check(cudaStreamCreate(&stream));
    std::cout << std::setprecision(17) << "{\"device\":\"" << properties.name
              << "\",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime
              << ",\"scope\":\"emitted XC feature consumers, not molecular SCF\",\"observations\":[";
    observe(stream, false);
    std::cout << ',';
    observe(stream, true);
    cuda_check(cudaStreamDestroy(stream));
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}
"""


def main() -> None:
    """Extract actual emitted consumers without loading a runtime or GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generated = emit_native_xc_contraction_kernels()
    args.output.write_text(
        PREAMBLE
        + cpp_record_definition(
            (ROOT / "src/dft/cuda_xc.hpp").read_text(), "CudaXcOccupiedDensity"
        )
        + ";\nnamespace generativeqc_grid_policy {\n"
        + emit_feature_policy(device=True)
        + "}\n"
        + cpp_function_definition(
            _emit_tiled(DEFAULT_XC_MATRIX_SCHEDULE),
            "tiled_density_product_body",
            include_template=True,
        )
        + cpp_function_definition(
            generated, "density_features_body", include_template=True
        )
        + _emit_occupied_features(DEFAULT_XC_MATRIX_SCHEDULE)
        + DRIVER
    )


if __name__ == "__main__":
    main()
