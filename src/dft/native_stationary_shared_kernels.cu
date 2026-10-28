#include <cstddef>

#include "dft/mixed_rsh_exchange.hpp"
#include "dft/native_stationary_shared_kernels.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::dft {
namespace {

__global__ void transport_source(const double* input, std::size_t coordinates, std::size_t channels,
                                 double sign, double* first, double* second, int* error) {
  const auto coordinate = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (coordinate >= coordinates) return;
  first[coordinate] = generativeqc_tensor::finite(sign * input[coordinate], error, 0);
  if (channels == 2) {
    const auto value =
        generativeqc_tensor::finite(sign * input[coordinates + coordinate], error, 0);
    if (second)
      second[coordinate] = value;
    else if (value != 0.0)
      atomicExch(error, 1);
  }
}

__global__ void recompose_range(const double* correction, std::size_t coordinates, double fraction,
                                double sign, double* short_range, double* long_range, int* error) {
  const auto coordinate = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (coordinate >= coordinates) return;
  double sr{}, lr{};
  mixed_rsh_exchange_sources(short_range[coordinate], sign * correction[coordinate], fraction, sr,
                             lr);
  short_range[coordinate] = generativeqc_tensor::finite(sr, error, 0);
  long_range[coordinate] = generativeqc_tensor::finite(lr, error, 0);
}

__global__ void reduce_sources(const double* input, std::size_t atoms, std::size_t sources,
                               double* output, int* error) {
  if (*error) return;
  const auto coordinate = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (coordinate >= 3 * atoms) return;
  double sum = 0;
  // ProgramIR's complete source order is retained. Each validated source has
  // coefficient exactly one; the method-specific scalar multiplications have
  // already happened at the original source owner.
  for (std::size_t source = 0; source < sources; ++source)
    sum += 1.0 * input[source * 3 * atoms + coordinate];
  output[coordinate] = generativeqc_tensor::finite(sum, error, 0);
}

}  // namespace

cudaError_t launch_native_stationary_source_transport(const double* input, std::size_t coordinates,
                                                      std::size_t channels, double sign,
                                                      double* first, double* second, int* error,
                                                      cudaStream_t stream) noexcept {
  if (!input || !first || !error || !coordinates || (channels != 1 && channels != 2))
    return cudaErrorInvalidValue;
  transport_source<<<generativeqc_tensor::blocks(coordinates, 64), 64, 0, stream>>>(
      input, coordinates, channels, sign, first, second, error);
  return cudaGetLastError();
}

cudaError_t launch_native_stationary_range_correction(const double* correction,
                                                      std::size_t coordinates, double fraction,
                                                      double sign, double* short_range,
                                                      double* long_range, int* error,
                                                      cudaStream_t stream) noexcept {
  if (!correction || !coordinates || !short_range || !long_range || !error)
    return cudaErrorInvalidValue;
  recompose_range<<<generativeqc_tensor::blocks(coordinates, 64), 64, 0, stream>>>(
      correction, coordinates, fraction, sign, short_range, long_range, error);
  return cudaGetLastError();
}

cudaError_t launch_native_stationary_source_reduction(const double* input, std::size_t atoms,
                                                      std::size_t sources, double* output,
                                                      int* error, cudaStream_t stream) noexcept {
  if (!input || !output || !error || !atoms || !sources || sources > 64)
    return cudaErrorInvalidValue;
  reduce_sources<<<generativeqc_tensor::blocks(3 * atoms, 64), 64, 0, stream>>>(
      input, atoms, sources, output, error);
  return cudaGetLastError();
}

}  // namespace generativeqc::dft
