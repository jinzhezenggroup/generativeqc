#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "runtime/compensated_atomic.cuh"
#include "runtime/cuda_resources.cuh"

namespace {

__global__ void scatter(const double* values, std::size_t count, double* sum, double* correction) {
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    generativeqc::runtime::atomicAdd({sum, correction}, values[index]);
}

/** Closed-form dyadic reference avoids reproducing the device summation.
 * Shuffle the same cancellation-heavy multiset across launches and block
 * shapes; residual recovery must cover both magnitude orderings and signs.
 */
void check_cancellation(cudaStream_t stream) {
  using namespace generativeqc::runtime;
  constexpr std::size_t copies = 8192;
  constexpr std::array<double, 5> terms{0x1p60, 1.0, -0x1p60, -0.5, 0.25};
  std::vector<double> values;
  for (std::size_t copy = 0; copy < copies; ++copy)
    values.insert(values.end(), terms.begin(), terms.end());
  OwnedCudaBuffer<double> input(0, values.size(), stream), output(0, 2, stream);
  std::mt19937 random(2019);
  for (unsigned repeat = 0; repeat < 24; ++repeat) {
    std::shuffle(values.begin(), values.end(), random);
    cuda_resource_check(cudaMemcpyAsync(input.get(), values.data(), values.size() * sizeof(double),
                                        cudaMemcpyHostToDevice, stream));
    cuda_resource_check(cudaMemsetAsync(output.get(), 0, 2 * sizeof(double), stream));
    scatter<<<1 + repeat * 13, 32 + 32 * (repeat % 8), 0, stream>>>(input.get(), values.size(),
                                                                    output.get(), output.get() + 1);
    cuda_resource_check(cudaGetLastError());
    std::array<double, 2> actual{};
    cuda_resource_check(cudaMemcpyAsync(actual.data(), output.get(), sizeof(actual),
                                        cudaMemcpyDeviceToHost, stream));
    cuda_resource_check(cudaStreamSynchronize(stream));
    if (actual[0] + actual[1] != copies * 0.75)
      throw std::runtime_error("compensated scatter lost exact dyadic sum");
  }
}

/** Nonfinite arithmetic cannot be repaired into a plausible finite output. */
void check_nonfinite(cudaStream_t stream) {
  using namespace generativeqc::runtime;
  OwnedCudaBuffer<double> input(0, 2, stream), output(0, 2, stream);
  for (double value :
       {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::max()}) {
    const std::array<double, 2> values{value, value};
    cuda_resource_check(cudaMemcpyAsync(input.get(), values.data(), sizeof(values),
                                        cudaMemcpyHostToDevice, stream));
    cuda_resource_check(cudaMemsetAsync(output.get(), 0, 2 * sizeof(double), stream));
    scatter<<<1, 32, 0, stream>>>(input.get(), 2, output.get(), output.get() + 1);
    cuda_resource_check(cudaGetLastError());
    std::array<double, 2> actual{};
    cuda_resource_check(cudaMemcpyAsync(actual.data(), output.get(), sizeof(actual),
                                        cudaMemcpyDeviceToHost, stream));
    cuda_resource_check(cudaStreamSynchronize(stream));
    if (std::isfinite(actual[0] + actual[1]))
      throw std::runtime_error("compensated scatter hid nonfinite arithmetic");
  }
}
}  // namespace

int main() {
  cudaStream_t stream{};
  try {
    generativeqc::runtime::cuda_resource_check(cudaSetDevice(0));
    generativeqc::runtime::cuda_resource_check(cudaStreamCreate(&stream));
    check_cancellation(stream);
    check_nonfinite(stream);
    generativeqc::runtime::cuda_resource_check(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& failure) {
    if (stream) cudaStreamDestroy(stream);
    std::fprintf(stderr, "%s\n", failure.what());
    return 1;
  }
}
