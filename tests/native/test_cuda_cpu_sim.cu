// Experimental CPU-simulator smoke test of the production FP64 scatter primitive.
// This checks CUDA execution semantics, not throughput or real-device acceptance.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/compensated_atomic.cuh"

namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__ void scatter(const double* values, std::size_t count, double* sum,
                        double* correction) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x)
    generativeqc::runtime::atomicAdd({sum, correction}, values[i]);
}

std::array<double, 2> execute(const std::vector<double>& values, unsigned blocks,
                              unsigned threads) {
  double* input = nullptr;
  double* accumulators = nullptr;
  try {
    check(cudaMalloc(&input, values.size() * sizeof(double)), "cudaMalloc input");
    check(cudaMalloc(&accumulators, 2 * sizeof(double)), "cudaMalloc accumulators");
    check(cudaMemcpy(input, values.data(), values.size() * sizeof(double),
                     cudaMemcpyHostToDevice), "cudaMemcpy input");
    check(cudaMemset(accumulators, 0, 2 * sizeof(double)), "cudaMemset accumulators");
    scatter<<<blocks, threads>>>(input, values.size(), accumulators, accumulators + 1);
    check(cudaGetLastError(), "scatter launch");
    check(cudaDeviceSynchronize(), "scatter synchronize");

    std::array<double, 2> result{};
    check(cudaMemcpy(result.data(), accumulators, sizeof(result), cudaMemcpyDeviceToHost),
          "cudaMemcpy output");
    check(cudaFree(accumulators), "cudaFree accumulators");
    accumulators = nullptr;
    check(cudaFree(input), "cudaFree input");
    input = nullptr;
    return result;
  } catch (...) {
    if (accumulators) (void)cudaFree(accumulators);
    if (input) (void)cudaFree(input);
    throw;
  }
}

}  // namespace

int main() {
  try {
    int devices = 0;
    check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices < 1) throw std::runtime_error("CUDA runtime did not expose a simulated GPU");
    check(cudaSetDevice(0), "cudaSetDevice");

    // Independent exact dyadic oracle: 2^60 + 1 - 2^60 - 1/2 + 1/4 = 3/4.
    // Keep the simulation small; the full stress suite belongs on actual GPUs.
    constexpr int copies = 16;
    constexpr std::array<double, 5> terms{0x1p60, 1.0, -0x1p60, -0.5, 0.25};
    std::vector<double> values;
    values.reserve(copies * terms.size());
    for (int i = 0; i < copies; ++i) values.insert(values.end(), terms.begin(), terms.end());

    std::mt19937 random(2019);
    constexpr std::array<std::pair<unsigned, unsigned>, 3> shapes{{{1, 32}, {2, 64}, {4, 32}}};
    for (const auto& [blocks, threads] : shapes) {
      std::shuffle(values.begin(), values.end(), random);
      for (double sign : {1.0, -1.0}) {
        std::vector<double> signed_values(values);
        for (auto& value : signed_values) value *= sign;
        const auto actual = execute(signed_values, blocks, threads);
        const double sum = actual[0] + actual[1];
        const double expected = sign * copies * 0.75;
        if (!std::isfinite(sum) || sum != expected) {
          std::fprintf(stderr, "FP64 scatter: blocks=%u threads=%u sum=%.17g expected=%.17g\n",
                       blocks, threads, sum, expected);
          return 1;
        }
      }
    }

    const auto nonfinite = execute({std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::infinity()},
                                   1, 32);
    if (std::isfinite(nonfinite[0] + nonfinite[1]))
      throw std::runtime_error("nonfinite FP64 atomic result was hidden");
    std::puts("CPU-simulated CUDA: 6 compensated-atomic cancellation cases and nonfinite gate passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CPU-simulated CUDA smoke failed: %s\n", error.what());
    return 1;
  }
}
