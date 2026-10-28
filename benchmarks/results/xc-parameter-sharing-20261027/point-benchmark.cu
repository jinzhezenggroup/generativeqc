// Opt-in worker timing. Complete public KS endpoints are qualified separately.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "generated_libxc_semilocal_cuda_registry.hpp"
static void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
int main(int argc, char** argv) {
  if (!std::getenv("SLURM_JOB_ID") || argc != 6) return 2;
  const auto code = static_cast<unsigned>(std::stoul(argv[1]));
  const std::size_t count = std::stoull(argv[2]), spins = std::stoull(argv[3]);
  const std::size_t tile = std::stoull(argv[4]), repeats = std::stoull(argv[5]);
  const auto* program = generativeqc::dft::generated::automatic_cuda_point_entry(code);
  if (!program) return 3;
  const auto channels = spins * program->feature_terms;
  std::vector<double> host(count * channels), weights(count, 0.7);
  const auto width = tile ? tile : count;
  for (std::size_t first = 0; first < count; first += width) {
    auto n = std::min(width, count - first);
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t s = 0; s < spins; ++s) {
        const double density = (0.01 + 0.001 * ((first + p) % 997)) / (s + 1);
        const double values[] = {density, 0.01 * density, -0.03 * density, 0.05 * density,
                                 1.1 * density};
        for (std::size_t c = 0; c < program->feature_terms; ++c)
          host[first * channels + (s * program->feature_terms + c) * n + p] = values[c];
      }
    }
  }
  double *features{}, *device_weights{}, *coefficients{}, *totals{};
  int* error{};
  check(cudaMalloc(&features, host.size() * sizeof(double)));
  check(cudaMalloc(&device_weights, count * sizeof(double)));
  check(cudaMalloc(&coefficients, host.size() * sizeof(double)));
  check(cudaMalloc(&totals, 3 * count * sizeof(double)));
  check(cudaMalloc(&error, sizeof(int)));
  check(cudaMemset(error, 0, sizeof(int)));
  check(cudaMemcpy(features, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemcpy(device_weights, weights.data(), count * sizeof(double), cudaMemcpyHostToDevice));
  const auto launch = [&] {
    if (tile)
      program->launch_batch(nullptr, features, device_weights, count, spins, coefficients, totals,
                            error, code, 1., 1., tile);
    else
      program->launch(nullptr, features, device_weights, count, spins, coefficients, totals, error,
                      code, 1., 1., nullptr);
  };
  for (unsigned warm = 0; warm < 10; ++warm) launch();
  check(cudaDeviceSynchronize());
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin));
  check(cudaEventCreate(&end));
  std::vector<float> samples;
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    check(cudaEventRecord(begin));
    launch();
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    float ms{};
    check(cudaEventElapsedTime(&ms, begin, end));
    samples.push_back(ms);
  }
  int error_value{};
  check(cudaMemcpy(&error_value, error, sizeof(int), cudaMemcpyDeviceToHost));
  if (error_value) return 4;
  std::cout << std::setprecision(10) << "{\"code\":" << code << ",\"points\":" << count
            << ",\"spins\":" << spins << ",\"tile\":" << tile << ",\"milliseconds\":[";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << samples[i];
  }
  std::cout << "]}\n";
  check(cudaEventDestroy(begin));
  check(cudaEventDestroy(end));
  check(cudaFree(features));
  check(cudaFree(device_weights));
  check(cudaFree(coefficients));
  check(cudaFree(totals));
  check(cudaFree(error));
}
