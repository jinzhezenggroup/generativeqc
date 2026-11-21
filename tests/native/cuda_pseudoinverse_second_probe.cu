#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generated_symmetric_matrix_function.cuh"

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceStorage {
  double* values{};
  ~DeviceStorage() {
    if (values) (void)cudaFree(values);
  }
};
}  // namespace

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  try {
    std::size_t dimension{};
    double threshold{};
    if (!(std::cin >> dimension >> threshold) || !dimension || dimension > 128)
      throw std::runtime_error("invalid spectral probe input");
    const auto square = dimension * dimension;
    std::vector<double> input(dimension + 3 * square);
    for (auto& value : input)
      if (!(std::cin >> value)) throw std::runtime_error("incomplete spectral probe input");
    DeviceStorage device;
    check(cudaMalloc(&device.values, (input.size() + 5 * square) * sizeof(double)));
    check(cudaMemcpy(device.values, input.data(), input.size() * sizeof(double),
                     cudaMemcpyHostToDevice));
    const auto* vectors = device.values + dimension;
    const auto* left = vectors + square;
    const auto* right = left + square;
    auto* scratch = device.values + input.size();
    auto* output = scratch + 4 * square;
    generativeqc::tensor::launch_symmetric_pseudoinverse_second(
        dimension, vectors, device.values, threshold, left, right, scratch, output, nullptr);
    check(cudaGetLastError());
    std::vector<double> result(square);
    check(cudaMemcpy(result.data(), output, square * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto value : result) std::cout << value << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
