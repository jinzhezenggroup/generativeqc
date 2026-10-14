#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks_kernels.hpp"

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

/** An independent scalar contraction supplies the exact secant coefficient.
 * Odd length exercises the reduction tail. Invalid/noncontracting histories
 * must yield an unchanged proposal selection, never a nonfinite coefficient. */
int main() {
  constexpr std::size_t count = 1025, bytes = count * sizeof(double);
  double* device = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&device), 6 * bytes + sizeof(double)));
  try {
    std::vector<double> anchor(count), proposal(count), delta(count), old_fock(count), fock(count),
        actual(count);
    for (unsigned scenario = 0; scenario < 7; ++scenario) {
      const double ratio = scenario == 1   ? -0.001
                           : scenario == 3 ? 0.5
                           : scenario == 4 ? 1.0
                           : scenario == 6 ? 0.08
                                           : 0.001;
      for (std::size_t i = 0; i < count; ++i) {
        anchor[i] = 0.01 * std::cos(double(i));
        delta[i] = scenario == 2 ? 0.0 : 0.001 * std::sin(double(i + 1));
        proposal[i] = anchor[i] + ratio * delta[i];
        old_fock[i] = 2.0 + std::cos(double(i));
        fock[i] = old_fock[i] + 0.02 * std::sin(double(i));
      }
      if (scenario == 5) proposal[513] = std::numeric_limits<double>::quiet_NaN();
      for (const auto& entry :
           {std::pair{device, &anchor}, std::pair{device + count, &proposal},
            std::pair{device + 2 * count, &delta}, std::pair{device + 3 * count, &fock},
            std::pair{device + 4 * count, &old_fock}})
        check(cudaMemcpy(entry.first, entry.second->data(), bytes, cudaMemcpyHostToDevice));
      auto* coefficient = device + 6 * count;
      generativeqc::dft::cuda_ks_detail::outer_secant_coefficient(
          nullptr, count, device, device + count, device + 2 * count, coefficient);
      check(cudaGetLastError());
      double value = 0;
      check(cudaMemcpy(&value, coefficient, sizeof(value), cudaMemcpyDeviceToHost));
      const double expected = scenario < 2 ? ratio / (1.0 - ratio) : 0.0;
      require(std::isfinite(value) && std::abs(value - expected) < 2e-14,
              "secant coefficient differs from independent contraction");
      generativeqc::dft::cuda_ks_detail::outer_secant_fock(
          nullptr, count, value, device + 3 * count, device + 4 * count, device + 5 * count);
      check(cudaGetLastError());
      check(cudaMemcpy(actual.data(), device + 5 * count, bytes, cudaMemcpyDeviceToHost));
      for (std::size_t i = 0; i < count; ++i)
        require(std::abs(actual[i] - (fock[i] + expected * (fock[i] - old_fock[i]))) < 2e-14,
                "secant Fock differs from independent linear combination");
    }
    check(cudaFree(device));
  } catch (...) {
    (void)cudaFree(device);
    throw;
  }
  std::cout << "seven independent secant admission and Fock cases passed\n";
}
