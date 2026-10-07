// Execute the production generated paired loop on host thread indices. This
// checks scalar order/work ownership, not CUDA concurrency or device execution.
#define __device__
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generated_gfn2_electronic_native.cuh"

struct Dimensions {
  unsigned x = 0, z = 0;
} blockIdx, threadIdx, blockDim, gridDim;
struct MatrixPair {
  std::int64_t row, column;
};
MatrixPair matrix_pair(std::int64_t pair) {
  std::int64_t row = 0;
  while (pair > row) pair -= ++row;
  return {row, pair};
}
enum class Gfn2DensityDeviceError {
  kNonfiniteDensityArithmetic = 8,
  kNonfiniteWeightedDensityArithmetic = 9
};
void record_system_error(std::uint32_t* errors, std::int64_t system, std::uint32_t* error,
                         Gfn2DensityDeviceError code) {
  if (errors[system] == 0) errors[system] = static_cast<std::uint32_t>(code);
  if (*error == 0) *error = static_cast<std::uint32_t>(code);
}
std::uint32_t run(std::int64_t count, unsigned threads, unsigned tiles,
                  const std::vector<double>& coefficients, const std::vector<double>& weights,
                  const std::vector<double>& energy_weights, std::vector<double>& p,
                  std::vector<double>& w) {
  struct Input {
    const double* coefficients;
  } input{coefficients.data()};
  struct Workspace {
    const double *weights, *energy_weights;
    double *density_scratch, *weighted_density_scratch;
  } workspace{weights.data(), energy_weights.data(), p.data(), w.data()};
  std::uint32_t status = 0, global = 0;
  std::uint32_t *system_errors = &status, *device_error = &global;
  const std::int64_t system = 0, matrix_begin = 0, orbital_begin = 0;
  const std::int64_t pair_count = count * (count + 1) / 2;
  blockDim.x = threads;
  gridDim.z = tiles;
  for (blockIdx.z = 0; blockIdx.z < tiles; ++blockIdx.z)
    for (threadIdx.x = 0; threadIdx.x < threads; ++threadIdx.x) {
#include "generated_gfn2_density_contract.inc"
    }
  return status;
}
void require(bool test) {
  if (!test) throw std::runtime_error("weighted Gram order gate");
}
bool exact(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}
int main() {
  constexpr double sentinel = -973.375;
  for (std::int64_t n : {1, 2, 3, 7, 17})
    for (unsigned threads : {1, 4, 16})
      for (unsigned tiles : {1, 2, 5}) {
        std::vector<double> c(n * n), f(n), e(n), ew(n), p(n * n, sentinel), w(n * n, sentinel);
        for (std::int64_t i = 0; i < n * n; ++i) c[i] = (i % 11 - 5) * 0.13;
        for (std::int64_t i = 0; i < n; ++i) {
          f[i] = (i % 7) * 0.17;
          e[i] = (i % 5 - 3) * 0.19;
          ew[i] = f[i] * e[i];
        }
        require(run(n, threads, tiles, c, f, ew, p, w) == 0);
        for (std::int64_t r = 0; r < n; ++r)
          for (std::int64_t col = 0; col <= r; ++col) {
            double op = 0, ow = 0;
            long double mp = 0, mw = 0;
            for (std::int64_t k = 0; k < n; ++k) {
              const double pl = c[r * n + k] * f[k], wl = c[r * n + k] * ew[k];
              const double pc = pl * c[col * n + k], wc = wl * c[col * n + k];
              require(std::isfinite(pc) && std::isfinite(wc));
              op = std::fma(pl, c[col * n + k], op);
              ow = std::fma(wl, c[col * n + k], ow);
              mp += static_cast<long double>(c[r * n + k]) * f[k] * c[col * n + k];
              mw += static_cast<long double>(c[r * n + k]) * f[k] * e[k] * c[col * n + k];
            }
            require(exact(p[r * n + col], op) && exact(p[col * n + r], op));
            require(exact(w[r * n + col], ow) && exact(w[col * n + r], ow));
            require(std::abs(static_cast<long double>(op) - mp) < 2e-14L);
            require(std::abs(static_cast<long double>(ow) - mw) < 2e-14L);
          }
      }
  // Every failure leaves both channel cells unpublished, even after P's update.
  for (int which = 0; which < 3; ++which) {
    const double c = which == 0 ? 1e308 : which == 1 ? 1e200 : 1e100;
    const double f = which == 0 ? 2.0 : 0.5;
    const double ew = which == 2 ? 5e149 : f;
    std::vector<double> p(1, sentinel), w(1, sentinel);
    require(run(1, 1, 1, {c}, {f}, {ew}, p, w) == (which == 2 ? 9U : 8U));
    require(p[0] == sentinel && w[0] == sentinel);
  }
  {
    std::vector<double> p(4, sentinel), w(4, sentinel);
    require(run(2, 1, 1, {1e154, 1e154, 0, 0}, {1, 1}, {1, 1}, p, w) == 8U);
    require(p[0] == sentinel && w[0] == sentinel);
  }
  using namespace generativeqc::xtb::generated;
  double output = sentinel;
  require(!gfn2_energy_weight_cuda_tensor(2, 1e308, output) && output == sentinel);
  require(!gfn2_density_contribution_cuda_tensor(1e308, 2, output) && output == sentinel);
  // A finite fused result must not erase the separate nonfused overflow gate.
  require(gfn2_density_update_cuda_tensor(1e308, 2, -1e308, output));
  require(output == std::fma(1e308, 2, -1e308));
  for (double zero : {0.0, -0.0}) {
    std::vector<double> p(1, sentinel), w(1, sentinel);
    require(run(1, 1, 1, {zero}, {0.5}, {-0.5}, p, w) == 0);
    require(exact(p[0], std::fma(zero * 0.5, zero, 0.0)));
    require(exact(w[0], std::fma(zero * -0.5, zero, 0.0)));
  }
}
