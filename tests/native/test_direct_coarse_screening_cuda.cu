#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "scf/cuda/direct_screening.cuh"

namespace {
using namespace generativeqc::scf::cuda_execution;
struct Case {
  double first, second, density, tolerance;
};

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void evaluate(const Case* cases, unsigned count, unsigned* results) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const auto c = cases[i];
  const double bounds[]{c.first, c.second};
  const double density[]{c.density};
  const auto force_tolerance = fmin(c.tolerance, kForceDensityProductScreeningTolerance);
  unsigned result =
      direct_fp32_block_rejects<false>(c.first, c.second, c.density, c.tolerance, force_tolerance);
  result |= unsigned(direct_fp32_block_rejects<true>(c.first, c.second, c.density, c.tolerance,
                                                     force_tolerance))
            << 1U;
  result |= unsigned(bounded_direct_block_pair_survives_screening<DirectScreeningPurpose::Fock>(
                0, 1, 0, c.tolerance, bounds, density, true))
            << 2U;
  result |= unsigned(bounded_direct_block_pair_survives_screening<DirectScreeningPurpose::Force>(
                0, 1, 0, c.tolerance, bounds, density, true))
            << 3U;
  results[i] = result;
}

/** Compare GPU admission with the incumbent sequential FP64 arithmetic, then
 * independently check every coarse rejection against extended-precision real
 * products. Volatile temporaries retain each FP64 rounding boundary.
 */
void validate(const std::vector<Case>& cases) {
  Case* input{};
  unsigned* output{};
  check(cudaMalloc(&input, cases.size() * sizeof(Case)));
  check(cudaMalloc(&output, cases.size() * sizeof(unsigned)));
  check(cudaMemcpy(input, cases.data(), cases.size() * sizeof(Case), cudaMemcpyHostToDevice));
  evaluate<<<(cases.size() + 255) / 256, 256>>>(input, cases.size(), output);
  check(cudaGetLastError());
  std::vector<unsigned> results(cases.size());
  check(cudaMemcpy(results.data(), output, results.size() * sizeof(unsigned),
                   cudaMemcpyDeviceToHost));
  check(cudaFree(input));
  check(cudaFree(output));
  std::array<std::size_t, 2> rejects{};
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto c = cases[i];
    volatile double q = c.first * c.second;
    volatile double fock = q * c.density;
    volatile double force = fock * c.density;
    // CUDA fmin selects the numeric operand even for signaling NaNs;
    // host libm implementations need not agree on that signaling case.
    const auto ft = std::isnan(c.tolerance)
                        ? kForceDensityProductScreeningTolerance
                        : std::fmin(c.tolerance, kForceDensityProductScreeningTolerance);
    const bool expected_fock = !(q < c.tolerance || fock < c.tolerance);
    const bool expected_force = expected_fock && !(force < ft);
    const bool fast_fock = (results[i] & 1U) != 0;
    const bool fast_force = (results[i] & 2U) != 0;
    const bool actual_fock = (results[i] & 4U) != 0;
    const bool actual_force = (results[i] & 8U) != 0;
    if (actual_fock != expected_fock || actual_force != expected_force ||
        (fast_fock && expected_fock) || (fast_force && expected_force)) {
      std::fprintf(stderr, "admission mismatch at %zu: %a %a %a %a result=%u\n", i, c.first,
                   c.second, c.density, c.tolerance, results[i]);
      throw std::runtime_error("FP32 changed the FP64 task set");
    }
    if (std::isfinite(c.first) && c.first >= 0 && std::isfinite(c.second) && c.second >= 0 &&
        std::isfinite(c.density) && c.density >= 0) {
      const long double exact_q = static_cast<long double>(c.first) * c.second;
      const long double exact_fock = exact_q * c.density;
      const long double exact_force = exact_fock * c.density;
      const bool real_fock = exact_q < c.tolerance || exact_fock < c.tolerance;
      const bool real_force = real_fock || exact_force < ft;
      if ((fast_fock && !real_fock) || (fast_force && !real_force))
        throw std::runtime_error("coarse bound violated extended-precision oracle");
    }
    rejects[0] += fast_fock;
    rejects[1] += fast_force;
  }
#if defined(__NVCC__)
  if (rejects[0] < 1000 || rejects[1] < 1000)
    throw std::runtime_error("vacuous coarse rejection coverage");
#else
  // Unqualified compatibility compilers must exercise only the FP64 fallback.
  if (rejects[0] != 0 || rejects[1] != 0)
    throw std::runtime_error("unqualified compiler enabled coarse rejection");
#endif
  std::printf(
      "{\"cases\":%zu,\"fock_rejections\":%zu,\"force_rejections\":%zu,"
      "\"task_set_mismatches\":0}\n",
      cases.size(), rejects[0], rejects[1]);
}
}  // namespace

int main() try {
  int devices = 0;
  const auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || devices == 0)
    return 77;
  check(status);
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> edges{0.0,
                                  -0.0,
                                  -1.0,
                                  -inf,
                                  nan,
                                  inf,
                                  std::numeric_limits<double>::denorm_min(),
                                  std::numeric_limits<double>::min(),
                                  std::numeric_limits<double>::max(),
                                  0x1p-150,
                                  0x1p-149,
                                  0x1p-126,
                                  0x1p-100,
                                  0x1p-48,
                                  0.5,
                                  1.0,
                                  std::nextafter(1.0, inf),
                                  0x1p100,
                                  0x1p128};
  std::vector<Case> cases;
  for (double a : edges)
    for (double b : edges)
      for (double d : edges)
        for (double t : {0.0, -1.0, 0x1p-1074, 0x1p-149, 1e-14, 1e-12, 1.0, inf, nan})
          cases.push_back({a, b, d, t});
  std::mt19937_64 random(20261023);
  for (unsigned i = 0; i < 300000; ++i) {
    auto sample = [&] {
      const std::uint64_t bits = random() & 0x7fffffffffffffffULL;
      double value;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    };
    const double a = sample(), b = sample(), d = sample();
    cases.push_back({a, b, d, sample()});
    // Include equality and both adjacent thresholds for all three gates.
    const double q = a * b, f = q * d, g = f * d;
    for (double threshold : {q, f, g})
      for (double t : {threshold, std::nextafter(threshold, 0.0), std::nextafter(threshold, inf)})
        cases.push_back({a, b, d, t});
  }
  validate(cases);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
