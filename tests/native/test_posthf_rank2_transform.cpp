#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "tensor/cpu_linalg.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<double> oracle(std::span<const double> c, std::span<const double> a, std::size_t n,
                           bool transposed) {
  std::vector<double> result(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) {
          const auto left = transposed ? c[p * n + i] : c[i * n + p];
          const auto right = transposed ? c[q * n + j] : c[j * n + q];
          result[i * n + j] += left * a[p * n + q] * right;
        }
  return result;
}

void require_close(std::span<const double> actual, std::span<const double> expected) {
  require(actual.size() == expected.size(), "rank-2 result shape mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    require(std::isfinite(actual[i]) && std::isfinite(expected[i]),
            "rank-2 comparison requires finite values");
    maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
  }
  require(maximum < 2e-13, "rank-2 congruence disagrees with independent oracle");
}

void nonfinite_comparisons_fail_closed() {
  const std::array<double, 2> finite{0.0, 1.0};
  for (const double invalid :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    const std::array<double, 2> nonfinite{0.0, invalid};
    for (const bool invalid_actual : {false, true}) {
      bool rejected = false;
      try {
        require_close(invalid_actual ? nonfinite : finite, invalid_actual ? finite : nonfinite);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected, "rank-2 oracle comparison accepted NaN or infinity");
    }
  }
  require_close(finite, finite);
}

void transforms_match_independent_oracles() {
  constexpr std::size_t n = 4;
  const std::array<double, n * n> c{0.91,  0.12, -0.08, 0.03, 0.16, 0.88,  0.17, -0.04,
                                    -0.07, 0.19, 0.93,  0.11, 0.05, -0.03, 0.14, 0.96};
  // Nonsymmetric data catches accidental transpose assumptions as well.
  const std::array<double, n * n> matrix{-1.2,  0.07, -0.03, 0.11, 0.02, -0.4,  0.09, -0.02,
                                         -0.05, 0.13, 0.25,  0.06, 0.17, -0.01, 0.04, 0.71};
  using namespace generativeqc::tensor;
  for (const auto provider : {CpuLinalgProvider::scalar, CpuLinalgProvider::automatic}) {
    const CpuLinalgPlan plan{provider, CpuLinalgThreadOwnership::task_parallel, 1};
    for (const char direction : {'N', 'T', 'n', 't'}) {
      const auto nan = std::numeric_limits<double>::quiet_NaN();
      std::array<double, n * n> result, workspace;
      result.fill(nan);
      workspace.fill(nan);
      cpu_congruence(direction, n, c.data(), matrix.data(), result.data(), workspace.data(), plan);
      require_close(result, oracle(c, matrix, n, direction == 'T' || direction == 't'));
    }
  }
}

void invalid_storage_fails_closed() {
  using namespace generativeqc::tensor;
  const CpuLinalgPlan plan{CpuLinalgProvider::scalar, CpuLinalgThreadOwnership::task_parallel, 1};
  const std::array<double, 4> coefficients{1.0, 0.0, 0.0, 1.0};
  std::array<double, 4> matrix{2.0, 0.0, 0.0, 3.0}, result{}, workspace{};
  for (const int failure : {0, 1, 2, 3}) {
    bool rejected = false;
    try {
      cpu_congruence(failure == 0 ? 'X' : 'N', 2, coefficients.data(), matrix.data(),
                     failure == 1 ? matrix.data() : result.data(),
                     failure == 2   ? result.data()
                     : failure == 3 ? nullptr
                                    : workspace.data(),
                     plan);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "rank-2 congruence accepted invalid or aliased storage");
  }
  // Preserve the canonical primitive's documented BLAS-style empty-operation contract.
  cpu_congruence('N', 0, nullptr, nullptr, nullptr, nullptr, plan);
}

}  // namespace

int main() {
  try {
    nonfinite_comparisons_fail_closed();
    transforms_match_independent_oracles();
    invalid_storage_fails_closed();
    std::cout << "post-HF canonical rank-2 congruence tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
