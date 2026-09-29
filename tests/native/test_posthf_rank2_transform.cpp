#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "posthf/rank2_transform.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<double> ao_to_mo_oracle(std::span<const double> c, std::span<const double> a,
                                    std::size_t n) {
  std::vector<double> result(n * n, 0.0);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          result[p * n + q] += c[mu * n + p] * a[mu * n + nu] * c[nu * n + q];
  return result;
}

std::vector<double> mo_to_ao_oracle(std::span<const double> c, std::span<const double> w,
                                    std::size_t n) {
  std::vector<double> result(n * n, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = 0; nu < n; ++nu)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q)
          result[mu * n + nu] += c[mu * n + p] * w[p * n + q] * c[nu * n + q];
  return result;
}

void require_close(std::span<const double> actual, std::span<const double> expected,
                   const char* message) {
  require(actual.size() == expected.size(), "rank-2 transform result shape mismatch");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    require(std::isfinite(actual[i]) && std::isfinite(expected[i]),
            "rank-2 transform comparison requires finite values");
    maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
  }
  require(maximum < 2e-13, message);
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
        require_close(invalid_actual ? nonfinite : finite, invalid_actual ? finite : nonfinite,
                      "nonfinite comparison accepted");
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected, "rank-2 oracle comparison accepted NaN or infinity");
    }
  }
  require_close(finite, finite, "finite equality failed");
}

void transforms_match_independent_oracles() {
  constexpr std::size_t n = 4;
  const std::array<double, n * n> c{0.91,  0.12, -0.08, 0.03, 0.16, 0.88,  0.17, -0.04,
                                    -0.07, 0.19, 0.93,  0.11, 0.05, -0.03, 0.14, 0.96};
  const std::array<double, n * n> matrix{-1.2,  0.07, -0.03, 0.11, 0.07, -0.4,  0.09, -0.02,
                                         -0.03, 0.09, 0.25,  0.06, 0.11, -0.02, 0.06, 0.71};

  const generativeqc::tensor::CpuLinalgPlan scalar{
      generativeqc::tensor::CpuLinalgProvider::scalar,
      generativeqc::tensor::CpuLinalgThreadOwnership::task_parallel, 1};

  require_close(generativeqc::posthf::rank2_ao_to_mo(c, matrix, n, scalar),
                ao_to_mo_oracle(c, matrix, n), "AO-to-MO GEMM transform disagrees with oracle");
  require_close(generativeqc::posthf::rank2_mo_to_ao(c, matrix, n, scalar),
                mo_to_ao_oracle(c, matrix, n), "MO-to-AO GEMM transform disagrees with oracle");
  require_close(generativeqc::posthf::rank2_ao_to_mo(c, matrix, n), ao_to_mo_oracle(c, matrix, n),
                "automatic AO-to-MO transform disagrees with oracle");
  require_close(generativeqc::posthf::rank2_mo_to_ao(c, matrix, n), mo_to_ao_oracle(c, matrix, n),
                "automatic MO-to-AO transform disagrees with oracle");

  require(generativeqc::posthf::rank2_transform_workspace_bytes(n) == n * n * sizeof(double),
          "rank-2 transform workspace accounting is not one dense matrix");
}

void invalid_shapes_fail_closed() {
  const std::array<double, 4> matrix{1, 0, 0, 1};
  bool rejected = false;
  try {
    (void)generativeqc::posthf::rank2_ao_to_mo(matrix, std::span(matrix).first(3), 2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "rank-2 transform accepted a wrong matrix shape");

  rejected = false;
  try {
    (void)generativeqc::posthf::rank2_mo_to_ao({}, {}, 0);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "rank-2 transform accepted a zero dimension");
}

}  // namespace

int main() {
  try {
    nonfinite_comparisons_fail_closed();
    transforms_match_independent_oracles();
    invalid_shapes_fail_closed();
    std::cout << "post-HF rank-2 transforms passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
