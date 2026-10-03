#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "scf/reference/linalg.hpp"
#include "scf/solver/cpu_target_eigen.hpp"
#include "tensor/cpu_linalg.hpp"

namespace {
using namespace generativeqc;
using scf::reference::Matrix;
void require(bool value, const char* detail) {
  if (!value) throw std::runtime_error(detail);
}
template <class F>
void rejects(F&& operation) {
  bool failed = false;
  try {
    operation();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed, "malformed target eigen input accepted");
}
void analytic_frames() {
  // Analytic 45-degree two-dimensional rotations and diagonal SPD metrics.
  // Eigenvalues are independently prescribed, including exact degeneracy.
  for (const auto values :
       {Matrix{-3, 7}, Matrix{2, 2}, Matrix{-1e-10, 3e-10}, Matrix{-1e5, 3e5}}) {
    for (const auto metric : {Matrix{1, 1}, Matrix{.25, 4}, Matrix{1e-4, 1e4}}) {
      const double a = .5 * (values[0] + values[1]);
      const double b = .5 * (values[0] - values[1]);
      const Matrix f{metric[0] * a, std::sqrt(metric[0] * metric[1]) * b,
                     std::sqrt(metric[0] * metric[1]) * b, metric[1] * a};
      const Matrix s{metric[0], 0, 0, metric[1]};
      const Matrix x{1 / std::sqrt(metric[0]), 0, 0, 1 / std::sqrt(metric[1])};
      const auto before_f = f, before_s = s, before_x = x;
      try {
        const auto result = scf::solver::cpu_target_eigen(f, &s, &x, 2);
        for (std::size_t k = 0; k != 2; ++k)
          require(std::abs(result.values[k] - values[k]) < 2e-10,
                  "independent analytic spectrum disagrees");
      } catch (const std::runtime_error& error) {
        // Large metric-scaled matrices must still obey absolute residual gates.
        if (metric[1] != 1e4 || std::abs(values[1]) < 1e5 ||
            std::string_view(error.what()) !=
                "DF eigensystem failed physical eigen residual or metric orthogonality checks")
          throw;
      }
      require(f == before_f && s == before_s && x == before_x, "input changed");
    }
  }
  const auto scalar = scf::solver::cpu_target_eigen(Matrix{4}, nullptr, nullptr, 1);
  require(scalar.values == Matrix{4} && scalar.vectors == Matrix{1}, "one-dimensional frame");
  const Matrix f{2, 1, 1, 2}, x{1, 0, 0, 1};
  const auto ordinary = scf::solver::cpu_target_eigen(f, nullptr, nullptr, 2);
  require(std::abs(ordinary.values[0] - 1) < 1e-14 && std::abs(ordinary.values[1] - 3) < 1e-14,
          "ordinary spectrum");
  rejects([&] { scf::solver::cpu_target_eigen(f, &x, nullptr, 2); });
  rejects([&] { scf::solver::cpu_target_eigen(f, nullptr, &x, 2); });
  rejects([&] { scf::solver::cpu_target_eigen(Matrix{1}, nullptr, nullptr, 2); });
  rejects([&] { scf::solver::cpu_target_eigen(f, nullptr, nullptr, 0); });
  rejects([&] {
    scf::solver::cpu_target_eigen_workspace_bytes(std::numeric_limits<std::size_t>::max());
  });
  auto bad = f;
  bad[0] = std::numeric_limits<double>::infinity();
  rejects([&] { scf::solver::cpu_target_eigen(bad, nullptr, nullptr, 2); });
  bad = Matrix{1, 0, 0, 2};
  rejects([&] { scf::solver::cpu_target_eigen(f, &bad, &x, 2); });
  bad = Matrix{1, 2, 3, 4};
  rejects([&] { scf::solver::cpu_target_eigen(bad, nullptr, nullptr, 2); });
}
void strict_scalar_cap() {
  using namespace tensor;
  const CpuLinalgPlan scalar{CpuLinalgProvider::scalar, CpuLinalgThreadOwnership::task_parallel, 1};
  // The default relative threshold skips this coupling; the explicit cap must
  // retain the independently known +/- off-diagonal splitting.
  const Matrix a{1e6, 1e-9, 1e-9, 1e6};
  const auto capped = cpu_symmetric_eigen(a, 2, scalar, 1e-14);
  require(capped.values[0] < 1e6 && capped.values[1] > 1e6, "absolute cap ignored");
  const auto legacy = cpu_symmetric_eigen(a, 2, scalar);
  const auto zero = cpu_symmetric_eigen(a, 2, scalar, 0.0);
  require(legacy.values == zero.values && legacy.vectors == zero.vectors, "default changed");
  rejects([&] { (void)cpu_symmetric_eigen(a, 2, {}, 1e-14); });
  rejects([&] { (void)cpu_symmetric_eigen(a, 2, scalar, -1); });
  rejects(
      [&] { (void)cpu_symmetric_eigen(a, 2, scalar, std::numeric_limits<double>::infinity()); });
}
}  // namespace
int main() {
  try {
    analytic_frames();
    strict_scalar_cap();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  std::cout << "CPU target eigen analytic and cap tests passed\n";
}
