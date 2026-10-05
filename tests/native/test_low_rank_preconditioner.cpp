#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "response/low_rank_preconditioner.hpp"
#include "response/native_gmres.hpp"

namespace {
using namespace generativeqc::response;
void require(bool value, const char* reason) {
  if (!value) throw std::runtime_error(reason);
}

void inverse_and_solve(std::size_t n, std::size_t rank) {
  std::vector<double> diagonal(n), u(rank * n), expected(n), matrix(n * n), rhs(n);
  for (std::size_t i = 0; i < n; ++i) {
    diagonal[i] = 0.7 + 0.07 * i;
    expected[i] = std::cos(double(i + 1));
    for (std::size_t q = 0; q < rank; ++q)
      u[q * n + i] = 0.12 * std::sin(double(3 * i + 2 * q + 1));
  }
  // Independent long-double dense construction is an oracle only. The
  // production preconditioner never owns a dimension-by-dimension matrix.
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      long double value = i == j ? diagonal[i] : 0.0;
      for (std::size_t q = 0; q < rank; ++q)
        value += static_cast<long double>(u[q * n + i]) * u[q * n + j];
      matrix[i * n + j] = static_cast<double>(value);
    }
  auto apply = [&](std::span<const double> x, std::span<double> y) {
    for (std::size_t i = 0; i < n; ++i) {
      long double value = 0.0;
      for (std::size_t j = 0; j < n; ++j)
        value += static_cast<long double>(matrix[i * n + j]) * x[j];
      y[i] = static_cast<double>(value);
    }
  };
  apply(expected, rhs);
  const auto capacity = LowRankPreconditioner::capacity_bytes(n, rank);
  require(!LowRankPreconditioner::prepare(diagonal, u, rank, capacity - 1),
          "short budget admitted");
  auto inverse = LowRankPreconditioner::prepare(diagonal, u, rank, capacity);
  require(inverse && inverse->owned_bytes() == capacity, "complete inverse capacity mismatch");
  std::vector<double> actual(n);
  inverse->apply(rhs, actual);
  for (std::size_t i = 0; i < n; ++i)
    require(std::abs(actual[i] - expected[i]) < 2e-13,
            "Woodbury independent dense oracle mismatch");
  actual = rhs;
  inverse->apply(actual, actual);
  for (std::size_t i = 0; i < n; ++i)
    require(std::abs(actual[i] - expected[i]) < 2e-13, "aliased inverse application");

  GmresOptions options;
  options.absolute_tolerance = 1e-11;
  options.relative_tolerance = 0.0;
  options.true_residual_every = 30;
  const auto plan = prepare_gmres(n, options);
  const auto result =
      solve_gmres(plan, apply, rhs, {}, {}, [&](auto x, auto y) { inverse->apply(x, y); });
  require(result.converged() && result.iterations == 1 && result.preconditioner_actions == 1,
          "exact low-rank inverse did not precondition GMRES in one step");
  apply(result.solution, actual);
  for (std::size_t i = 0; i < n; ++i) actual[i] -= rhs[i];
  require(stable_norm(actual) <= options.absolute_tolerance, "GMRES independent residual gate");

  for (const double unsafe : {0.0, -1.0, 1e-12, std::numeric_limits<double>::infinity()}) {
    const auto old = diagonal[0];
    diagonal[0] = unsafe;
    require(!LowRankPreconditioner::prepare(diagonal, u, rank, capacity),
            "unsafe diagonal admitted");
    diagonal[0] = old;
  }
  u[0] = std::numeric_limits<double>::max();
  require(!LowRankPreconditioner::prepare(diagonal, u, rank, capacity), "Gram overflow admitted");
}

void nonfinite_callback() {
  unsigned actions = 0;
  auto identity = [&](auto x, auto y) {
    ++actions;
    std::copy(x.begin(), x.end(), y.begin());
  };
  auto bad = [&](auto, auto y) {
    std::fill(y.begin(), y.end(), std::numeric_limits<double>::quiet_NaN());
  };
  const auto plan = prepare_gmres(2, GmresOptions{});
  const std::array<double, 2> rhs{1.0, 2.0};
  const auto result = solve_gmres(plan, identity, rhs, {}, {}, bad);
  require(result.status == GmresStatus::nonfinite_preconditioner && actions == 0,
          "nonfinite preconditioner reached physical operator");
  bool rejected = false;
  try {
    (void)solve_gmres(plan, identity, rhs, {}, rhs, bad);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "conflicting preconditioners accepted");
}
}  // namespace

int main() {
  try {
    for (std::size_t n : {1, 2, 8, 24})
      for (std::size_t rank : {1, 3, 7}) inverse_and_solve(n, rank);
    nonfinite_callback();
    std::cout << "Low-rank inverse, independent oracle, GMRES and refusal gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
