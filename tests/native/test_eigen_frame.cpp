#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/solver/eigen_frame.hpp"

namespace {
using Matrix = std::vector<double>;
void require(bool value, const char* detail) {
  if (!value) throw std::runtime_error(detail);
}
void checked_frames() {
  using namespace generativeqc::scf::solver;
  // An analytic generalized frame with nonidentity metric. The ordinary
  // validator never needs a production or reference eigensolver to test it.
  const Matrix f{2, 0, 0, 12}, s{2, 0, 0, 4}, values{1, 3};
  const Matrix c{1 / std::sqrt(2.0), 0, 0, .5};
  EigenFrameDiagnostic diagnostic;
  std::string detail;
  require(validate_eigen_frame(f, &s, values, c, 2, diagnostic, detail), detail.c_str());
  for (const int info : {-1, 1}) {
    diagnostic.solver_info = info;
    require(!validate_eigen_frame(f, &s, values, c, 2, diagnostic, detail),
            "nonzero solver info accepted a successful-looking frame");
    require(detail == "DF eigensolver returned nonzero solver info: " + std::to_string(info),
            "solver info failure did not identify the provider status");
  }
  diagnostic = {};
  auto bad = c;
  bad[0] *= 1.01;
  require(!validate_eigen_frame(f, &s, values, bad, 2, diagnostic, detail),
          "nonorthogonal frame passed");
  auto shifted = values;
  shifted[0] += 1e-6;
  require(!validate_eigen_frame(f, &s, shifted, c, 2, diagnostic, detail),
          "incorrect eigenvalue passed physical residual gate");
  require(!validate_eigen_frame(f, &s, Matrix{3, 1}, c, 2, diagnostic, detail),
          "unordered eigenvalues passed");
  require(
      detail == "DF eigensolver returned unordered eigenvalues at index 1: previous=3, current=1",
      "ordering failure did not identify the inverted eigenvalue pair");
  bad[0] = std::numeric_limits<double>::quiet_NaN();
  require(!validate_eigen_frame(f, &s, values, bad, 2, diagnostic, detail),
          "nonfinite output passed");
  require(detail == "DF eigensolver returned nonfinite frame inputs or outputs",
          "nonfinite failure was conflated with solver info or eigenvalue order");
  bad.assign(4, std::numeric_limits<double>::max());
  require(!validate_eigen_frame(f, &s, values, bad, 2, diagnostic, detail),
          "overflowing finite-input validation products passed");
  require(!validate_eigen_frame(f, &s, values, Matrix{1}, 2, diagnostic, detail),
          "truncated output passed");
  require(detail == "DF eigensolver returned invalid frame dimensions",
          "shape failure was conflated with solver info or eigenvalue order");
  // A degenerate ordinary frame accepts rotations within its entire eigenspace.
  const double a = std::sqrt(.5);
  require(validate_eigen_frame(Matrix{2, 0, 0, 2}, nullptr, Matrix{2, 2}, Matrix{a, -a, a, a}, 2,
                               diagnostic, detail),
          "degenerate rotation was rejected");
}

void ordered_pairs() {
  using namespace generativeqc::scf::solver;
  Matrix values{4, -1, 2}, coefficients{0, 1, 0, 0, 0, 1, 1, 0, 0};
  const Matrix identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
  require(order_eigenpairs(values, coefficients, 3), "valid eigenpairs could not be ordered");
  require(values == Matrix{-1, 2, 4} && coefficients == identity,
          "a permutation cycle did not move whole row-major coefficient columns");
  EigenFrameDiagnostic diagnostic;
  std::string detail;
  require(validate_eigen_frame(Matrix{-1, 0, 0, 0, 2, 0, 0, 0, 4}, nullptr, values, coefficients, 3,
                               diagnostic, detail),
          "reordered pairs failed their original analytic eigenframe");
  values = {2, 1, 2};
  coefficients = {0, 1, 0, 0, 0, 1, 1, 0, 0};
  require(order_eigenpairs(values, coefficients, 3) && values == Matrix{1, 2, 2} &&
              coefficients == Matrix{1, 0, 0, 0, 0, 1, 0, 1, 0},
          "equal eigenvalues lost their stable provider-column order");
  const double larger = std::nextafter(2.0, std::numeric_limits<double>::infinity());
  values = {larger, 2};
  coefficients = {0, 1, 1, 0};
  require(order_eigenpairs(values, coefficients, 2) && values == Matrix{2, larger} &&
              coefficients == Matrix{1, 0, 0, 1},
          "roundoff-scale inversion was rounded or separated from its coefficient column");
  require(validate_eigen_frame(Matrix{2, 0, 0, larger}, nullptr, values, coefficients, 2,
                               diagnostic, detail),
          "roundoff-scale reordered pairs failed the unmodified physical gates");
  for (const int info : {-1, 1}) {
    diagnostic.solver_info = info;
    require(!validate_eigen_frame(Matrix{2, 0, 0, larger}, nullptr, values, coefficients, 2,
                                  diagnostic, detail),
            "ordering hid a nonzero provider status");
  }
  values = {3, 1};
  coefficients = {1, 0, 0, 1};
  require(order_eigenpairs(values, coefficients, 2), "finite wrong pairs could not be presented");
  diagnostic = {};
  require(!validate_eigen_frame(Matrix{1, 0, 0, 3}, nullptr, values, coefficients, 2, diagnostic,
                                detail),
          "ordering falsely accepted eigenvalues with incorrect coefficient columns");
  const auto before_values = values, before_coefficients = coefficients;
  require(!order_eigenpairs(values, coefficients, 3) && values == before_values &&
              coefficients == before_coefficients,
          "malformed frame was mutated by ordering");
  values[0] = std::numeric_limits<double>::infinity();
  require(!order_eigenpairs(values, coefficients, 2) && std::isinf(values[0]) &&
              coefficients == before_coefficients,
          "nonfinite eigenvalue was repaired or its frame was mutated");
  values = {3, 1};
  coefficients[0] = std::numeric_limits<double>::infinity();
  require(!order_eigenpairs(values, coefficients, 2) && values == Matrix{3, 1} &&
              std::isinf(coefficients[0]),
          "nonfinite coefficient frame was reordered");
  values = {1};
  require(!order_eigenpairs(values, values, 1) && values == Matrix{1},
          "aliased eigenpair storage was accepted or modified");
  Matrix empty;
  require(!order_eigenpairs(empty, coefficients, 0) &&
              !order_eigenpairs(empty, coefficients, std::numeric_limits<std::size_t>::max()),
          "invalid or overflowing dimension was accepted");
  std::vector<std::size_t> permutation{0, 1, 2, 3};
  do {
    values.resize(permutation.size());
    coefficients.assign(permutation.size() * permutation.size(), 0);
    for (std::size_t column = 0; column < permutation.size(); ++column) {
      values[column] = static_cast<double>(permutation[column]);
      coefficients[permutation[column] * permutation.size() + column] = 1;
    }
    require(order_eigenpairs(values, coefficients, permutation.size()),
            "a finite permutation was refused");
    for (std::size_t row = 0; row < permutation.size(); ++row) {
      require(values[row] == static_cast<double>(row), "permuted eigenvalue has wrong destination");
      for (std::size_t column = 0; column < permutation.size(); ++column)
        require(coefficients[row * permutation.size() + column] == (row == column ? 1 : 0),
                "permuted coefficient column has wrong destination");
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));
}
}  // namespace
int main() {
  try {
    checked_frames();
    ordered_pairs();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
