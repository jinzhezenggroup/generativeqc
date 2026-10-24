#include "scf/solver/eigen_frame.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

#include "scf/solver/validation_products.hpp"

namespace generativeqc::scf::solver {
bool order_eigenpairs(std::vector<double>& eigenvalues, std::vector<double>& coefficients,
                      std::size_t dimension) {
  const auto finite = [](const auto& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (&eigenvalues == &coefficients || dimension == 0 ||
      dimension > std::numeric_limits<std::size_t>::max() / dimension ||
      eigenvalues.size() != dimension || coefficients.size() != dimension * dimension ||
      !finite(eigenvalues) || !finite(coefficients))
    return false;
  if (std::is_sorted(eigenvalues.begin(), eigenvalues.end())) return true;
  std::vector<std::size_t> order(dimension), destination(dimension);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    return eigenvalues[left] < eigenvalues[right];
  });
  for (std::size_t column = 0; column < dimension; ++column) destination[order[column]] = column;
  // Each swap fixes one original column's destination. Permutation cycles
  // therefore cost at most n-1 full-column swaps without a second n-by-n frame.
  for (std::size_t column = 0; column < dimension; ++column)
    while (destination[column] != column) {
      const auto other = destination[column];
      std::swap(eigenvalues[column], eigenvalues[other]);
      for (std::size_t row = 0; row < dimension; ++row)
        std::swap(coefficients[row * dimension + column], coefficients[row * dimension + other]);
      std::swap(destination[column], destination[other]);
    }
  return true;
}

bool validate_eigen_frame(const std::vector<double>& matrix, const std::vector<double>* overlap,
                          const std::vector<double>& values,
                          const std::vector<double>& coefficients, std::size_t n,
                          EigenFrameDiagnostic& diagnostic, std::string& detail,
                          bool require_eigen_residual) {
  const auto finite = [](const auto& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (diagnostic.solver_info != 0) {
    detail =
        "DF eigensolver returned nonzero solver info: " + std::to_string(diagnostic.solver_info);
    return false;
  }
  if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n || matrix.size() != n * n ||
      coefficients.size() != n * n || values.size() != n || (overlap && overlap->size() != n * n)) {
    detail = "DF eigensolver returned invalid frame dimensions";
    return false;
  }
  if (!finite(matrix) || !finite(coefficients) || !finite(values) ||
      (overlap && !finite(*overlap))) {
    detail = "DF eigensolver returned nonfinite frame inputs or outputs";
    return false;
  }
  const auto unordered = std::is_sorted_until(values.begin(), values.end());
  if (unordered != values.end()) {
    const auto position = static_cast<std::size_t>(unordered - values.begin());
    std::ostringstream message;
    message.precision(std::numeric_limits<double>::max_digits10);
    message << "DF eigensolver returned unordered eigenvalues at index " << position
            << ": previous=" << values[position - 1] << ", current=" << values[position];
    detail = message.str();
    return false;
  }
  // Validate the original F/S inputs on the host. Only the dense product
  // schedule is optional; every finiteness, norm and physical gate below stays.
  const ValidationProducts multiply;
  const auto sc = overlap ? multiply(*overlap, coefficients, n) : coefficients;
  const auto gram = multiply(coefficients, sc, n, true);
  // Allocate FC after any fallback transpose temporary has been released.
  const auto fc = multiply(matrix, coefficients, n);
  // Finite inputs may still overflow a product. Do not let NaNs disappear
  // through std::max or an infinite scale hide an invalid residual.
  if (!finite(fc) || !finite(sc) || !finite(gram)) {
    detail = "DF eigenframe validation products are nonfinite";
    return false;
  }
  double residual_norm = 0, matrix_norm = 0, coefficient_norm = 0, rhs_norm = 0;
  diagnostic.maximum_eigen_residual = diagnostic.maximum_metric_error = 0;
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      const auto k = row * n + column;
      const double rhs = sc[k] * values[column];
      const double residual = fc[k] - rhs;
      residual_norm = std::hypot(residual_norm, residual);
      matrix_norm = std::hypot(matrix_norm, matrix[k]);
      coefficient_norm = std::hypot(coefficient_norm, coefficients[k]);
      rhs_norm = std::hypot(rhs_norm, rhs);
      diagnostic.maximum_eigen_residual =
          std::max(diagnostic.maximum_eigen_residual, std::abs(residual));
      diagnostic.maximum_metric_error = std::max(diagnostic.maximum_metric_error,
                                                 std::abs(gram[k] - (row == column ? 1.0 : 0.0)));
    }
  }
  const double scale = matrix_norm * coefficient_norm + rhs_norm;
  diagnostic.scaled_eigen_residual = scale == 0 ? residual_norm : residual_norm / scale;
  if (!std::isfinite(scale) || !std::isfinite(residual_norm)) {
    detail = "DF eigenframe validation norm overflow";
    return false;
  }
  return accept_eigen_frame(diagnostic, detail, require_eigen_residual);
}

bool accept_eigen_frame(const EigenFrameDiagnostic& diagnostic, std::string& detail,
                        bool require_eigen_residual) {
  if (diagnostic.solver_info != 0 || !std::isfinite(diagnostic.maximum_eigen_residual) ||
      !std::isfinite(diagnostic.maximum_metric_error) ||
      !std::isfinite(diagnostic.scaled_eigen_residual) || diagnostic.maximum_eigen_residual < 0 ||
      diagnostic.maximum_metric_error < 0 || diagnostic.scaled_eigen_residual < 0 ||
      diagnostic.maximum_metric_error > 1e-8 ||
      (require_eigen_residual &&
       (diagnostic.maximum_eigen_residual > 1e-8 || diagnostic.scaled_eigen_residual > 1e-12))) {
    detail = require_eigen_residual
                 ? "DF eigensystem failed physical eigen residual or metric orthogonality checks"
                 : "DF determinant frame failed metric orthogonality checks";
    return false;
  }
  return true;
}
}  // namespace generativeqc::scf::solver
