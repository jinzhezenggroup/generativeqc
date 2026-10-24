#ifndef GENERATIVEQC_SOLVER_DIIS_AFFINE_HPP
#define GENERATIVEQC_SOLVER_DIIS_AFFINE_HPP

#include <cfloat>
#include <cstddef>

namespace generativeqc::solver::detail {

/** Check the Pulay affine constraint before accepting a normalized KS solve.
 *
 * A finite pivoted solve can lose sum(c)=1 for nearly dependent residuals.
 * Such coefficients change even a Fock common to every history entry. Allow
 * eight FP64 summation-error budgets, gamma_(n+1) * max(1, ||c||_1), for the
 * sum and subtraction of one. This is a backward-error check, not a cap on
 * individual coefficients or a change to SCF convergence tolerances. Reject
 * nonfinite arithmetic and cancellation so large that the bound reaches one.
 * Callers retire dependent history or retain the current Fock on failure;
 * renormalizing an inaccurate solve would hide its loss of precision.
 *
 * The Lagrange multiplier is excluded: count is the number of Fock weights.
 * Host and CUDA use this same arithmetic and acceptance policy.
 */
#ifdef __CUDACC__
__host__ __device__
#endif
    inline bool diis_affine_constraint(const double* coefficients, std::size_t count) {
  if (!count) return false;
  double sum = 0.0, norm = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = coefficients[i];
    if (!(value >= -DBL_MAX && value <= DBL_MAX)) return false;
    sum += value;
    norm += value < 0.0 ? -value : value;
  }
  if (!(norm <= DBL_MAX)) return false;
  const double rounding = (static_cast<double>(count) + 1.0) * DBL_EPSILON;
  if (!(rounding < 1.0)) return false;
  const double tolerance = 8.0 * (rounding / (1.0 - rounding)) * (norm > 1.0 ? norm : 1.0);
  const double residual = sum - 1.0;
  return tolerance < 1.0 && residual >= -tolerance && residual <= tolerance;
}

/** Solve an already normalized augmented Pulay system in place.
 *
 * Partial-pivot forward elimination followed by back substitution avoids
 * repeatedly updating the affine row with ill-conditioned intermediate rows,
 * as Gauss-Jordan elimination does. The pivot threshold is unchanged. Check
 * the affine constraint after solving; failure leaves scratch unspecified and
 * asks the owner to retire history or retain current Fock. Matrix and rhs are
 * disjoint writable arrays of (count+1)^2 and count+1 doubles, respectively.
 */
#ifdef __CUDACC__
__host__ __device__
#endif
    inline bool solve_affine_diis_system(double* matrix, double* rhs, std::size_t count) {
  const std::size_t dimension = count + 1;
  for (std::size_t column = 0; column < dimension; ++column) {
    std::size_t pivot = column;
    double largest = 0.0;
    for (std::size_t row = column; row < dimension; ++row) {
      const double value = matrix[row * dimension + column];
      const double magnitude = value < 0.0 ? -value : value;
      if (magnitude > largest) {
        largest = magnitude;
        pivot = row;
      }
    }
    if (!(largest >= 1e-14 && largest <= DBL_MAX)) return false;
    if (pivot != column) {
      for (std::size_t item = column; item < dimension; ++item) {
        const double value = matrix[column * dimension + item];
        matrix[column * dimension + item] = matrix[pivot * dimension + item];
        matrix[pivot * dimension + item] = value;
      }
      const double value = rhs[column];
      rhs[column] = rhs[pivot];
      rhs[pivot] = value;
    }
    const double diagonal = matrix[column * dimension + column];
    for (std::size_t row = column + 1; row < dimension; ++row) {
      const double factor = matrix[row * dimension + column] / diagonal;
      matrix[row * dimension + column] = 0.0;
      for (std::size_t item = column + 1; item < dimension; ++item)
        matrix[row * dimension + item] -= factor * matrix[column * dimension + item];
      rhs[row] -= factor * rhs[column];
    }
  }
  for (std::size_t row = dimension; row-- > 0;) {
    double value = rhs[row];
    for (std::size_t column = row + 1; column < dimension; ++column)
      value -= matrix[row * dimension + column] * rhs[column];
    rhs[row] = value / matrix[row * dimension + row];
    if (!(rhs[row] >= -DBL_MAX && rhs[row] <= DBL_MAX)) return false;
  }
  return diis_affine_constraint(rhs, count);
}

}  // namespace generativeqc::solver::detail

#endif
