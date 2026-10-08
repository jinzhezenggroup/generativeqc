bool dot_product(const double* first, const double* second, std::size_t count, double& result) {
  double sum = 0.0;
  for (std::size_t index = 0u; index < count; ++index) {
    sum += first[index] * second[index];
    if (!std::isfinite(sum)) {
      return false;
    }
  }
  result = sum;
  return true;
}

bool cholesky_solve(double* matrix, double* right_hand_side, std::size_t dimension) {
  /* beta is symmetric positive definite because omega0^2 is added to I. */
  for (std::size_t row = 0u; row < dimension; ++row) {
    for (std::size_t column = 0u; column <= row; ++column) {
      double value = matrix[row * dimension + column];
      for (std::size_t inner = 0u; inner < column; ++inner) {
        value -= matrix[row * dimension + inner] * matrix[column * dimension + inner];
      }
      if (!std::isfinite(value)) {
        return false;
      }
      if (row == column) {
        if (!(value > 0.0)) {
          return false;
        }
        matrix[row * dimension + column] = std::sqrt(value);
      } else {
        const double diagonal = matrix[column * dimension + column];
        value /= diagonal;
        if (!std::isfinite(value)) {
          return false;
        }
        matrix[row * dimension + column] = value;
      }
    }
  }

  for (std::size_t row = 0u; row < dimension; ++row) {
    double value = right_hand_side[row];
    for (std::size_t column = 0u; column < row; ++column) {
      value -= matrix[row * dimension + column] * right_hand_side[column];
    }
    value /= matrix[row * dimension + row];
    if (!std::isfinite(value)) {
      return false;
    }
    right_hand_side[row] = value;
  }
  for (std::size_t reverse = dimension; reverse > 0u; --reverse) {
    const std::size_t row = reverse - 1u;
    double value = right_hand_side[row];
    for (std::size_t column = row + 1u; column < dimension; ++column) {
      value -= matrix[column * dimension + row] * right_hand_side[column];
    }
    value /= matrix[row * dimension + row];
    if (!std::isfinite(value)) {
      return false;
    }
    right_hand_side[row] = value;
  }
  return true;
}