#include "scf/occupied_projection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "integrals/s_integrals.hpp"
#include "scf/initial_guess/density.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::scf::projection {
namespace {

std::size_t elements(std::size_t rows, std::size_t columns) {
  if (rows && columns > std::numeric_limits<std::size_t>::max() / rows / sizeof(double))
    throw Rejected("occupied projection dimensions overflow");
  return rows * columns;
}

struct Matrix {
  std::size_t rows{}, columns{};
  std::vector<double> data;

  Matrix(std::size_t row_count, std::size_t column_count)
      : rows(row_count), columns(column_count), data(elements(rows, columns)) {}

  Matrix(std::size_t row_count, std::size_t column_count, std::span<const double> values,
         const char* name)
      : rows(row_count), columns(column_count) {
    if (values.size() != elements(rows, columns) ||
        !std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); }))
      throw Rejected(std::string(name) + " has invalid dimensions or nonfinite values");
    data.assign(values.begin(), values.end());
  }

  double& operator()(std::size_t row, std::size_t column) { return data[row * columns + column]; }
  double operator()(std::size_t row, std::size_t column) const {
    return data[row * columns + column];
  }
};

void finite(const Matrix& matrix) {
  if (!std::all_of(matrix.data.begin(), matrix.data.end(),
                   [](double value) { return std::isfinite(value); }))
    throw Rejected("occupied projection produced nonfinite numeric state");
}

Matrix product(const Matrix& left, const Matrix& right, bool transpose_left = false,
               bool transpose_right = false, double scale = 1.0) {
  const auto rows = transpose_left ? left.columns : left.rows;
  const auto inner = transpose_left ? left.rows : left.columns;
  const auto columns = transpose_right ? right.rows : right.columns;
  if (inner != (transpose_right ? right.columns : right.rows))
    throw std::logic_error("inconsistent occupied projection contraction");
  Matrix output(rows, columns);
  if (rows && columns && inner)
    tensor::cpu_gemm(transpose_left ? 'T' : 'N', transpose_right ? 'T' : 'N', rows, columns, inner,
                     left.data.data(), right.data.data(), output.data.data(), scale);
  finite(output);
  return output;
}

double norm(const Matrix& matrix) {
  double value = 0.0;
  for (const auto item : matrix.data) value = std::hypot(value, item);
  return value;
}

double difference_norm(const Matrix& left, const Matrix& right) {
  double value = 0.0;
  for (std::size_t index = 0; index < left.data.size(); ++index)
    value = std::hypot(value, left.data[index] - right.data[index]);
  return value;
}

Matrix symmetrize(Matrix matrix) {
  for (std::size_t row = 0; row < matrix.rows; ++row)
    for (std::size_t column = row + 1; column < matrix.columns; ++column)
      matrix(row, column) = matrix(column, row) =
          0.5 * matrix(row, column) + 0.5 * matrix(column, row);
  return matrix;
}

struct Metric {
  Matrix matrix, vectors;
  std::vector<double> values;
};

Metric metric(std::span<const double> input, std::size_t dimension, const char* name,
              const Policy& policy) {
  Matrix matrix(dimension, dimension, input, name);
  double largest = 0.0;
  for (const auto value : matrix.data) largest = std::max(largest, std::abs(value));
  if (largest == 0.0) throw Rejected(std::string(name) + " is not positive semidefinite");
  // The public symmetry gate uses the spectral norm of the unsymmetrized
  // input. Normalize before A^T A so finite extreme metrics do not square
  // their physical scale or silently replace that gate with a Frobenius norm.
  Matrix scaled = matrix;
  for (auto& value : scaled.data) value /= largest;
  auto squared = product(scaled, scaled, true);
  const auto singular = tensor::cpu_symmetric_eigen(std::move(squared.data), dimension);
  const auto spectral_scale = std::sqrt(std::max(0.0, singular.values.back()));
  const auto scale = std::max(spectral_scale, std::numeric_limits<double>::min() / largest);
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = row + 1; column < dimension; ++column)
      if (std::abs(scaled(row, column) - scaled(column, row)) > policy.validation_tolerance * scale)
        throw Rejected(std::string(name) + " is not Hermitian");
  matrix = symmetrize(std::move(matrix));
  auto spectrum = tensor::cpu_symmetric_eigen(matrix.data, dimension);
  if (spectrum.values.back() <= 0.0 ||
      spectrum.values.front() < -policy.validation_tolerance * spectrum.values.back())
    throw Rejected(std::string(name) + " is not positive semidefinite");
  const auto cutoff = policy.relative_threshold * spectrum.values.back();
  const auto first = static_cast<std::size_t>(
      std::upper_bound(spectrum.values.begin(), spectrum.values.end(), cutoff) -
      spectrum.values.begin());
  Matrix vectors(dimension, dimension - first);
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = first; column < dimension; ++column)
      vectors(row, column - first) = spectrum.vectors[row * dimension + column];
  return {std::move(matrix), std::move(vectors),
          std::vector<double>(spectrum.values.begin() + first, spectrum.values.end())};
}

void validate(const Input& input, const Policy& policy) {
  for (const auto value :
       {policy.relative_threshold, policy.validation_tolerance, policy.maximum_residual})
    if (!std::isfinite(value) || value <= 0.0 || value > 1.0)
      throw std::invalid_argument("occupied projection thresholds must be finite and in (0,1]");
  if (!policy.maximum_ao || !input.source_ao || !input.target_ao ||
      input.source_ao > policy.maximum_ao || input.target_ao > policy.maximum_ao)
    throw Rejected("overlap must be a bounded nonempty square AO metric");
  if (input.occupation != 1 && input.occupation != 2)
    throw std::invalid_argument("occupation must be 1 or 2");
  if (input.occupied > std::min(input.source_ao, input.target_ao))
    throw Rejected("source or target metric has insufficient occupied rank");
}

Result occupied(const Input& input, const Policy& policy, const Metric& source,
                const Matrix& coefficients) {
  const auto target = metric(input.target_overlap, input.target_ao, "target overlap", policy);
  if (input.occupied > std::min(source.values.size(), target.values.size()))
    throw Rejected("source or target metric has insufficient occupied rank");
  const Matrix cross(input.target_ao, input.source_ao, input.cross_overlap, "cross overlap");
  const auto retained = product(source.vectors, product(source.vectors, coefficients, true));
  if (difference_norm(coefficients, retained) >
      policy.validation_tolerance * std::max(1.0, norm(coefficients)))
    throw Rejected("source occupied orbitals contain discarded metric directions");
  auto inverse_rooted = target.vectors;
  for (std::size_t row = 0; row < inverse_rooted.rows; ++row)
    for (std::size_t column = 0; column < inverse_rooted.columns; ++column)
      inverse_rooted(row, column) /= std::sqrt(target.values[column]);
  auto orthogonalizer = product(inverse_rooted, target.vectors, false, true);
  Matrix square_coefficients(input.source_ao, input.source_ao);
  for (std::size_t row = 0; row < input.source_ao; ++row)
    for (std::size_t column = 0; column < input.occupied; ++column)
      square_coefficients(row, column) = coefficients(row, column);
  integrals::IntegralData target_data;
  target_data.nbf = input.target_ao;
  target_data.overlap = target.matrix.data;
  initial_guess::OccupiedProjectionResult transported;
  try {
    transported = initial_guess::project_occupied_density_with_controls(
        target_data, orthogonalizer.data, source.matrix.data, cross.data, input.source_ao,
        square_coefficients.data, input.occupied, input.occupation, policy.maximum_residual,
        {policy.validation_tolerance, policy.relative_threshold, 0.0, 0.0, true,
         [](const reference::Matrix& matrix, const reference::Matrix*, const reference::Matrix*,
            std::size_t dimension) {
           auto result = tensor::cpu_symmetric_eigen(matrix, dimension);
           return reference::EigenResult{std::move(result.values), std::move(result.vectors)};
         }});
  } catch (const std::invalid_argument& error) {
    throw Rejected(error.what());
  }
  Matrix orthogonal(input.target_ao, input.occupied, transported.coefficients,
                    "target coefficients");
  Matrix density(input.target_ao, input.target_ao, transported.density, "target density");
  auto metric_gram = product(orthogonal, product(target.matrix, orthogonal), true);
  for (std::size_t index = 0; index < input.occupied; ++index) metric_gram(index, index) -= 1.0;
  const auto metric_error = norm(metric_gram);
  auto density_metric = product(density, target.matrix);
  double trace = 0.0;
  for (std::size_t index = 0; index < input.target_ao; ++index)
    trace += density_metric(index, index);
  auto squared_density = product(density_metric, density);
  for (std::size_t index = 0; index < density.data.size(); ++index)
    squared_density.data[index] -= input.occupation * density.data[index];
  const auto idempotency = norm(squared_density);
  if (!std::isfinite(trace) ||
      metric_error > policy.validation_tolerance * std::max<std::size_t>(1, input.occupied) ||
      std::abs(trace - input.occupation * input.occupied) >
          policy.validation_tolerance *
              std::max<std::size_t>(1, input.occupation * input.occupied) ||
      idempotency > policy.validation_tolerance * input.occupation * std::max(1.0, norm(density)))
    throw Rejected("target seed failed metric/electron validation");
  return {std::move(orthogonal.data),
          std::move(density.data),
          {source.values.size(), target.values.size(), input.occupied,
           transported.minimum_projected_norm, transported.projection_residual,
           transported.normal_equation_residual, metric_error, trace, idempotency}};
}

}  // namespace

Result project_occupied(const Input& input, const Policy& policy) {
  validate(input, policy);
  const auto source = metric(input.source_overlap, input.source_ao, "source overlap", policy);
  const Matrix coefficients(input.source_ao, input.occupied, input.source_state,
                            "source coefficients");
  return occupied(input, policy, source, coefficients);
}

Result project_density(const Input& input, const Policy& policy) {
  validate(input, policy);
  const auto source = metric(input.source_overlap, input.source_ao, "source overlap", policy);
  const Matrix density(input.source_ao, input.source_ao, input.source_state, "source density");
  for (std::size_t row = 0; row < input.source_ao; ++row)
    for (std::size_t column = row + 1; column < input.source_ao; ++column)
      if (std::abs(density(row, column) - density(column, row)) > policy.validation_tolerance)
        throw Rejected("source density is not Hermitian");
  if (input.occupied > source.values.size())
    throw Rejected("source metric has insufficient occupied rank");
  auto rooted = source.vectors;
  for (std::size_t row = 0; row < rooted.rows; ++row)
    for (std::size_t column = 0; column < rooted.columns; ++column)
      rooted(row, column) *= std::sqrt(source.values[column]);
  auto orthogonal_density = symmetrize(product(rooted, product(density, rooted), true));
  auto spectrum = rooted.columns ? tensor::cpu_symmetric_eigen(std::move(orthogonal_density.data),
                                                               rooted.columns)
                                 : tensor::CpuSymmetricEigenResult{};
  const auto first = source.values.size() - input.occupied;
  for (std::size_t index = 0; index < spectrum.values.size(); ++index)
    if (std::abs(spectrum.values[index] - (index < first ? 0.0 : input.occupation)) >
        policy.validation_tolerance)
      throw Rejected("source density has inconsistent electron count or fractional occupations");
  Matrix selected(source.values.size(), input.occupied);
  for (std::size_t row = 0; row < selected.rows; ++row)
    for (std::size_t column = 0; column < selected.columns; ++column)
      selected(row, column) = spectrum.vectors[row * selected.rows + first + column];
  auto inverse_rooted = source.vectors;
  for (std::size_t row = 0; row < inverse_rooted.rows; ++row)
    for (std::size_t column = 0; column < inverse_rooted.columns; ++column)
      inverse_rooted(row, column) /= std::sqrt(source.values[column]);
  auto coefficients = product(inverse_rooted, selected);
  const auto reconstructed = product(coefficients, coefficients, false, true, input.occupation);
  for (std::size_t index = 0; index < density.data.size(); ++index)
    if (std::abs(reconstructed.data[index] - density.data[index]) >
        policy.validation_tolerance * (1.0 + std::abs(density.data[index])))
      throw Rejected("source density contains unsupported discarded metric components");
  return occupied(input, policy, source, coefficients);
}

}  // namespace generativeqc::scf::projection
