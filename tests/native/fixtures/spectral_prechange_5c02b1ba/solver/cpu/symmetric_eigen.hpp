#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/cpu/lp64_provider.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::solver::cpu {

/** Ordinary symmetric eigen execution only. Owners retain admission, generalized
 * transforms, numerical acceptance and publication. Completion is host-return.
 * Owned row-major calls may allocate; borrowed column-major calls never allocate,
 * query a provider, replace buffers, or change thread settings. */
enum class SymmetricEigenFamily {
  scalar_owned_row_major,
  lapack_owned_row_major,
  lapack_work_column_major
};
using LapackInt = ::generativeqc::tensor::cpu::LapackInt;
using DsyevdWork = ::generativeqc::tensor::cpu::LapackDsyevdWork;
using OwnedLapack = tensor::CpuSymmetricEigenResult (*)(std::vector<double>, std::size_t,
                                                        const void*);

struct SymmetricEigenWork {
  LapackInt doubles{};
  LapackInt integers{};
};

class PreparedSymmetricEigen {
 public:
  bool ready() const noexcept { return maximum_ != 0; }
  std::size_t maximum_order() const noexcept { return maximum_; }
  SymmetricEigenFamily family() const noexcept { return family_; }
  SymmetricEigenWork required() const noexcept { return work_; }
  double scalar_cap() const noexcept { return scalar_cap_; }

 private:
  std::size_t maximum_{};
  SymmetricEigenFamily family_{SymmetricEigenFamily::scalar_owned_row_major};
  SymmetricEigenWork work_{};
  double scalar_cap_{};
  friend PreparedSymmetricEigen prepare_owned_symmetric_eigen(std::size_t, SymmetricEigenFamily,
                                                              double);
  friend bool prepare_borrowed_symmetric_eigen(std::size_t, PreparedSymmetricEigen&) noexcept;
};

inline PreparedSymmetricEigen prepare_owned_symmetric_eigen(std::size_t n,
                                                            SymmetricEigenFamily family,
                                                            double scalar_cap = 0.0) {
  if (!n || n > std::numeric_limits<std::size_t>::max() / n ||
      (family != SymmetricEigenFamily::scalar_owned_row_major &&
       family != SymmetricEigenFamily::lapack_owned_row_major) ||
      !std::isfinite(scalar_cap) || scalar_cap < 0.0 ||
      (scalar_cap > 0.0 && family != SymmetricEigenFamily::scalar_owned_row_major))
    throw std::invalid_argument("invalid owned CPU symmetric eigen preparation");
  PreparedSymmetricEigen output;
  output.maximum_ = n;
  output.family_ = family;
  output.scalar_cap_ = scalar_cap;
  return output;
}

/** The owner's maximum-system formula, not an optimal-workspace query. Exact
 * counts are retained even when a smaller system borrows a padded arena. Failure
 * leaves output untouched and does not initialize any runtime provider. */
inline bool prepare_borrowed_symmetric_eigen(std::size_t maximum,
                                             PreparedSymmetricEigen& output) noexcept {
  constexpr auto limit = static_cast<std::uint64_t>(std::numeric_limits<LapackInt>::max());
  if (!maximum || maximum > limit) return false;
  const auto n = static_cast<std::uint64_t>(maximum);
  const auto doubles = 1u + 6u * n + 2u * n * n;
  const auto integers = 3u + 5u * n;
  if (doubles > limit || integers > limit) return false;
  PreparedSymmetricEigen candidate;
  candidate.maximum_ = maximum;
  candidate.family_ = SymmetricEigenFamily::lapack_work_column_major;
  candidate.work_ = {static_cast<LapackInt>(doubles), static_cast<LapackInt>(integers)};
  output = candidate;
  return true;
}

struct SymmetricEigenStorage {
  double* work{};
  std::size_t double_capacity{};
  LapackInt* integer_work{};
  std::size_t integer_capacity{};
};

struct SymmetricEigenWorkBinding {
  double* work{};
  LapackInt* integer_work{};
  SymmetricEigenWork counts{};
};

namespace detail {
struct Range {
  std::uintptr_t begin{}, end{};
};
inline bool range(const void* pointer, std::size_t count, std::size_t size, std::size_t alignment,
                  Range& output) noexcept {
  if (!pointer || !count || count > std::numeric_limits<std::size_t>::max() / size) return false;
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  const auto bytes = count * size;
  if (begin % alignment || begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
  output = {begin, begin + bytes};
  return true;
}
inline bool disjoint(Range a, Range b) noexcept { return a.end <= b.begin || b.end <= a.begin; }

// Keep the inherited scalar body text and compilation context unchanged.
// clang-format off
inline tensor::CpuSymmetricEigenResult scalar_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                               double absolute_tolerance = 0.0) {
  // Jacobi angle differences and doubled off-diagonals can overflow even
  // when every input and eigenvalue is representable. Normalize only extreme
  // scales; preserve established ordinary-range arithmetic and eigenvectors.
  double scale = 0.0, output_scale = 1.0;
  for (double value : matrix) scale = std::max(scale, std::abs(value));
  if (scale > std::sqrt(std::numeric_limits<double>::max()) ||
      (scale > 0.0 && scale < std::sqrt(std::numeric_limits<double>::min()))) {
    output_scale = scale;
    for (double& value : matrix) value /= scale;
  }
  std::vector<double> vectors(matrix.size(), 0.0);
  for (std::size_t item = 0; item < n; ++item) vectors[item * n + item] = 1.0;
  constexpr std::size_t maximum_sweeps = 100;
  bool converged = n == 1;
  for (std::size_t sweep = 0; sweep < maximum_sweeps && !converged; ++sweep) {
    double matrix_scale = 0.0;
    for (double value : matrix) matrix_scale = std::max(matrix_scale, std::abs(value));
    if (matrix_scale == 0.0) {
      converged = true;
      break;
    }
    double tolerance = 1.0e-14 * matrix_scale;
    if (absolute_tolerance > 0.0)
      tolerance = std::min(tolerance, absolute_tolerance / output_scale);
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        const double apq = matrix[p * n + q];
        if (std::abs(apq) <= tolerance) continue;
        const double app = matrix[p * n + p], aqq = matrix[q * n + q];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle), sine = std::sin(angle);
        for (std::size_t k = 0; k < n; ++k) {
          if (k == p || k == q) continue;
          const double mkp = matrix[k * n + p], mkq = matrix[k * n + q];
          matrix[k * n + p] = matrix[p * n + k] = cosine * mkp - sine * mkq;
          matrix[k * n + q] = matrix[q * n + k] = sine * mkp + cosine * mkq;
        }
        matrix[p * n + p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q * n + q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p * n + q] = matrix[q * n + p] = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
          const double vkp = vectors[row * n + p], vkq = vectors[row * n + q];
          vectors[row * n + p] = cosine * vkp - sine * vkq;
          vectors[row * n + q] = sine * vkp + cosine * vkq;
        }
      }
    }
    double largest_off_diagonal = 0.0;
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t column = row + 1; column < n; ++column)
        largest_off_diagonal = std::max(largest_off_diagonal, std::abs(matrix[row * n + column]));
    converged = largest_off_diagonal <= tolerance;
  }
  if (!converged) throw std::runtime_error("CPU symmetric eigensolver did not converge");
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return matrix[a * n + a] < matrix[b * n + b]; });
  tensor::CpuSymmetricEigenResult result;
  result.values.resize(n);
  result.vectors.resize(matrix.size());
  for (std::size_t column = 0; column < n; ++column) {
    const std::size_t source = order[column];
    result.values[column] = matrix[source * n + source] * output_scale;
    if (!std::isfinite(result.values[column]))
      throw std::overflow_error("CPU symmetric eigenvalue exceeds finite FP64 range");
    for (std::size_t row = 0; row < n; ++row)
      result.vectors[row * n + column] = vectors[row * n + source];
  }
  return result;
}

// clang-format on
}  // namespace detail

inline bool bind_symmetric_eigen_work(const PreparedSymmetricEigen& prepared,
                                      const SymmetricEigenStorage& storage,
                                      SymmetricEigenWorkBinding& output) noexcept {
  const auto required = prepared.required();
  if (!prepared.ready() || prepared.family() != SymmetricEigenFamily::lapack_work_column_major ||
      storage.double_capacity < static_cast<std::size_t>(required.doubles) ||
      storage.integer_capacity < static_cast<std::size_t>(required.integers))
    return false;
  detail::Range work, integers;
  if (!detail::range(storage.work, required.doubles, sizeof(double), alignof(double), work) ||
      !detail::range(storage.integer_work, required.integers, sizeof(LapackInt), alignof(LapackInt),
                     integers) ||
      !detail::disjoint(work, integers))
    return false;
  output = {storage.work, storage.integer_work, required};
  return true;
}

struct BorrowedSymmetricEigenProblem {
  std::size_t n{};
  double* matrix{};
  std::size_t matrix_capacity{};
  double* values{};
  std::size_t value_capacity{};
};
enum class SymmetricEigenError { none, invalid_binding };
struct SymmetricEigenCallResult {
  SymmetricEigenError error{SymmetricEigenError::none};
  LapackInt info{};
  bool submitted() const noexcept { return error == SymmetricEigenError::none; }
};

/** The provider is already admitted and the owner's existing thread scope is
 * active. Return raw LAPACK info; never map method status or retry. */
namespace detail {
template <class Provider>
inline SymmetricEigenCallResult invoke_symmetric_eigen(
    const PreparedSymmetricEigen& prepared, Provider provider,
    const BorrowedSymmetricEigenProblem& problem,
    const SymmetricEigenWorkBinding& binding) noexcept {
  const auto invalid = SymmetricEigenCallResult{SymmetricEigenError::invalid_binding, 0};
  const auto required = prepared.required();
  if (!provider || !prepared.ready() ||
      prepared.family() != SymmetricEigenFamily::lapack_work_column_major || !problem.n ||
      problem.n > prepared.maximum_order() ||
      problem.n > std::numeric_limits<std::size_t>::max() / problem.n ||
      problem.matrix_capacity < problem.n * problem.n || problem.value_capacity < problem.n ||
      binding.counts.doubles != required.doubles || binding.counts.integers != required.integers)
    return invalid;
  detail::Range matrix, values, work, integers;
  if (!detail::range(problem.matrix, problem.n * problem.n, sizeof(double), alignof(double),
                     matrix) ||
      !detail::range(problem.values, problem.n, sizeof(double), alignof(double), values) ||
      !detail::range(binding.work, required.doubles, sizeof(double), alignof(double), work) ||
      !detail::range(binding.integer_work, required.integers, sizeof(LapackInt), alignof(LapackInt),
                     integers) ||
      !detail::disjoint(matrix, values) || !detail::disjoint(matrix, work) ||
      !detail::disjoint(matrix, integers) || !detail::disjoint(values, work) ||
      !detail::disjoint(values, integers) || !detail::disjoint(work, integers))
    return invalid;
  const auto n = static_cast<LapackInt>(problem.n);
  return {SymmetricEigenError::none,
          provider(102, 'V', 'L', n, problem.matrix, n, problem.values, binding.work,
                   required.doubles, binding.integer_work, required.integers)};
}

}  // namespace detail

inline SymmetricEigenCallResult execute_symmetric_eigen(
    const PreparedSymmetricEigen& prepared, DsyevdWork provider,
    const BorrowedSymmetricEigenProblem& problem,
    const SymmetricEigenWorkBinding& binding) noexcept {
  return detail::invoke_symmetric_eigen(prepared, provider, problem, binding);
}

inline SymmetricEigenCallResult execute_symmetric_eigen(
    const PreparedSymmetricEigen& prepared,
    ::generativeqc::tensor::cpu::SymmetricEigenBinding provider,
    const BorrowedSymmetricEigenProblem& problem,
    const SymmetricEigenWorkBinding& binding) noexcept {
  return detail::invoke_symmetric_eigen(prepared, provider, problem, binding);
}

/** The canonical facade has checked finite input and provider admission before
 * entering. Vector ownership is moved, never copied into an extra matrix. The
 * linked-LAPACKE callback retains its current allocation and thread-guard scope. */
inline tensor::CpuSymmetricEigenResult execute_symmetric_eigen(
    const PreparedSymmetricEigen& prepared, std::vector<double> matrix,
    OwnedLapack provider = nullptr, const void* context = nullptr) {
  const auto n = prepared.maximum_order();
  if (!prepared.ready() || n > std::numeric_limits<std::size_t>::max() / n ||
      matrix.size() != n * n)
    throw std::invalid_argument("invalid owned CPU symmetric eigen binding");
  if (prepared.family() == SymmetricEigenFamily::scalar_owned_row_major)
    return detail::scalar_symmetric_eigen(std::move(matrix), n, prepared.scalar_cap());
  if (prepared.family() == SymmetricEigenFamily::lapack_owned_row_major && provider)
    return provider(std::move(matrix), n, context);
  throw std::invalid_argument("invalid owned CPU symmetric eigen provider");
}
}  // namespace generativeqc::solver::cpu
