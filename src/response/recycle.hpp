#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "response/solve.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::response {

/** Borrowed FP64 columns, including reversed, repeated and noncontiguous views.
 * Storage starts at the lowest declared address; origin identifies logical (0,0).
 * Metadata is checked before allocation/read. No pointer or array is retained.
 */
struct ResponseColumns {
  std::span<const double> storage;
  std::size_t rows{}, columns{}, origin{};
  std::int64_t row_stride{}, column_stride{};

  static std::size_t magnitude(std::int64_t stride) {
    const auto value = stride < 0 ? static_cast<std::uint64_t>(-(stride + 1)) + 1
                                  : static_cast<std::uint64_t>(stride);
    if (value > std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("recycle stride exceeds native size_t");
    return static_cast<std::size_t>(value);
  }

  void validate(std::size_t dimension) const {
    if (rows != dimension) throw std::invalid_argument("recycle column dimension mismatch");
    (void)detail::response_checked_multiply(storage.size(), sizeof(double));
    if (!columns) return;
    const auto row_extent = detail::response_checked_multiply(rows - 1, magnitude(row_stride));
    const auto column_extent =
        detail::response_checked_multiply(columns - 1, magnitude(column_stride));
    const auto negative = detail::response_checked_add(row_stride < 0 ? row_extent : 0,
                                                       column_stride < 0 ? column_extent : 0);
    const auto positive = detail::response_checked_add(row_stride >= 0 ? row_extent : 0,
                                                       column_stride >= 0 ? column_extent : 0);
    if (origin < negative || detail::response_checked_add(origin, positive) >= storage.size())
      throw std::invalid_argument("recycle columns exceed declared storage");
  }

  [[nodiscard]] double at(std::size_t row, std::size_t column) const {
    auto index = origin;
    if (row_stride < 0) index -= row * magnitude(row_stride);
    if (column_stride < 0) index -= column * magnitude(column_stride);
    if (row_stride >= 0) index += row * magnitude(row_stride);
    if (column_stride >= 0) index += column * magnitude(column_stride);
    return storage[index];
  }
};

struct ResponseRecycleCapacity {
  std::size_t dimension{}, vector_capacity{}, retained_bytes{};
  std::size_t projection_workspace_bytes{}, update_workspace_bytes{};
};

/** Conservative additional numeric workspace; existing retained bytes are separate.
 * Projection covers its native temporary and caller publication. Update covers
 * a complete replacement plus bounded working/publication temporaries, never a
 * panel containing every old/solved Arnoldi candidate. Object overhead is separate.
 */
inline ResponseRecycleCapacity response_recycle_capacity(std::size_t dimension,
                                                         std::size_t max_vectors,
                                                         std::size_t max_retained_bytes) {
  if (!dimension || !max_vectors || !max_retained_bytes)
    throw std::invalid_argument("recycle dimension and limits must be positive");
  const auto vector_bytes = detail::response_checked_multiply(dimension, sizeof(double));
  const auto capacity = std::min({dimension, max_vectors, max_retained_bytes / vector_bytes});
  return {
      dimension, capacity, detail::response_checked_multiply(capacity, vector_bytes),
      detail::response_checked_multiply(2, vector_bytes),
      detail::response_checked_multiply(detail::response_checked_add(capacity, 3), vector_bytes)};
}

class RecycleWorkspaceLimit : public std::runtime_error {
 public:
  RecycleWorkspaceLimit() : std::runtime_error("recycle numeric workspace limit exceeded") {}
};

/** Native CPU retained-vector owner, not an operator or physical-reference owner.
 * Method adapters retain compatibility/lease validation. Generation checks reject
 * stale mutations/publications of this vector state. Updates preserve old-first,
 * twice-reorthogonalized MGS and the existing absolute/relative 1e-12 cutoff.
 * Results are copied out; no borrowed native vector can survive replacement.
 */
class ResponseRecycleSpace {
 public:
  ResponseRecycleSpace(std::size_t dimension, std::size_t max_vectors,
                       std::size_t max_retained_bytes)
      : capacity_(response_recycle_capacity(dimension, max_vectors, max_retained_bytes)),
        counter_(std::make_shared<runtime::AllocationCounter>()) {}

  ResponseRecycleSpace(const ResponseRecycleSpace&) = delete;
  ResponseRecycleSpace& operator=(const ResponseRecycleSpace&) = delete;
  ResponseRecycleSpace(ResponseRecycleSpace&&) = default;
  ResponseRecycleSpace& operator=(ResponseRecycleSpace&&) = default;

  [[nodiscard]] const ResponseRecycleCapacity& capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t vector_count() const noexcept { return vectors_.size(); }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] std::size_t storage_bytes() const noexcept {
    return vectors_.size() * capacity_.dimension * sizeof(double);
  }
  [[nodiscard]] runtime::AllocationSnapshot allocations() const { return counter_->snapshot(); }

  void check_generation(std::uint64_t expected) const {
    if (expected != generation_) throw std::invalid_argument("stale recycle vector generation");
  }

  [[nodiscard]] runtime::TrackedVector<double> project(std::uint64_t expected,
                                                       const ResponseColumns& rhs,
                                                       std::size_t max_workspace_bytes) const {
    check_generation(expected);
    rhs.validate(capacity_.dimension);
    if (rhs.columns != 1) throw std::invalid_argument("recycle projection requires one RHS");
    if (max_workspace_bytes < capacity_.projection_workspace_bytes) throw RecycleWorkspaceLimit();
    for (std::size_t row = 0; row < capacity_.dimension; ++row)
      if (!std::isfinite(rhs.at(row, 0))) throw std::invalid_argument("recycle RHS must be finite");
    auto guess = work_vector();
    for (const auto& vector : vectors_) {
      double coefficient = 0.0;
      if (rhs.row_stride == 1) {
        tensor::cpu_gemv('N', 1, capacity_.dimension, vector.data(),
                         rhs.storage.data() + rhs.origin, &coefficient);
      } else {
        for (std::size_t row = 0; row < capacity_.dimension; ++row)
          coefficient += vector[row] * rhs.at(row, 0);
      }
      for (std::size_t row = 0; row < capacity_.dimension; ++row)
        guess[row] += coefficient * vector[row];
    }
    for (const auto value : guess)
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite recycle projection");
    return guess;
  }

  /** Publish replacement only after every visited candidate passes its checks.
   * Unvisited candidate values are not read after capacity is filled. Failed
   * attempts preserve vectors/generation, but allocation telemetry records work.
   */
  void update(std::uint64_t expected, const ResponseColumns& solution, const ResponseColumns& basis,
              std::size_t max_vectors, std::size_t max_retained_bytes,
              std::size_t max_workspace_bytes) {
    check_generation(expected);
    const auto next_capacity =
        response_recycle_capacity(capacity_.dimension, max_vectors, max_retained_bytes);
    solution.validate(capacity_.dimension);
    basis.validate(capacity_.dimension);
    if (solution.columns != 1) throw std::invalid_argument("recycle update requires one solution");
    if (max_workspace_bytes < next_capacity.update_workspace_bytes) throw RecycleWorkspaceLimit();
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("recycle generation overflow");
    std::vector<runtime::TrackedVector<double>> replacement;
    replacement.reserve(next_capacity.vector_capacity);
    const auto offer = [&](runtime::TrackedVector<double> work) {
      const auto original_norm = stable_norm(work);
      for (unsigned pass = 0; pass < 2; ++pass) {
        for (const auto& prior : replacement) {
          double coefficient = 0.0;
          tensor::cpu_gemv('N', 1, capacity_.dimension, prior.data(), work.data(), &coefficient);
          for (std::size_t row = 0; row < capacity_.dimension; ++row)
            work[row] -= coefficient * prior[row];
        }
      }
      const auto norm = stable_norm(work);
      if (norm > 1e-12 * std::max(1.0, original_norm)) {
        for (auto& value : work) value /= norm;
        replacement.push_back(std::move(work));
      }
    };
    for (const auto& vector : vectors_) {
      if (replacement.size() == next_capacity.vector_capacity) break;
      offer(vector);
    }
    const auto offer_columns = [&](const ResponseColumns& columns) {
      for (std::size_t column = 0; column < columns.columns; ++column) {
        if (replacement.size() == next_capacity.vector_capacity) break;
        auto work = work_vector();
        for (std::size_t row = 0; row < capacity_.dimension; ++row)
          work[row] = columns.at(row, column);
        offer(std::move(work));
      }
    };
    offer_columns(solution);
    offer_columns(basis);
    vectors_.swap(replacement);
    capacity_ = next_capacity;
    ++generation_;
  }

  /** Detached column-major diagnostic publication; not used by solver updates. */
  void copy(std::uint64_t expected, std::size_t first_column, std::size_t column_count,
            std::span<double> output) const {
    check_generation(expected);
    if (first_column > vectors_.size() || column_count > vectors_.size() - first_column ||
        output.size() != column_count * capacity_.dimension)
      throw std::invalid_argument("recycle publication size mismatch");
    for (std::size_t column = 0; column < column_count; ++column)
      std::copy(vectors_[first_column + column].begin(), vectors_[first_column + column].end(),
                output.begin() + column * capacity_.dimension);
  }

 private:
  [[nodiscard]] runtime::TrackedVector<double> work_vector() const {
    return runtime::TrackedVector<double>(capacity_.dimension, 0.0,
                                          runtime::TrackedAllocator<double>(counter_));
  }

  ResponseRecycleCapacity capacity_;
  std::shared_ptr<runtime::AllocationCounter> counter_;
  std::vector<runtime::TrackedVector<double>> vectors_;
  std::uint64_t generation_{};
};

/** Shape-only capacity of one recycle group. Retained data never crosses the
 * group boundary. The operator's own scratch is reserved separately.
 */
struct RecycledBatchResources {
  GmresPlan solve;
  ResponseRecycleCapacity recycle;
  std::size_t retained_solution_bytes{}, retained_diagnostic_bytes{}, peak_bytes{};
  bool admitted{};
};

inline RecycledBatchResources recycled_batch_resources(std::size_t dimension,
                                                       const GmresOptions& options,
                                                       std::size_t count,
                                                       std::size_t max_numeric_bytes) {
  using detail::response_checked_add;
  using detail::response_checked_multiply;
  auto internal = options;
  internal.collect_basis = options.collect_basis || count > 1;
  RecycledBatchResources result;
  result.solve = prepare_gmres(dimension, internal);
  const auto vector_bytes = response_checked_multiply(dimension, sizeof(double));
  const auto history_bytes =
      options.collect_history ? response_checked_multiply(
                                    response_checked_add(options.max_iterations, 2), sizeof(double))
                              : 0;
  const auto basis_bytes =
      options.collect_basis ? response_checked_multiply(result.solve.restart, vector_bytes) : 0;
  const auto diagnostic_bytes = response_checked_add(history_bytes, basis_bytes);
  result.retained_solution_bytes = response_checked_multiply(count, vector_bytes);
  result.retained_diagnostic_bytes = response_checked_multiply(count, diagnostic_bytes);
  if (count) {
    result.peak_bytes = response_checked_add(
        result.solve.workspace_bytes,
        response_checked_multiply(count - 1, response_checked_add(vector_bytes, diagnostic_bytes)));
  }
  if (count > 1) {
    result.recycle = response_recycle_capacity(dimension, 8, 8U << 20);
    // During projection/solve/update, the old recycle space remains live.
    // Recycle scratch includes the projected guess or a complete replacement;
    // those do not overlap each other. Arnoldi storage is in the GMRES bound.
    result.peak_bytes = response_checked_add(
        result.peak_bytes, response_checked_add(result.recycle.retained_bytes,
                                                std::max(result.recycle.projection_workspace_bytes,
                                                         result.recycle.update_workspace_bytes)));
  }
  result.admitted =
      result.peak_bytes <= max_numeric_bytes && result.peak_bytes <= options.max_workspace_bytes;
  return result;
}

/** One native sequential solve group with the existing recycle vector policy.
 * Recycle vectors are guesses, never a substitute for GMRES or a residual
 * check. Validated caller RHS/preconditioners are borrowed synchronously.
 */
inline ResponseBatchResult solve_recycled_response_batch(
    const LinearResponseProblem& problem, const GmresOptions& options,
    std::span<const ResponseSolveRequest> requests, std::size_t max_numeric_bytes) {
  const auto dimension = problem.dimension();
  const auto capacity =
      recycled_batch_resources(dimension, options, requests.size(), max_numeric_bytes);
  for (const auto& request : requests) {
    if (request.rhs.size() != dimension || !request.initial_guess.empty())
      throw std::invalid_argument(
          "recycled response requires shaped RHS and owns the initial guess");
  }
  if (!capacity.admitted) return {{}, ResponseBatchStatus::workspace_limit, capacity.peak_bytes, 0};
  ResponseBatchResult result;
  result.planned_peak_numeric_bytes = capacity.peak_bytes;
  result.retained_solution_bytes = capacity.retained_solution_bytes;
  result.retained_diagnostic_bytes = capacity.retained_diagnostic_bytes;
  result.responses.reserve(requests.size());
  if (requests.empty()) return result;
  const auto plan = prepare_response(problem, capacity.solve.options);
  std::optional<ResponseRecycleSpace> recycle;
  if (requests.size() > 1 && capacity.recycle.vector_capacity)
    recycle.emplace(dimension, 8, 8U << 20);
  for (const auto& request : requests) {
    auto solved = [&] {
      const auto guess =
          (recycle && recycle->vector_count())
              ? recycle->project(recycle->generation(), {request.rhs, dimension, 1, 0, 1, 0},
                                 capacity.recycle.projection_workspace_bytes)
              : runtime::TrackedVector<double>{};
      return solve_response(plan, problem, request.rhs, guess, request.diagonal_preconditioner,
                            request.right_preconditioner);
    }();
    if (recycle && solved.converged() && result.responses.size() + 1 < requests.size()) {
      recycle->update(recycle->generation(), {solved.solution, dimension, 1, 0, 1, 0},
                      {solved.basis, dimension, solved.basis_columns, 0, 1,
                       static_cast<std::int64_t>(dimension)},
                      8, 8U << 20, capacity.recycle.update_workspace_bytes);
    }
    if (!options.collect_basis) {
      runtime::TrackedVector<double>{}.swap(solved.basis);
      solved.basis_columns = 0;
    }
    result.responses.push_back(std::move(solved));
  }
  return result;
}

}  // namespace generativeqc::response
