#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
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

}  // namespace generativeqc::response
