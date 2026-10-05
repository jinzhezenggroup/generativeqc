#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/cpu_linalg.hpp"

namespace generativeqc::response {

/** Bounded numerical inverse of D + U^T U, with U in rank-by-dimension order.
 *
 * This class owns no physical response equation. It is a right preconditioner:
 * callers must still qualify their solution with the original physical action.
 * Positive D and successful Cholesky are admission conditions, never clipping
 * policies. All conversion/solve scratch is retained and included in capacity.
 * The shared scalar linear-algebra provider keeps this small host solver free
 * of unreported external-provider workspace; it is not a physical CPU fallback.
 */
class LowRankPreconditioner {
 public:
  LowRankPreconditioner(LowRankPreconditioner&&) = default;
  LowRankPreconditioner& operator=(LowRankPreconditioner&&) = default;
  LowRankPreconditioner(const LowRankPreconditioner&) = delete;
  LowRankPreconditioner& operator=(const LowRankPreconditioner&) = delete;
  static std::size_t capacity_bytes(std::size_t n, std::size_t rank) {
    auto elements = checked_add(checked_mul(rank, n), checked_mul(rank, rank));
    elements = checked_add(elements, checked_add(checked_mul(2, n), rank));
    return checked_mul(elements, sizeof(double));
  }

  /** Refuse unsafe numerics or a short budget before publishing an inverse.
   * Shape/extent errors are programming errors and throw. Input spans remain
   * borrowed, so the caller must charge them while setup copies coexist.
   */
  static std::optional<LowRankPreconditioner> prepare(std::span<const double> diagonal,
                                                      std::span<const double> u, std::size_t rank,
                                                      std::size_t maximum_bytes,
                                                      double minimum_diagonal = 1e-10) {
    const auto n = diagonal.size();
    if (!n || !rank || u.size() != checked_mul(rank, n) || !(minimum_diagonal > 0.0) ||
        !std::isfinite(minimum_diagonal))
      throw std::invalid_argument("invalid low-rank preconditioner dimensions or threshold");
    if (capacity_bytes(n, rank) > maximum_bytes) return std::nullopt;
    for (double value : diagonal)
      if (!std::isfinite(value) || value <= minimum_diagonal) return std::nullopt;
    for (double value : u)
      if (!std::isfinite(value)) return std::nullopt;
    LowRankPreconditioner result;
    result.n_ = n;
    result.rank_ = rank;
    result.inverse_root_.resize(n);
    result.scaled_u_.resize(u.size());
    result.cholesky_.assign(checked_mul(rank, rank), 0.0);
    result.vector_.resize(n);
    result.reduced_.resize(rank);
    for (std::size_t i = 0; i < n; ++i) result.inverse_root_[i] = 1.0 / std::sqrt(diagonal[i]);
    for (std::size_t q = 0; q < rank; ++q)
      for (std::size_t i = 0; i < n; ++i) {
        const auto value = u[q * n + i] * result.inverse_root_[i];
        if (!std::isfinite(value)) return std::nullopt;
        result.scaled_u_[q * n + i] = value;
      }
    for (std::size_t q = 0; q < rank; ++q) result.cholesky_[q * rank + q] = 1.0;
    // S = I + (U D^-1/2)(U D^-1/2)^T. Only the lower triangle is
    // constructed/read, and the original U never requires another copy.
    tensor::cpu_syrk('L', 'N', rank, n, result.scaled_u_.data(), result.cholesky_.data(), 1.0, 1.0,
                     scalar_plan());
    for (std::size_t i = 0; i < rank; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        if (!std::isfinite(result.cholesky_[i * rank + j])) return std::nullopt;
    if (tensor::cpu_cholesky_lower(result.cholesky_.data(), rank, scalar_plan()) != 0)
      return std::nullopt;
    if (result.owned_bytes() > maximum_bytes) return std::nullopt;
    return std::optional<LowRankPreconditioner>(std::move(result));
  }

  [[nodiscard]] std::size_t dimension() const noexcept { return n_; }
  [[nodiscard]] std::size_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::size_t owned_bytes() const {
    std::size_t result = 0;
    for (const auto* values : {&inverse_root_, &scaled_u_, &cholesky_, &vector_, &reduced_})
      result = checked_add(result, checked_mul(values->capacity(), sizeof(double)));
    return result;
  }

  /** Apply Woodbury with two triangular solves. Scratch makes this owner
   * deliberately non-reentrant; separate simultaneous solves need separate
   * owners. Input/output spans may alias because input is staged first.
   */
  void apply(std::span<const double> input, std::span<double> output) {
    if (input.size() != n_ || output.size() != n_)
      throw std::invalid_argument("low-rank preconditioner vector shape mismatch");
    for (std::size_t i = 0; i < n_; ++i) vector_[i] = inverse_root_[i] * input[i];
    tensor::cpu_gemv('N', rank_, n_, scaled_u_.data(), vector_.data(), reduced_.data(), 1.0, 0.0,
                     scalar_plan());
    tensor::cpu_trsm('L', 'L', 'N', 'N', rank_, 1, cholesky_.data(), reduced_.data(), 1.0,
                     scalar_plan());
    tensor::cpu_trsm('L', 'L', 'T', 'N', rank_, 1, cholesky_.data(), reduced_.data(), 1.0,
                     scalar_plan());
    tensor::cpu_gemv('T', rank_, n_, scaled_u_.data(), reduced_.data(), vector_.data(), -1.0, 1.0,
                     scalar_plan());
    for (std::size_t i = 0; i < n_; ++i) {
      output[i] = inverse_root_[i] * vector_[i];
      if (!std::isfinite(output[i]))
        throw std::runtime_error("nonfinite low-rank preconditioner result");
    }
  }

 private:
  // Numerical workspace admission is method-neutral. Keep the signed pointer
  // extent bound without importing molecular/post-HF resource policy here.
  static std::size_t checked_add(std::size_t a, std::size_t b) {
    constexpr auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (a > limit || b > limit - a) throw std::overflow_error("low-rank workspace overflow");
    return a + b;
  }
  static std::size_t checked_mul(std::size_t a, std::size_t b) {
    constexpr auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (a && b > limit / a) throw std::overflow_error("low-rank workspace overflow");
    return a * b;
  }
  LowRankPreconditioner() = default;
  static tensor::CpuLinalgPlan scalar_plan() {
    tensor::CpuLinalgPlan plan;
    plan.provider = tensor::CpuLinalgProvider::scalar;
    return plan;
  }
  std::size_t n_{}, rank_{};
  std::vector<double> inverse_root_, scaled_u_, cholesky_, vector_, reduced_;
};

}  // namespace generativeqc::response
