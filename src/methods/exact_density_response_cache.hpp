#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <vector>

namespace generativeqc::methods::detail {

/** Optional contraction storage inside one immutable operator/geometry owner.
 *
 * The caller authenticates the current final-state token before lookup. This
 * key compares every density bit, including signed zero; an epoch, pointer or
 * tolerance is never an equality proof. Only density-dependent two-electron
 * response belongs here. Pulay response depends on W and must run separately.
 * One exactly sized allocation holds the key and contracted response. Failure
 * to retain it is an ordinary cache miss, never a failure of the contraction.
 */
class ExactDensityResponseCache {
 public:
  using Blocks = std::vector<std::vector<double>>;

  void clear() noexcept {
    storage_.reset();
    spins_ = matrix_ = response_ = 0;
  }

  std::size_t bytes() const noexcept {
    return storage_ ? (spins_ * matrix_ + response_) * sizeof(double) : 0;
  }

  bool matches(const Blocks& density, std::size_t response_size,
               std::size_t maximum_bytes) const noexcept {
    if (!storage_ || bytes() > maximum_bytes || density.size() != spins_ ||
        response_size != response_)
      return false;
    for (std::size_t spin = 0; spin < spins_; ++spin)
      if (density[spin].size() != matrix_ ||
          std::memcmp(density[spin].data(), storage_.get() + spin * matrix_,
                      matrix_ * sizeof(double)) != 0)
        return false;
    return true;
  }

  std::span<const double> response() const noexcept {
    return storage_ ? std::span(storage_.get() + spins_ * matrix_, response_)
                    : std::span<const double>{};
  }

  /** Retain a successful finite result, or release the old key on refusal.
   * Clearing before allocation bounds peak storage to the one requested slab.
   */
  bool store(const Blocks& density, std::span<const double> response,
             std::size_t maximum_bytes) noexcept {
    clear();
    if (density.empty() || density.front().empty() || response.empty()) return false;
    const auto matrix = density.front().size();
    const auto capacity = maximum_bytes / sizeof(double);
    if (response.size() > capacity || density.size() > (capacity - response.size()) / matrix)
      return false;
    const auto finite = [](double value) { return std::isfinite(value); };
    for (const auto& block : density)
      if (block.size() != matrix || !std::all_of(block.begin(), block.end(), finite)) return false;
    if (!std::all_of(response.begin(), response.end(), finite)) return false;
    const auto elements = density.size() * matrix + response.size();
    try {
      auto storage = std::make_unique<double[]>(elements);
      for (std::size_t spin = 0; spin < density.size(); ++spin)
        std::copy(density[spin].begin(), density[spin].end(), storage.get() + spin * matrix);
      std::copy(response.begin(), response.end(), storage.get() + density.size() * matrix);
      storage_ = std::move(storage);
      spins_ = density.size();
      matrix_ = matrix;
      response_ = response.size();
      return true;
    } catch (const std::bad_alloc&) {
      return false;
    }
  }

 private:
  std::unique_ptr<double[]> storage_;
  std::size_t spins_{}, matrix_{}, response_{};
};

}  // namespace generativeqc::methods::detail
