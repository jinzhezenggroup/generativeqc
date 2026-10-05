#pragma once

#include <cstddef>
#include <stdexcept>

namespace generativeqc::solver {

/** Chronological metadata for caller-owned, fixed-capacity history rows.
 *
 * Neither insertion nor retirement moves a stored vector. A Gram matrix indexed
 * by physical row therefore keeps all old-old entries valid until a row is
 * overwritten. The owner must fill the returned slot before consuming it.
 * This class has no residual, coefficient, precision or backend policy.
 */
class DiisRing {
 public:
  explicit DiisRing(std::size_t capacity) : capacity_(capacity) {}
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t first() const noexcept { return first_; }

  [[nodiscard]] std::size_t slot(std::size_t chronological) const {
    if (chronological >= size_) throw std::out_of_range("DIIS history row is not live");
    return physical(chronological);
  }

  std::size_t push() {
    if (!capacity_) throw std::invalid_argument("cannot append to disabled DIIS history");
    if (size_ == capacity_) retire_oldest();
    return physical(size_++);
  }

  void retire_oldest() noexcept {
    if (!size_) return;
    first_ = first_ + 1 == capacity_ ? 0 : first_ + 1;
    --size_;
  }

  void clear() noexcept { first_ = size_ = 0; }

 private:
  [[nodiscard]] std::size_t physical(std::size_t chronological) const noexcept {
    // Avoid an overflowing first+chronological even for a generic large ring.
    const auto tail = capacity_ - first_;
    return chronological < tail ? first_ + chronological : chronological - tail;
  }
  std::size_t capacity_{}, first_{}, size_{};
};

}  // namespace generativeqc::solver
