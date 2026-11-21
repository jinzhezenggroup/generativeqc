#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace generativeqc::integrals {

/** Owned FP64 values for the eightfold orbits of real scalar Coulomb ERIs.
 * Arbitrary external tensors and nuclear derivative panels are not accepted.
 */
class PackedSphericalEri {
 public:
  static std::size_t required_bytes(std::size_t basis) {
    if (basis == 0) throw std::invalid_argument("packed ERI basis cannot be empty");
    const auto elements = checked_triangle(checked_triangle(basis));
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
      throw std::overflow_error("packed ERI byte count overflow");
    return elements * sizeof(double);
  }

  PackedSphericalEri(std::size_t basis, std::size_t maximum_bytes)
      : basis_(basis),
        count_(admit(basis, maximum_bytes)),
        values_(std::make_unique<double[]>(count_)) {}

  std::size_t basis() const noexcept { return basis_; }
  std::span<const double> values() const noexcept { return {values_.get(), count_}; }
  std::size_t capacity_bytes() const noexcept { return count_ * sizeof(double); }

  /** Access an orbit using four valid AO indices in the admitted basis. */
  double value(std::size_t first, std::size_t second, std::size_t third,
               std::size_t fourth) const noexcept {
    return values_[pair(pair(first, second), pair(third, fourth))];
  }

  /** Publish one representative, as produced by the canonical shell visitor. */
  void store(std::size_t first, std::size_t second, std::size_t third, std::size_t fourth,
             double value) noexcept {
    values_[pair(pair(first, second), pair(third, fourth))] = value;
  }

 private:
  static std::size_t checked_triangle(std::size_t count) {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (count == maximum) throw std::overflow_error("packed ERI triangular count overflow");
    const auto first = count % 2 == 0 ? count / 2 : count;
    const auto second = count % 2 == 0 ? count + 1 : (count + 1) / 2;
    if (first != 0 && second > maximum / first)
      throw std::overflow_error("packed ERI triangular count overflow");
    return first * second;
  }

  static std::size_t admit(std::size_t basis, std::size_t maximum_bytes) {
    const auto bytes = required_bytes(basis);
    if (bytes > maximum_bytes) throw std::length_error("packed spherical ERI capacity exceeded");
    return bytes / sizeof(double);
  }

  static std::size_t pair(std::size_t first, std::size_t second) noexcept {
    const auto [lower, upper] = std::minmax(first, second);
    return (upper % 2 == 0 ? (upper / 2) * (upper + 1) : upper * ((upper + 1) / 2)) + lower;
  }

  std::size_t basis_;
  /** The array has exactly the admitted numeric extent, with no growth capacity. */
  std::size_t count_;
  std::unique_ptr<double[]> values_;
};

}  // namespace generativeqc::integrals
