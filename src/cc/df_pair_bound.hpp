#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace generativeqc::cc::pair_bound {

static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::radix == 2 &&
              std::numeric_limits<double>::digits == 53 &&
              std::numeric_limits<double>::max_exponent == 1024);

// This is admission metadata, not a physical value or an alternate CC algebra.
// Every bounded value denotes a nonnegative real upper bound 2^exponent;
// zero is exact and refusal must never become a physical arithmetic failure.
struct Power {
  enum class State : unsigned char { zero, bounded, refused };
  State state{State::zero};
  int exponent{};

  static constexpr Power zero() { return {}; }
  static constexpr Power refused() { return {State::refused, 0}; }
  static constexpr Power of(int exponent) {
    return exponent >= -8192 && exponent <= 8192 ? Power{State::bounded, exponent} : refused();
  }
};

inline constexpr std::uint64_t magnitude_mask = 0x7fffffffffffffffULL;
inline constexpr std::uint64_t infinity_bits = 0x7ff0000000000000ULL;

inline std::uint64_t magnitude_bits(double value) {
  return std::bit_cast<std::uint64_t>(value) & magnitude_mask;
}

inline int floor_exponent(std::uint64_t magnitude) {
  const auto encoded = static_cast<int>(magnitude >> 52);
  return encoded ? encoded - 1023 : static_cast<int>(std::bit_width(magnitude)) - 1 - 1074;
}

inline Power magnitude_upper(std::uint64_t magnitude) {
  if (magnitude >= infinity_bits) return Power::refused();
  if (!magnitude) return Power::zero();
  const auto encoded = magnitude >> 52;
  const bool exact_power =
      encoded ? !(magnitude & 0xfffffffffffffULL) : std::has_single_bit(magnitude);
  return Power::of(floor_exponent(magnitude) + !exact_power);
}

inline Power extent_upper(std::size_t extent) {
  return extent ? Power::of(static_cast<int>(std::bit_width(extent - 1))) : Power::zero();
}

inline Power product(std::initializer_list<Power> values) {
  bool zero = false;
  int exponent = 0;
  for (const auto value : values) {
    if (value.state == Power::State::refused) return Power::refused();
    zero = zero || value.state == Power::State::zero;
    if (value.exponent < -8192 || value.exponent > 8192) return Power::refused();
    exponent += value.exponent;
    if (exponent < -8192 || exponent > 8192) return Power::refused();
  }
  return zero ? Power::zero() : Power::of(exponent);
}

inline Power maximum(Power first, Power second) {
  if (first.state == Power::State::refused || second.state == Power::State::refused)
    return Power::refused();
  if (first.state == Power::State::zero) return second;
  if (second.state == Power::State::zero) return first;
  return Power::of(std::max(first.exponent, second.exponent));
}

inline Power sum(std::initializer_list<Power> values) {
  auto upper = Power::zero();
  std::size_t nonzero = 0;
  for (const auto value : values) {
    upper = maximum(upper, value);
    nonzero += value.state != Power::State::zero;
  }
  return product({upper, extent_upper(nonzero)});
}

// A global exponent supplies a fixed integer quantum. Each entry's upper
// power is rounded upward to that quantum, including subnormal/tiny entries.
// Grouped integer sums are substantially tighter than N*max(abs(matrix)) and
// cannot silently flush a positive contribution to zero.
inline Power row_l1_upper(const double* values, std::size_t rows, std::size_t columns,
                          unsigned summed_axes, std::size_t& inspected) {
  if (summed_axes & ~3U) return Power::refused();
  if (rows && columns > std::numeric_limits<std::size_t>::max() / rows) return Power::refused();
  const auto count = rows * columns;
  if ((count && !values) || count > (std::numeric_limits<std::size_t>::max() - inspected) / 2)
    return Power::refused();
  std::uint64_t largest = 0;
  for (std::size_t index = 0; index < count; ++index) {
    largest = std::max(largest, magnitude_bits(values[index]));
    ++inspected;
  }
  const auto reference = magnitude_upper(largest);
  if (reference.state != Power::State::bounded || !summed_axes) return reference;
  const auto quantum = reference.exponent - 32;
  const auto groups = summed_axes == 1 ? columns : summed_axes == 2 ? rows : 1;
  const auto terms = summed_axes == 1 ? rows : summed_axes == 2 ? columns : count;
  std::uint64_t largest_units = 0;
  for (std::size_t group = 0; group < groups; ++group) {
    std::uint64_t units = 0;
    for (std::size_t term = 0; term < terms; ++term) {
      const auto index = summed_axes == 1   ? term * columns + group
                         : summed_axes == 2 ? group * columns + term
                                            : term;
      const auto entry = magnitude_upper(magnitude_bits(values[index]));
      ++inspected;
      if (entry.state == Power::State::refused) return Power::refused();
      if (entry.state == Power::State::zero) continue;
      const auto shift = entry.exponent - quantum;
      const auto contribution = shift < 0 ? 1ULL : 1ULL << shift;
      if (contribution > std::numeric_limits<std::uint64_t>::max() - units) return Power::refused();
      units += contribution;
    }
    largest_units = std::max(largest_units, units);
  }
  return product({Power::of(quantum), extent_upper(largest_units)});
}

inline bool within_residual_margin(Power error, double tolerance) {
  const auto encoded = std::bit_cast<std::uint64_t>(tolerance);
  if (!encoded || encoded >= infinity_bits || error.state == Power::State::refused) return false;
  if (error.state == Power::State::zero) return true;
  // A lower power of tolerance/8 remains meaningful even below DBL_TRUE_MIN.
  return error.exponent <= floor_exponent(encoded) - 3;
}

struct ProjectionMaxima {
  std::uint64_t tau_error_bits{}, t1_magnitude_bits{};
  unsigned refused{};
};

#if defined(GENERATIVEQC_CUDA_PROVIDER_CUMETAL) && GENERATIVEQC_CUDA_PROVIDER_CUMETAL
inline constexpr bool projection_supported = false;
#else
inline constexpr bool projection_supported = true;
#endif

}  // namespace generativeqc::cc::pair_bound
