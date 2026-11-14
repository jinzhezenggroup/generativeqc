#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "tensor/ordered_binary_contraction.hpp"

namespace {
double sample(std::size_t index, std::size_t pattern) {
  constexpr std::array<double, 8> edges{0.0,
                                        -0.0,
                                        std::numeric_limits<double>::denorm_min(),
                                        -std::numeric_limits<double>::denorm_min(),
                                        1.0,
                                        -1.0,
                                        0x1.0000000000001p0,
                                        -0x1.0000000000001p0};
  if (pattern == 0) return edges[index % edges.size()];
  const auto numerator = static_cast<int>((index * 17 + pattern * 11) % 31) - 15;
  const auto exponent = pattern == 1 ? 0 : static_cast<int>((index * 97) % 901) - 450;
  return std::ldexp(static_cast<double>(numerator) / 16, exponent);
}

/** Compare each column-stride/row-sharing layout with the coefficient-first
 * serial sum. Guard values cover row/column tails, including shared operands;
 * huge empty shapes must not traverse or allocate anything. These are execution
 * controls, not a molecular oracle.
 */
bool check(std::size_t rows, std::size_t columns, std::size_t reductions, std::size_t left_column,
           std::size_t right_column, std::size_t pattern, bool share_left, bool share_right) {
  using generativeqc::tensor::ordered_binary_contraction::apply;
  if (!rows || !columns) {
    apply(rows, columns, reductions, nullptr, 0, left_column, 0, nullptr, 0, right_column, 0,
          nullptr);
    return true;
  }
  const std::size_t left_reduction = 2, right_reduction = 5;
  const auto left_row =
      share_left ? 0 : columns * (left_column + 1) + reductions * left_reduction + 19;
  const auto right_row =
      share_right ? 0 : columns * (right_column + 1) + reductions * right_reduction + 23;
  const auto count = [&](std::size_t row_stride, std::size_t column_stride,
                         std::size_t reduction_stride) {
    return reductions ? (rows - 1) * row_stride + (columns - 1) * column_stride +
                            (reductions - 1) * reduction_stride + 1
                      : 0;
  };
  std::vector<double> left(count(left_row, left_column, left_reduction));
  std::vector<double> right(count(right_row, right_column, right_reduction));
  for (std::size_t index = 0; index < left.size(); ++index) left[index] = sample(index, pattern);
  for (std::size_t index = 0; index < right.size(); ++index)
    right[index] = sample(index + 13, pattern);
  std::vector<double> expected(rows * columns + 8, -789.25), actual = expected;
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < columns; ++column) {
      double sum{};
      for (std::size_t reduction = 0; reduction < reductions; ++reduction)
        sum += 1.0 * left[row * left_row + column * left_column + reduction * left_reduction] *
               right[row * right_row + column * right_column + reduction * right_reduction];
      expected[4 + row * columns + column] = sum;
    }
  apply(rows, columns, reductions, left.data(), left_row, left_column, left_reduction, right.data(),
        right_row, right_column, right_reduction, actual.data() + 4);
  for (std::size_t index = 0; index < actual.size(); ++index)
    if (std::bit_cast<std::uint64_t>(actual[index]) !=
        std::bit_cast<std::uint64_t>(expected[index]))
      return false;
  return true;
}
}  // namespace

extern "C" int ordered_binary_qualification() {
  constexpr auto huge = std::numeric_limits<std::size_t>::max();
  constexpr std::array<std::array<std::size_t, 3>, 12> shapes{{{0, 0, 0},
                                                               {0, huge, huge},
                                                               {huge, 0, huge},
                                                               {4, 9, 0},
                                                               {5, 4, 0},
                                                               {7, 9, 0},
                                                               {1, 1, 1},
                                                               {3, 7, 17},
                                                               {4, 4, 5},
                                                               {8, 8, 9},
                                                               {7, 19, 23},
                                                               {17, 81, 257}}};
  int cases{};
  for (const auto& shape : shapes)
    for (const auto left_column : {0u, 1u, 7u})
      for (const auto right_column : {0u, 1u, 9u})
        for (std::size_t pattern = 0; pattern < 3; ++pattern)
          for (const auto share_left : {false, true})
            for (const auto share_right : {false, true}) {
              if (!check(shape[0], shape[1], shape[2], left_column, right_column, pattern,
                         share_left, share_right))
                return -1;
              ++cases;
            }
  return cases;
}

#if defined(GQC_ORDERED_BINARY_STANDALONE)
int main() { return ordered_binary_qualification() > 0 ? 0 : 1; }
#endif
