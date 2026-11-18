#pragma once

#include <cstddef>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(GQC_ORDERED_BINARY_FORCE_SCALAR)
#include <immintrin.h>
#define GQC_ORDERED_BINARY_AVX2 1
#else
#define GQC_ORDERED_BINARY_AVX2 0
#endif

namespace generativeqc::tensor::ordered_binary_contraction {

/** Runtime capability only; generated layout/alias/finite guards own admission. */
inline bool available() noexcept {
#if GQC_ORDERED_BINARY_AVX2
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

#if GQC_ORDERED_BINARY_AVX2
enum class LoadMode { broadcast, contiguous, strided };

template <LoadMode Mode>
__attribute__((target("avx2"))) inline __m256d load(const double* values, std::size_t offset,
                                                    std::size_t stride) noexcept {
  if constexpr (Mode == LoadMode::broadcast) return _mm256_set1_pd(values[offset]);
  if constexpr (Mode == LoadMode::contiguous) return _mm256_loadu_pd(values + offset);
  return _mm256_set_pd(values[offset + 3 * stride], values[offset + 2 * stride],
                       values[offset + stride], values[offset]);
}

/** Lanes are independent columns, not terms of a horizontal reduction. Each
 * entry retains the TensorIR operand order and serial reduction sequence.
 * Only exact-unit binary products qualify: moving a nonunit coefficient could
 * change intermediate rounding/overflow. No panels, FMA or BLAS are used.
 */
template <LoadMode LeftMode, LoadMode RightMode>
__attribute__((target("avx2"))) inline void unit(std::size_t rows, std::size_t columns,
                                                 std::size_t reductions, const double* left,
                                                 std::size_t left_row, std::size_t left_column,
                                                 std::size_t left_reduction, const double* right,
                                                 std::size_t right_row, std::size_t right_column,
                                                 std::size_t right_reduction,
                                                 double* output) noexcept {
  for (std::size_t row = 0; row < rows; ++row) {
    std::size_t column{};
    for (; columns - column >= 4; column += 4) {
      auto sum = _mm256_setzero_pd();
      const auto left_base = row * left_row + column * left_column;
      const auto right_base = row * right_row + column * right_column;
      for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
        const auto product = _mm256_mul_pd(
            load<LeftMode>(left, left_base + reduction * left_reduction, left_column),
            load<RightMode>(right, right_base + reduction * right_reduction, right_column));
        sum = _mm256_add_pd(sum, product);
      }
      _mm256_storeu_pd(output + row * columns + column, sum);
    }
    for (; column < columns; ++column) {
      double sum{};
      for (std::size_t reduction = 0; reduction < reductions; ++reduction)
        sum += left[row * left_row + column * left_column + reduction * left_reduction] *
               right[row * right_row + column * right_column + reduction * right_reduction];
      output[row * columns + column] = sum;
    }
  }
}

/** Reuse one operand across four independent output rows. The four vector
 * accumulators each retain the original reduction order; register blocking
 * shares loads, never reduction terms or caller-owned numeric storage.
 * ShareLeft requires left_row == 0, otherwise right_row must be zero.
 */
template <LoadMode LeftMode, LoadMode RightMode, bool ShareLeft>
__attribute__((target("avx2"))) inline void shared_rows(
    std::size_t rows, std::size_t columns, std::size_t reductions, const double* left,
    std::size_t left_row, std::size_t left_column, std::size_t left_reduction, const double* right,
    std::size_t right_row, std::size_t right_column, std::size_t right_reduction,
    double* output) noexcept {
  const auto* shared = ShareLeft ? left : right;
  const auto* separate = ShareLeft ? right : left;
  const auto shared_column = ShareLeft ? left_column : right_column;
  const auto shared_reduction = ShareLeft ? left_reduction : right_reduction;
  const auto separate_row = ShareLeft ? right_row : left_row;
  const auto separate_column = ShareLeft ? right_column : left_column;
  const auto separate_reduction = ShareLeft ? right_reduction : left_reduction;
  std::size_t row{};
  for (; rows - row >= 4; row += 4) {
    std::size_t column{};
    for (; columns - column >= 4; column += 4) {
      auto first_sum = _mm256_setzero_pd();
      auto second_sum = _mm256_setzero_pd();
      auto third_sum = _mm256_setzero_pd();
      auto fourth_sum = _mm256_setzero_pd();
      const auto shared_base = column * shared_column;
      const auto separate_base = row * separate_row + column * separate_column;
      for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
        const auto common = load<ShareLeft ? LeftMode : RightMode>(
            shared, shared_base + reduction * shared_reduction, shared_column);
        const auto first = load<ShareLeft ? RightMode : LeftMode>(
            separate, separate_base + reduction * separate_reduction, separate_column);
        const auto second = load<ShareLeft ? RightMode : LeftMode>(
            separate, separate_base + separate_row + reduction * separate_reduction,
            separate_column);
        const auto third = load<ShareLeft ? RightMode : LeftMode>(
            separate, separate_base + 2 * separate_row + reduction * separate_reduction,
            separate_column);
        const auto fourth = load<ShareLeft ? RightMode : LeftMode>(
            separate, separate_base + 3 * separate_row + reduction * separate_reduction,
            separate_column);
        if constexpr (ShareLeft) {
          first_sum = _mm256_add_pd(first_sum, _mm256_mul_pd(common, first));
          second_sum = _mm256_add_pd(second_sum, _mm256_mul_pd(common, second));
          third_sum = _mm256_add_pd(third_sum, _mm256_mul_pd(common, third));
          fourth_sum = _mm256_add_pd(fourth_sum, _mm256_mul_pd(common, fourth));
        } else {
          first_sum = _mm256_add_pd(first_sum, _mm256_mul_pd(first, common));
          second_sum = _mm256_add_pd(second_sum, _mm256_mul_pd(second, common));
          third_sum = _mm256_add_pd(third_sum, _mm256_mul_pd(third, common));
          fourth_sum = _mm256_add_pd(fourth_sum, _mm256_mul_pd(fourth, common));
        }
      }
      _mm256_storeu_pd(output + row * columns + column, first_sum);
      _mm256_storeu_pd(output + (row + 1) * columns + column, second_sum);
      _mm256_storeu_pd(output + (row + 2) * columns + column, third_sum);
      _mm256_storeu_pd(output + (row + 3) * columns + column, fourth_sum);
    }
    for (std::size_t offset = 0; offset < 4; ++offset)
      for (std::size_t tail = column; tail < columns; ++tail) {
        double sum{};
        for (std::size_t reduction = 0; reduction < reductions; ++reduction)
          sum +=
              left[(row + offset) * left_row + tail * left_column + reduction * left_reduction] *
              right[(row + offset) * right_row + tail * right_column + reduction * right_reduction];
        output[(row + offset) * columns + tail] = sum;
      }
  }
  if (row < rows)
    unit<LeftMode, RightMode>(rows - row, columns, reductions,
                              reductions ? left + row * left_row : left, left_row, left_column,
                              left_reduction, reductions ? right + row * right_row : right,
                              right_row, right_column, right_reduction, output + row * columns);
}

/** Only a real row-invariant operand selects register reuse. Small extents
 * and contractions without that property retain the incumbent row schedule.
 */
template <LoadMode LeftMode, LoadMode RightMode>
inline void selected_unit(std::size_t rows, std::size_t columns, std::size_t reductions,
                          const double* left, std::size_t left_row, std::size_t left_column,
                          std::size_t left_reduction, const double* right, std::size_t right_row,
                          std::size_t right_column, std::size_t right_reduction,
                          double* output) noexcept {
  if (rows >= 4 && columns >= 4) {
    if (!left_row) {
      shared_rows<LeftMode, RightMode, true>(rows, columns, reductions, left, left_row, left_column,
                                             left_reduction, right, right_row, right_column,
                                             right_reduction, output);
      return;
    }
    if (!right_row) {
      shared_rows<LeftMode, RightMode, false>(rows, columns, reductions, left, left_row,
                                              left_column, left_reduction, right, right_row,
                                              right_column, right_reduction, output);
      return;
    }
  }
  unit<LeftMode, RightMode>(rows, columns, reductions, left, left_row, left_column, left_reduction,
                            right, right_row, right_column, right_reduction, output);
}

/** Select invariant load modes outside the ordered reduction. */
template <LoadMode LeftMode>
inline void right_dispatch(std::size_t rows, std::size_t columns, std::size_t reductions,
                           const double* left, std::size_t left_row, std::size_t left_column,
                           std::size_t left_reduction, const double* right, std::size_t right_row,
                           std::size_t right_column, std::size_t right_reduction,
                           double* output) noexcept {
  if (!right_column)
    selected_unit<LeftMode, LoadMode::broadcast>(rows, columns, reductions, left, left_row,
                                                 left_column, left_reduction, right, right_row,
                                                 right_column, right_reduction, output);
  else if (right_column == 1)
    selected_unit<LeftMode, LoadMode::contiguous>(rows, columns, reductions, left, left_row,
                                                  left_column, left_reduction, right, right_row,
                                                  right_column, right_reduction, output);
  else
    selected_unit<LeftMode, LoadMode::strided>(rows, columns, reductions, left, left_row,
                                               left_column, left_reduction, right, right_row,
                                               right_column, right_reduction, output);
}
#endif

/** Execute an admitted row-major output in caller-owned scratch. Empty output
 * dimensions return before traversing another, potentially huge extent; empty
 * reductions publish zeros without reading either operand. The scalar fallback
 * also makes direct use safe on CPUs without AVX2. Compile with FP contraction
 * disabled, as for the serial host emitter.
 */
inline void apply(std::size_t rows, std::size_t columns, std::size_t reductions, const double* left,
                  std::size_t left_row, std::size_t left_column, std::size_t left_reduction,
                  const double* right, std::size_t right_row, std::size_t right_column,
                  std::size_t right_reduction, double* output) noexcept {
  if (!rows || !columns) return;
#if GQC_ORDERED_BINARY_AVX2
  if (available()) {
    if (!left_column)
      right_dispatch<LoadMode::broadcast>(rows, columns, reductions, left, left_row, left_column,
                                          left_reduction, right, right_row, right_column,
                                          right_reduction, output);
    else if (left_column == 1)
      right_dispatch<LoadMode::contiguous>(rows, columns, reductions, left, left_row, left_column,
                                           left_reduction, right, right_row, right_column,
                                           right_reduction, output);
    else
      right_dispatch<LoadMode::strided>(rows, columns, reductions, left, left_row, left_column,
                                        left_reduction, right, right_row, right_column,
                                        right_reduction, output);
    return;
  }
#endif
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < columns; ++column) {
      double sum{};
      for (std::size_t reduction = 0; reduction < reductions; ++reduction)
        sum += left[row * left_row + column * left_column + reduction * left_reduction] *
               right[row * right_row + column * right_column + reduction * right_reduction];
      output[row * columns + column] = sum;
    }
}
}  // namespace generativeqc::tensor::ordered_binary_contraction

#undef GQC_ORDERED_BINARY_AVX2
