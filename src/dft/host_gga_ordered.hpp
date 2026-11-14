#pragma once

#include <cstddef>
#include <span>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define GQC_HOST_GGA_ORDERED_AVX2 1
#else
#define GQC_HOST_GGA_ORDERED_AVX2 0
#endif

namespace generativeqc::dft::host_gga_ordered {

/** Scalar control keeps the incumbent point/row/column arithmetic verbatim. */
inline void scalar_fields(std::size_t basis, std::size_t count, std::size_t spins, const double* ao,
                          const double* density, const double* direction, double* primal,
                          double* tangent) {
  for (std::size_t spin = 0; spin < spins; ++spin)
    for (std::size_t point = 0; point < count; ++point) {
      double fields[4]{}, delta[4]{};
      for (std::size_t row = 0; row < basis; ++row)
        for (std::size_t column = 0; column < basis; ++column) {
          const auto index = (spin * basis + row) * basis + column;
          const auto left = ao[point * basis + row];
          const auto right = ao[point * basis + column];
          const auto product = left * right;
          fields[0] += density[index] * product;
          delta[0] += direction[index] * product;
          for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto gradient = ao[((axis + 1) * count + point) * basis + row] * right +
                                  left * ao[((axis + 1) * count + point) * basis + column];
            fields[axis + 1] += density[index] * gradient;
            delta[axis + 1] += direction[index] * gradient;
          }
        }
      for (std::size_t component = 0; component < 4; ++component) {
        primal[(4 * spin + component) * count + point] = fields[component];
        tangent[(4 * spin + component) * count + point] = delta[component];
      }
    }
}

inline void scalar_pullback(std::size_t basis, std::size_t count, std::size_t spins,
                            const double* ao, const double* coefficients, const double* weights,
                            double* result) {
  for (std::size_t point = 0; point < count; ++point)
    for (std::size_t spin = 0; spin < spins; ++spin)
      for (std::size_t row = 0; row < basis; ++row)
        for (std::size_t column = 0; column < basis; ++column) {
          const auto left = ao[point * basis + row];
          const auto right = ao[point * basis + column];
          double value = coefficients[4 * spin * count + point] * left * right;
          for (std::size_t axis = 0; axis < 3; ++axis)
            value += coefficients[(4 * spin + axis + 1) * count + point] *
                     (ao[((axis + 1) * count + point) * basis + row] * right +
                      left * ao[((axis + 1) * count + point) * basis + column]);
          result[(spin * basis + row) * basis + column] += weights[point] * value;
        }
}

#if GQC_HOST_GGA_ORDERED_AVX2
/** Four lanes are rho/x/y/z, NOT four terms of a reduction. Ordered AO pairs
 * and multiply/add grouping remain exact, including signed nonsymmetric Ds.
 * The caller owns only four doubles per AO of packing scratch. No FMA, BLAS,
 * horizontal sum, altered vacuum branch, or cross-point reassociation is used.
 */
__attribute__((target("avx2"))) inline void packed_fields_avx2(std::size_t basis, std::size_t count,
                                                               std::size_t spins, const double* ao,
                                                               const double* density,
                                                               const double* direction,
                                                               double* primal, double* tangent,
                                                               double* packed) {
  for (std::size_t point = 0; point < count; ++point) {
    for (std::size_t row = 0; row < basis; ++row)
      for (std::size_t component = 0; component < 4; ++component)
        packed[4 * row + component] = ao[(component * count + point) * basis + row];
    for (std::size_t spin = 0; spin < spins; ++spin) {
      auto fields = _mm256_setzero_pd();
      auto delta = _mm256_setzero_pd();
      for (std::size_t row = 0; row < basis; ++row) {
        const auto left = _mm256_loadu_pd(packed + 4 * row);
        const auto value_left = _mm256_set1_pd(packed[4 * row]);
        for (std::size_t column = 0; column < basis; ++column) {
          const auto right = _mm256_loadu_pd(packed + 4 * column);
          const auto value_right = _mm256_set1_pd(packed[4 * column]);
          const auto first = _mm256_mul_pd(left, value_right);
          const auto gradient = _mm256_add_pd(first, _mm256_mul_pd(value_left, right));
          const auto product = _mm256_blend_pd(gradient, first, 1);
          const auto index = (spin * basis + row) * basis + column;
          fields = _mm256_add_pd(fields, _mm256_mul_pd(_mm256_set1_pd(density[index]), product));
          delta = _mm256_add_pd(delta, _mm256_mul_pd(_mm256_set1_pd(direction[index]), product));
        }
      }
      double field_values[4], delta_values[4];
      _mm256_storeu_pd(field_values, fields);
      _mm256_storeu_pd(delta_values, delta);
      for (std::size_t component = 0; component < 4; ++component) {
        primal[(4 * spin + component) * count + point] = field_values[component];
        tangent[(4 * spin + component) * count + point] = delta_values[component];
      }
    }
  }
}

/** SIMD lanes are output columns. Each entry visits points and gradient
 * components in incumbent order; the scalar tail covers non-multiple sizes.
 */
__attribute__((target("avx2"))) inline void packed_pullback_avx2(
    std::size_t basis, std::size_t count, std::size_t spins, const double* ao,
    const double* coefficients, const double* weights, double* result) {
  for (std::size_t point = 0; point < count; ++point) {
    const auto* values = ao + point * basis;
    const auto* gradient_x = ao + (count + point) * basis;
    const auto* gradient_y = ao + (2 * count + point) * basis;
    const auto* gradient_z = ao + (3 * count + point) * basis;
    const auto weight = _mm256_set1_pd(weights[point]);
    for (std::size_t spin = 0; spin < spins; ++spin) {
      const auto coefficient = coefficients[4 * spin * count + point];
      const auto coefficient_x = _mm256_set1_pd(coefficients[(4 * spin + 1) * count + point]);
      const auto coefficient_y = _mm256_set1_pd(coefficients[(4 * spin + 2) * count + point]);
      const auto coefficient_z = _mm256_set1_pd(coefficients[(4 * spin + 3) * count + point]);
      for (std::size_t row = 0; row < basis; ++row) {
        const auto value_row = _mm256_set1_pd(values[row]);
        const auto leading = _mm256_set1_pd(coefficient * values[row]);
        const auto row_x = _mm256_set1_pd(gradient_x[row]);
        const auto row_y = _mm256_set1_pd(gradient_y[row]);
        const auto row_z = _mm256_set1_pd(gradient_z[row]);
        auto* output = result + (spin * basis + row) * basis;
        std::size_t column = 0;
        for (; column + 4 <= basis; column += 4) {
          const auto value_column = _mm256_loadu_pd(values + column);
          auto value = _mm256_mul_pd(leading, value_column);
          const auto product_x =
              _mm256_add_pd(_mm256_mul_pd(row_x, value_column),
                            _mm256_mul_pd(value_row, _mm256_loadu_pd(gradient_x + column)));
          value = _mm256_add_pd(value, _mm256_mul_pd(coefficient_x, product_x));
          const auto product_y =
              _mm256_add_pd(_mm256_mul_pd(row_y, value_column),
                            _mm256_mul_pd(value_row, _mm256_loadu_pd(gradient_y + column)));
          value = _mm256_add_pd(value, _mm256_mul_pd(coefficient_y, product_y));
          const auto product_z =
              _mm256_add_pd(_mm256_mul_pd(row_z, value_column),
                            _mm256_mul_pd(value_row, _mm256_loadu_pd(gradient_z + column)));
          value = _mm256_add_pd(value, _mm256_mul_pd(coefficient_z, product_z));
          _mm256_storeu_pd(output + column, _mm256_add_pd(_mm256_loadu_pd(output + column),
                                                          _mm256_mul_pd(weight, value)));
        }
        for (; column < basis; ++column) {
          double value = coefficient * values[row] * values[column];
          for (std::size_t axis = 0; axis < 3; ++axis)
            value += coefficients[(4 * spin + axis + 1) * count + point] *
                     (ao[((axis + 1) * count + point) * basis + row] * values[column] +
                      values[row] * ao[((axis + 1) * count + point) * basis + column]);
          output[column] += weights[point] * value;
        }
      }
    }
  }
}
#endif

inline bool packed_available() {
#if GQC_HOST_GGA_ORDERED_AVX2
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

/** Reuse a not-yet-live panel in the caller's admitted arena. Small tiles or
 * unsupported CPUs retain the scalar schedule without allocation or new caps.
 * AO/input/output/packing panels must be disjoint, except D may equal dD.
 * Compile this owner with strict FP and contraction disabled.
 */
inline void fields(std::size_t basis, std::size_t count, std::size_t spins, const double* ao,
                   const double* density, const double* direction, double* primal, double* tangent,
                   std::span<double> packing) {
#if GQC_HOST_GGA_ORDERED_AVX2
  if (packed_available() && packing.size() / 4 >= basis) {
    packed_fields_avx2(basis, count, spins, ao, density, direction, primal, tangent,
                       packing.data());
    return;
  }
#endif
  (void)packing;
  scalar_fields(basis, count, spins, ao, density, direction, primal, tangent);
}

inline void pullback(std::size_t basis, std::size_t count, std::size_t spins, const double* ao,
                     const double* coefficients, const double* weights, double* result) {
#if GQC_HOST_GGA_ORDERED_AVX2
  if (packed_available()) {
    packed_pullback_avx2(basis, count, spins, ao, coefficients, weights, result);
    return;
  }
#endif
  scalar_pullback(basis, count, spins, ao, coefficients, weights, result);
}

/** Batched inputs are [spin,AO-pair,input], with fixed D first. Output is
 * [input,spin,field,point]. Each independent field retains ordered AO pairs;
 * only the invariant pair products are shared. No density symmetry is assumed.
 */
inline void scalar_batch_fields(std::size_t basis, std::size_t count, std::size_t spins,
                                std::size_t inputs, const double* ao, const double* densities,
                                double* fields) {
  for (std::size_t spin = 0; spin < spins; ++spin)
    for (std::size_t point = 0; point < count; ++point) {
      double sums[5][4]{};
      for (std::size_t row = 0; row < basis; ++row)
        for (std::size_t column = 0; column < basis; ++column) {
          const auto left = ao[point * basis + row];
          const auto right = ao[point * basis + column];
          double products[4]{left * right};
          for (std::size_t axis = 0; axis < 3; ++axis)
            products[axis + 1] = ao[((axis + 1) * count + point) * basis + row] * right +
                                 left * ao[((axis + 1) * count + point) * basis + column];
          const auto index = ((spin * basis + row) * basis + column) * inputs;
          for (std::size_t input = 0; input < inputs; ++input)
            for (std::size_t component = 0; component < 4; ++component)
              sums[input][component] += densities[index + input] * products[component];
        }
      for (std::size_t input = 0; input < inputs; ++input)
        for (std::size_t component = 0; component < 4; ++component)
          fields[((input * spins + spin) * 4 + component) * count + point] = sums[input][component];
    }
}

/** Coefficients are [spin,field,point,direction], output is
 * [spin,AO-pair,direction]. Point and component reduction orders are unchanged.
 */
inline void scalar_batch_pullback(std::size_t basis, std::size_t count, std::size_t spins,
                                  std::size_t directions, const double* ao,
                                  const double* coefficients, const double* weights,
                                  double* result) {
  for (std::size_t point = 0; point < count; ++point)
    for (std::size_t spin = 0; spin < spins; ++spin)
      for (std::size_t row = 0; row < basis; ++row)
        for (std::size_t column = 0; column < basis; ++column) {
          const auto left = ao[point * basis + row];
          const auto right = ao[point * basis + column];
          double products[3]{};
          for (std::size_t axis = 0; axis < 3; ++axis)
            products[axis] = ao[((axis + 1) * count + point) * basis + row] * right +
                             left * ao[((axis + 1) * count + point) * basis + column];
          for (std::size_t direction = 0; direction < directions; ++direction) {
            double value =
                coefficients[(4 * spin * count + point) * directions + direction] * left * right;
            for (std::size_t axis = 0; axis < 3; ++axis)
              value +=
                  coefficients[((4 * spin + axis + 1) * count + point) * directions + direction] *
                  products[axis];
            result[((spin * basis + row) * basis + column) * directions + direction] +=
                weights[point] * value;
          }
        }
}

#if GQC_HOST_GGA_ORDERED_AVX2
/** SIMD lanes remain rho/x/y/z, never summands or coupled spin channels. */
template <unsigned Inputs>
__attribute__((target("avx2"))) inline void packed_batch_fields_avx2(
    std::size_t basis, std::size_t count, std::size_t spins, const double* ao,
    const double* densities, double* fields, double* packed) {
  for (std::size_t point = 0; point < count; ++point) {
    for (std::size_t row = 0; row < basis; ++row)
      for (std::size_t component = 0; component < 4; ++component)
        packed[4 * row + component] = ao[(component * count + point) * basis + row];
    for (std::size_t spin = 0; spin < spins; ++spin) {
      __m256d sums[Inputs];
      for (auto& sum : sums) sum = _mm256_setzero_pd();
      for (std::size_t row = 0; row < basis; ++row) {
        const auto left = _mm256_loadu_pd(packed + 4 * row);
        const auto value_left = _mm256_set1_pd(packed[4 * row]);
        for (std::size_t column = 0; column < basis; ++column) {
          const auto right = _mm256_loadu_pd(packed + 4 * column);
          const auto value_right = _mm256_set1_pd(packed[4 * column]);
          const auto first = _mm256_mul_pd(left, value_right);
          const auto gradient = _mm256_add_pd(first, _mm256_mul_pd(value_left, right));
          const auto product = _mm256_blend_pd(gradient, first, 1);
          const auto index = ((spin * basis + row) * basis + column) * Inputs;
          for (unsigned input = 0; input < Inputs; ++input)
            sums[input] = _mm256_add_pd(
                sums[input], _mm256_mul_pd(_mm256_set1_pd(densities[index + input]), product));
        }
      }
      for (unsigned input = 0; input < Inputs; ++input) {
        double values[4];
        _mm256_storeu_pd(values, sums[input]);
        for (std::size_t component = 0; component < 4; ++component)
          fields[((input * spins + spin) * 4 + component) * count + point] = values[component];
      }
    }
  }
}

/** SIMD lanes are independent directions. Masked tails never read or write
 * beyond a two/three-direction panel; ordered point sums stay in their lanes.
 */
template <unsigned Directions>
__attribute__((target("avx2"))) inline void packed_batch_pullback_avx2(
    std::size_t basis, std::size_t count, std::size_t spins, const double* ao,
    const double* coefficients, const double* weights, double* result) {
  const auto mask = _mm256_set_epi64x(Directions > 3 ? -1LL : 0, Directions > 2 ? -1LL : 0,
                                      Directions > 1 ? -1LL : 0, -1LL);
  for (std::size_t point = 0; point < count; ++point)
    for (std::size_t spin = 0; spin < spins; ++spin) {
      __m256d coefficient[4];
      for (std::size_t component = 0; component < 4; ++component) {
        const auto* source = coefficients + ((4 * spin + component) * count + point) * Directions;
        if constexpr (Directions == 4)
          coefficient[component] = _mm256_loadu_pd(source);
        else
          coefficient[component] = _mm256_maskload_pd(source, mask);
      }
      const auto weight = _mm256_set1_pd(weights[point]);
      for (std::size_t row = 0; row < basis; ++row) {
        const auto left = ao[point * basis + row];
        const auto leading = _mm256_mul_pd(coefficient[0], _mm256_set1_pd(left));
        for (std::size_t column = 0; column < basis; ++column) {
          const auto right = ao[point * basis + column];
          auto value = _mm256_mul_pd(leading, _mm256_set1_pd(right));
          for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto product = ao[((axis + 1) * count + point) * basis + row] * right +
                                 left * ao[((axis + 1) * count + point) * basis + column];
            value =
                _mm256_add_pd(value, _mm256_mul_pd(coefficient[axis + 1], _mm256_set1_pd(product)));
          }
          auto* destination = result + ((spin * basis + row) * basis + column) * Directions;
          if constexpr (Directions == 4)
            _mm256_storeu_pd(destination, _mm256_add_pd(_mm256_loadu_pd(destination),
                                                        _mm256_mul_pd(weight, value)));
          else
            _mm256_maskstore_pd(
                destination, mask,
                _mm256_add_pd(_mm256_maskload_pd(destination, mask), _mm256_mul_pd(weight, value)));
        }
      }
    }
}
#endif

/** Bounded execution templates; callers validate 3--5 inputs / 2--4 directions
 * and own disjoint packing/output panels. Scalar implementations are real
 * portable fallbacks, not alternate scientific operators.
 */
inline void batch_fields(std::size_t basis, std::size_t count, std::size_t spins,
                         std::size_t inputs, const double* ao, const double* densities,
                         double* fields, std::span<double> packing) {
#if GQC_HOST_GGA_ORDERED_AVX2
  if (packed_available() && packing.size() / 4 >= basis) {
    if (inputs == 3)
      packed_batch_fields_avx2<3>(basis, count, spins, ao, densities, fields, packing.data());
    else if (inputs == 4)
      packed_batch_fields_avx2<4>(basis, count, spins, ao, densities, fields, packing.data());
    else
      packed_batch_fields_avx2<5>(basis, count, spins, ao, densities, fields, packing.data());
    return;
  }
#endif
  (void)packing;
  scalar_batch_fields(basis, count, spins, inputs, ao, densities, fields);
}

inline void batch_pullback(std::size_t basis, std::size_t count, std::size_t spins,
                           std::size_t directions, const double* ao, const double* coefficients,
                           const double* weights, double* result) {
#if GQC_HOST_GGA_ORDERED_AVX2
  if (packed_available()) {
    if (directions == 2)
      packed_batch_pullback_avx2<2>(basis, count, spins, ao, coefficients, weights, result);
    else if (directions == 3)
      packed_batch_pullback_avx2<3>(basis, count, spins, ao, coefficients, weights, result);
    else
      packed_batch_pullback_avx2<4>(basis, count, spins, ao, coefficients, weights, result);
    return;
  }
#endif
  scalar_batch_pullback(basis, count, spins, directions, ao, coefficients, weights, result);
}

}  // namespace generativeqc::dft::host_gga_ordered

#undef GQC_HOST_GGA_ORDERED_AVX2
