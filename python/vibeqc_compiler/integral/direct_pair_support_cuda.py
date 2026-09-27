"""Compiler-owned emission of Direct pair and Hermite support.

The emitted headers preserve the qualified low-order pair expansions and exact
shell-pair Hermite workspace while removing maintained native CUDA formula owners.
"""

from __future__ import annotations

_SOURCES = {
    "generated_direct_pair_order2.cuh": r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned Direct pair/Hermite lowering.
// Do not edit this build artifact: change
// python/vibeqc_compiler/integral/direct_pair_support_cuda.py instead.
// Arithmetic/workspace order is preserved; host plans and queue policy remain native.
namespace vibeqc::scf::cuda_execution {

/** One nonzero three-dimensional Hermite coefficient through order two. */
template <typename Scalar>
struct LowOrderHermiteTerm {
  // Each Cartesian derivative occupies two bits. Adding two states therefore
  // combines pair derivatives without carrying between x, y, and z.
  unsigned derivative_state;
  Scalar coefficient;
};

/** Compact shell-pair expansion; total order two reaches at most four terms. */
template <typename Scalar>
struct LowOrderPairExpansion {
  LowOrderHermiteTerm<Scalar> terms[4];
};

__device__ inline unsigned low_order_derivative_state(int axis) { return 1U << (2 * axis); }

__device__ inline unsigned low_order_derivative_total(unsigned state) {
  return (state & 3U) + ((state >> 2U) & 3U) + ((state >> 4U) & 3U);
}

/**
 * Generate only the nonzero Hermite terms of one order-0/1/2 shell pair.
 *
 * The Gaussian pair decay is deliberately excluded and applied once by the
 * primitive quartet. At order two, retaining duplicate first-derivative terms
 * for a repeated axis keeps one runtime component path for d and p-p AOs while
 * still bounding the expansion at four entries.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, typename Scalar>
__device__ inline LowOrderPairExpansion<Scalar> make_low_order_pair_expansion(
    double exponent, const Vec3<Scalar>& product, const Vec3<Scalar>& first,
    const Angular& angular_first, const Vec3<Scalar>& second, const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 2);
  LowOrderPairExpansion<Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[2];
    Scalar shifts[2];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = low_order_derivative_state(axis);
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        ++quantum_count;
      }
    }

    expansion.terms[0] = {0U, shifts[0]};
    expansion.terms[1] = {derivative_states[0], scalar<Scalar>(inverse_two_exponent)};
    if constexpr (PairOrder == 2) {
      // A repeated Cartesian axis contributes the recurrence's +1/(2p)
      // correction. The two first-derivative entries then share a state and
      // sum to the exact E1 coefficient during contraction.
      const double repeated_axis_correction =
          derivative_states[0] == derivative_states[1] ? inverse_two_exponent : 0.0;
      expansion.terms[0].coefficient =
          shifts[0] * shifts[1] + scalar<Scalar>(repeated_axis_correction);
      expansion.terms[1].coefficient = inverse_two_exponent * shifts[1];
      expansion.terms[2] = {derivative_states[1], inverse_two_exponent * shifts[0]};
      expansion.terms[3] = {derivative_states[0] + derivative_states[1],
                            scalar<Scalar>(inverse_two_exponent * inverse_two_exponent)};
    }
  }
  return expansion;
}

}  // namespace vibeqc::scf::cuda_execution
""",
    "generated_direct_pair_order3.cuh": r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "generated_direct_eri_order2.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "generated_direct_pair_order2.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned Direct pair/Hermite lowering.
// Do not edit this build artifact: change
// python/vibeqc_compiler/integral/direct_pair_support_cuda.py instead.
// Arithmetic/workspace order is preserved; host plans and queue policy remain native.
namespace vibeqc::scf::cuda_execution {

/** Exact-sized sparse pair expansion used only by total-order-3 quartets. */
template <unsigned PairOrder, typename Scalar>
struct ThirdOrderPairExpansion {
  static_assert(PairOrder <= 3);
  LowOrderHermiteTerm<Scalar> terms[1U << PairOrder];
};

/**
 * Generate a shell pair through order three from its angular quanta.
 *
 * The base expansion is the product of one first-order factor per quantum.
 * Two quanta on the same Cartesian axis additionally have one Gaussian Wick
 * contraction, 1/(2p). Through order three, adding that contraction to the
 * surviving base terms produces the complete Hermite expansion while keeping
 * the exact 1/2/4/8-term bound.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, typename Scalar>
__device__ inline ThirdOrderPairExpansion<FirstShellAngular + SecondShellAngular, Scalar>
make_third_order_pair_expansion(double exponent, const Vec3<Scalar>& product,
                                const Vec3<Scalar>& first, const Angular& angular_first,
                                const Vec3<Scalar>& second, const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 3);
  constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  ThirdOrderPairExpansion<PairOrder, Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[QuantumStorage];
    Scalar shifts[QuantumStorage];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = low_order_derivative_state(axis);
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        ++quantum_count;
      }
    }

    for (unsigned subset = 0; subset < (1U << PairOrder); ++subset) {
      unsigned derivative_state = 0;
      Scalar coefficient = scalar<Scalar>(1.0);
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if ((subset & (1U << quantum)) != 0) {
          derivative_state += derivative_states[quantum];
          coefficient = inverse_two_exponent * coefficient;
        } else {
          coefficient = coefficient * shifts[quantum];
        }
      }
      expansion.terms[subset] = {derivative_state, coefficient};
    }

    if constexpr (PairOrder == 2) {
      if (derivative_states[0] == derivative_states[1]) {
        expansion.terms[0].coefficient =
            expansion.terms[0].coefficient + scalar<Scalar>(inverse_two_exponent);
      }
    } else if constexpr (PairOrder == 3) {
      for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 3; ++second_quantum) {
          if (derivative_states[first_quantum] != derivative_states[second_quantum]) {
            continue;
          }
          const unsigned remaining_quantum = 3U - first_quantum - second_quantum;
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient + inverse_two_exponent * shifts[remaining_quantum];
          const unsigned surviving_derivative = 1U << remaining_quantum;
          expansion.terms[surviving_derivative].coefficient =
              expansion.terms[surviving_derivative].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most three. */
template <typename Scalar>
__device__ inline Scalar third_order_coulomb(unsigned derivative_state, double rho,
                                             const Vec3<Scalar>& product_difference,
                                             const Scalar* boys) {
  const unsigned x_order = derivative_state & 3U;
  const unsigned y_order = (derivative_state >> 2U) & 3U;
  const unsigned z_order = (derivative_state >> 4U) & 3U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 3) {
    return low_order_coulomb(derivative_state, rho, product_difference, boys);
  }

  const double third_order_factor = -8.0 * rho * rho * rho;
  if (x_order == 3 || y_order == 3 || z_order == 3) {
    const Scalar coordinate = x_order == 3
                                  ? product_difference.x
                                  : (y_order == 3 ? product_difference.y : product_difference.z);
    return third_order_factor * coordinate * coordinate * coordinate * boys[3] +
           (12.0 * rho * rho) * coordinate * boys[2];
  }

  if (x_order == 2 || y_order == 2 || z_order == 2) {
    const Scalar repeated_coordinate =
        x_order == 2 ? product_difference.x
                     : (y_order == 2 ? product_difference.y : product_difference.z);
    const Scalar single_coordinate =
        x_order == 1 ? product_difference.x
                     : (y_order == 1 ? product_difference.y : product_difference.z);
    return third_order_factor * repeated_coordinate * repeated_coordinate * single_coordinate *
               boys[3] +
           (4.0 * rho * rho) * single_coordinate * boys[2];
  }

  return third_order_factor * product_difference.x * product_difference.y * product_difference.z *
         boys[3];
}

}  // namespace vibeqc::scf::cuda_execution
""",
    "generated_direct_shell_pair_hermite.cuh": r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned Direct pair/Hermite lowering.
// Do not edit this build artifact: change
// python/vibeqc_compiler/integral/direct_pair_support_cuda.py instead.
// Arithmetic/workspace order is preserved; host plans and queue policy remain native.
namespace vibeqc::scf::cuda_execution {

/** Hermite workspace bounded by one exact shell-pair class. */
template <typename Scalar, unsigned FirstAngular, unsigned SecondAngular>
struct ShellPairHermiteCoefficients {
  static constexpr unsigned kIDimension = FirstAngular + 1;
  static constexpr unsigned kJDimension = SecondAngular + 1;
  // One zero boundary element is required because the recurrence reads t+1.
  static constexpr unsigned kTDimension = FirstAngular + SecondAngular + 2;
  Scalar data[kIDimension * kJDimension * kTDimension];

  __device__ inline Scalar& at(unsigned i, unsigned j, unsigned t) {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
  __device__ inline const Scalar& at(unsigned i, unsigned j, unsigned t) const {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
};

template <unsigned FirstAngular, unsigned SecondAngular, typename Scalar>
__device__ inline void fill_shell_pair_hermite(
    unsigned maximum_i, unsigned maximum_j, Scalar product, Scalar center_a, Scalar center_b,
    double alpha, double beta,
    ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>& coefficients) {
  static_assert(FirstAngular <= kMaximumAngularMomentum);
  static_assert(SecondAngular <= kMaximumAngularMomentum);
  for (unsigned item = 0;
       item < ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>::kIDimension *
                  ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>::kJDimension *
                  ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>::kTDimension;
       ++item) {
    coefficients.data[item] = scalar<Scalar>(0.0);
  }
  const double p = alpha + beta;
  const double mu = alpha * beta / p;
  const Scalar ab = center_a - center_b;
  coefficients.at(0, 0, 0) = qexp(-mu * ab * ab);
  const Scalar pa = product - center_a;
  const Scalar pb = product - center_b;
  const double inverse_two_p = 0.5 / p;

  for (unsigned i = 0; i <= maximum_i; ++i) {
    for (unsigned j = 0; j <= maximum_j; ++j) {
      if (i == 0 && j == 0) continue;
      if (i > 0) {
        coefficients.at(i, j, 0) = pa * coefficients.at(i - 1, j, 0) + coefficients.at(i - 1, j, 1);
      } else {
        coefficients.at(i, j, 0) = pb * coefficients.at(i, j - 1, 0) + coefficients.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coefficients.at(i, j, t) = pa * coefficients.at(i - 1, j, t) +
                                     inverse_two_p * coefficients.at(i - 1, j, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i - 1, j, t + 1);
        } else {
          coefficients.at(i, j, t) = pb * coefficients.at(i, j - 1, t) +
                                     inverse_two_p * coefficients.at(i, j - 1, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i, j - 1, t + 1);
        }
      }
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution
""",
}


def emit_direct_pair_support_headers() -> dict[str, str]:
    """Return generated Direct pair/Hermite headers keyed by output filename."""

    return dict(_SOURCES)
