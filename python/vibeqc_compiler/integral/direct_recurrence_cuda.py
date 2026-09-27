"""Compiler-owned emission of Direct primitive recurrence support.

The emitted headers preserve the qualified order-2/3/4 recurrence and shell-class
dispatch arithmetic while removing maintained native CUDA formula owners. Runtime
queues, screening, pair helpers and selector policy remain outside this module.
"""

from __future__ import annotations

_SOURCES = {
    "generated_direct_eri_order2.cuh": r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "generated_direct_pair_order2.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned Direct recurrence lowering.
// Do not edit this build artifact: change
// python/vibeqc_compiler/integral/direct_recurrence_cuda.py instead.
// Arithmetic/workspace order is preserved; host plans and queue policy remain native.
namespace vibeqc::scf::cuda_execution {

/** Evaluate a Cartesian Coulomb derivative of total order at most two. */
template <typename Scalar>
__device__ inline Scalar low_order_coulomb(unsigned derivative_state, double rho,
                                           const Vec3<Scalar>& product_difference,
                                           const Scalar* boys) {
  const unsigned x_order = derivative_state & 3U;
  const unsigned y_order = (derivative_state >> 2U) & 3U;
  const unsigned z_order = (derivative_state >> 4U) & 3U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order == 0) return boys[0];

  if (total_order == 1) {
    const Scalar coordinate = x_order != 0
                                  ? product_difference.x
                                  : (y_order != 0 ? product_difference.y : product_difference.z);
    return (-2.0 * rho) * coordinate * boys[1];
  }

  const double second_order_factor = 4.0 * rho * rho;
  if (x_order == 2 || y_order == 2 || z_order == 2) {
    const Scalar coordinate = x_order == 2
                                  ? product_difference.x
                                  : (y_order == 2 ? product_difference.y : product_difference.z);
    return second_order_factor * coordinate * coordinate * boys[2] - (2.0 * rho) * boys[1];
  }

  Scalar coordinate_product = scalar<Scalar>(1.0);
  if (x_order != 0) coordinate_product = coordinate_product * product_difference.x;
  if (y_order != 0) coordinate_product = coordinate_product * product_difference.y;
  if (z_order != 0) coordinate_product = coordinate_product * product_difference.z;
  return second_order_factor * coordinate_product * boys[2];
}

/** Ten unique Cartesian Coulomb states through total order two. */
struct Order2CoulombValues {
  double c0;
  double cx;
  double cy;
  double cz;
  double cxx;
  double cxy;
  double cxz;
  double cyy;
  double cyz;
  double czz;
};

/** Build the complete order-two Coulomb tensor once per primitive quartet. */
__device__ __forceinline__ Order2CoulombValues order2_coulomb_values(double rho, double x, double y,
                                                                     double z,
                                                                     const double (&boys)[3]) {
  const double twice_rho = 2.0 * rho;
  const double twice_rho_squared = twice_rho * twice_rho;
  return {
      boys[0],
      -twice_rho * x * boys[1],
      -twice_rho * y * boys[1],
      -twice_rho * z * boys[1],
      twice_rho_squared * x * x * boys[2] - twice_rho * boys[1],
      twice_rho_squared * x * y * boys[2],
      twice_rho_squared * x * z * boys[2],
      twice_rho_squared * y * y * boys[2] - twice_rho * boys[1],
      twice_rho_squared * y * z * boys[2],
      twice_rho_squared * z * z * boys[2] - twice_rho * boys[1],
  };
}

/**
 * Closed order-2 contraction for canonical (d s|s s), (p p|s s), and
 * (p s|p s) primitive quartets.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ inline Scalar primitive_eri_order2(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder = FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder = ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 2);
  static_assert(FirstPairOrder <= 2 && SecondPairOrder <= 2);
  constexpr unsigned FirstTermCount = FirstPairOrder == 0 ? 1 : (FirstPairOrder == 1 ? 2 : 4);
  constexpr unsigned SecondTermCount = SecondPairOrder == 0 ? 1 : (SecondPairOrder == 1 ? 2 : 4);

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const LowOrderPairExpansion<Scalar> first_expansion =
      make_low_order_pair_expansion<FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const LowOrderPairExpansion<Scalar> second_expansion =
      make_low_order_pair_expansion<ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[3];
  boys_values<2>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
      const LowOrderHermiteTerm<Scalar>& first_item = first_expansion.terms[first_term];
      const LowOrderHermiteTerm<Scalar>& second_item = second_expansion.terms[second_term];
      const double sign =
          (low_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      value =
          value + sign * first_item.coefficient * second_item.coefficient *
                      low_order_coulomb(first_item.derivative_state + second_item.derivative_state,
                                        rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

/** Cartesian source component count for one s, p, or d shell. */
template <unsigned ShellAngular>
__host__ __device__ constexpr unsigned order2_shell_component_count() {
  static_assert(ShellAngular <= 2);
  return (ShellAngular + 1) * (ShellAngular + 2) / 2;
}

/** Return one CCA-ordered Cartesian component through d angular momentum. */
template <unsigned ShellAngular>
__device__ __forceinline__ Angular order2_shell_component(unsigned component) {
  static_assert(ShellAngular <= 2);
  if constexpr (ShellAngular == 0) {
    (void)component;
    return {0, 0, 0};
  } else if constexpr (ShellAngular == 1) {
    return component == 0 ? Angular{1, 0, 0} : component == 1 ? Angular{0, 1, 0} : Angular{0, 0, 1};
  } else {
    switch (component) {
      case 0:
        return {2, 0, 0};
      case 1:
        return {1, 1, 0};
      case 2:
        return {1, 0, 1};
      case 3:
        return {0, 2, 0};
      case 4:
        return {0, 1, 1};
      default:
        return {0, 0, 2};
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution
""",
    "generated_direct_eri_order3.cuh": r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "generated_direct_pair_order2.cuh"
#include "generated_direct_pair_order3.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned Direct recurrence lowering.
// Do not edit this build artifact: change
// python/vibeqc_compiler/integral/direct_recurrence_cuda.py instead.
// Arithmetic/workspace order is preserved; host plans and queue policy remain native.
namespace vibeqc::scf::cuda_execution {

/** Value and P-Q chain derivatives of a weighted order-two Hermite DAG. */
struct WeightedOrder2Coulomb {
  double c0;
  double cx;
  double cy;
  double cz;
  double value;
  double chain[3];
};

/**
 * Contract the ten order-zero-through-two Hermite coefficients with exactly
 * twenty unique Cartesian Coulomb states. The returned chain is only the
 * derivative through P-Q; shell-class helpers add their explicit PA/PB/QC
 * coefficient derivatives and Gaussian-pair decay separately.
 */
__device__ __forceinline__ WeightedOrder2Coulomb contract_weighted_order2_coulomb(
    double rho, double x, double y, double z, const double (&boys)[4], double h0, double hx,
    double hy, double hz, double hxx, double hxy, double hxz, double hyy, double hyz, double hzz) {
  const double twice_rho = 2.0 * rho;
  const double twice_rho_squared = twice_rho * twice_rho;
  const double twice_rho_cubed = twice_rho_squared * twice_rho;
  const double c0 = boys[0];
  const double cx = -twice_rho * x * boys[1];
  const double cy = -twice_rho * y * boys[1];
  const double cz = -twice_rho * z * boys[1];
  const double cxx = twice_rho_squared * x * x * boys[2] - twice_rho * boys[1];
  const double cxy = twice_rho_squared * x * y * boys[2];
  const double cxz = twice_rho_squared * x * z * boys[2];
  const double cyy = twice_rho_squared * y * y * boys[2] - twice_rho * boys[1];
  const double cyz = twice_rho_squared * y * z * boys[2];
  const double czz = twice_rho_squared * z * z * boys[2] - twice_rho * boys[1];
  const double cxxx =
      -twice_rho_cubed * x * x * x * boys[3] + 3.0 * twice_rho_squared * x * boys[2];
  const double cxxy = -twice_rho_cubed * x * x * y * boys[3] + twice_rho_squared * y * boys[2];
  const double cxxz = -twice_rho_cubed * x * x * z * boys[3] + twice_rho_squared * z * boys[2];
  const double cxyy = -twice_rho_cubed * x * y * y * boys[3] + twice_rho_squared * x * boys[2];
  const double cxyz = -twice_rho_cubed * x * y * z * boys[3];
  const double cxzz = -twice_rho_cubed * x * z * z * boys[3] + twice_rho_squared * x * boys[2];
  const double cyyy =
      -twice_rho_cubed * y * y * y * boys[3] + 3.0 * twice_rho_squared * y * boys[2];
  const double cyyz = -twice_rho_cubed * y * y * z * boys[3] + twice_rho_squared * z * boys[2];
  const double cyzz = -twice_rho_cubed * y * z * z * boys[3] + twice_rho_squared * y * boys[2];
  const double czzz =
      -twice_rho_cubed * z * z * z * boys[3] + 3.0 * twice_rho_squared * z * boys[2];

  WeightedOrder2Coulomb result{};
  result.c0 = c0;
  result.cx = cx;
  result.cy = cy;
  result.cz = cz;
  result.value = h0 * c0 + hx * cx + hy * cy + hz * cz + hxx * cxx + hxy * cxy + hxz * cxz +
                 hyy * cyy + hyz * cyz + hzz * czz;
  result.chain[0] = h0 * cx + hx * cxx + hy * cxy + hz * cxz + hxx * cxxx + hxy * cxxy +
                    hxz * cxxz + hyy * cxyy + hyz * cxyz + hzz * cxzz;
  result.chain[1] = h0 * cy + hx * cxy + hy * cyy + hz * cyz + hxx * cxxy + hxy * cxyy +
                    hxz * cxyz + hyy * cyyy + hyz * cyyz + hzz * cyzz;
  result.chain[2] = h0 * cz + hx * cxz + hy * cyz + hz * czz + hxx * cxxz + hxy * cxyz +
                    hxz * cxzz + hyy * cyyz + hyz * cyzz + hzz * czzz;
  return result;
}

/**
 * Closed order-3 contraction for canonical (f s|s s), (d p|s s),
 * (d s|p s), and (p p|p s) primitive quartets.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ inline Scalar primitive_eri_order3(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder = FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder = ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 3);
  static_assert(FirstPairOrder <= 3 && SecondPairOrder <= 3);
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const ThirdOrderPairExpansion<FirstPairOrder, Scalar> first_expansion =
      make_third_order_pair_expansion<FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const ThirdOrderPairExpansion<SecondPairOrder, Scalar> second_expansion =
      make_third_order_pair_expansion<ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[4];
  boys_values<3>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
      const LowOrderHermiteTerm<Scalar>& first_item = first_expansion.terms[first_term];
      const LowOrderHermiteTerm<Scalar>& second_item = second_expansion.terms[second_term];
      const double sign =
          (low_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      value = value +
              sign * first_item.coefficient * second_item.coefficient *
                  third_order_coulomb(first_item.derivative_state + second_item.derivative_state,
                                      rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

}  // namespace vibeqc::scf::cuda_execution
""",
    "generated_direct_eri_order4.cuh": r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "generated_direct_pair_order3.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned Direct recurrence lowering.
// Do not edit this build artifact: change
// python/vibeqc_compiler/integral/direct_recurrence_cuda.py instead.
// Arithmetic/workspace order is preserved; host plans and queue policy remain native.
namespace vibeqc::scf::cuda_execution {

/** One sparse Hermite coefficient with three bits per Cartesian derivative. */
template <typename Scalar>
struct FourthOrderHermiteTerm {
  // Order four needs values 0--4 on one axis, so the two-bit encoding used by
  // lower orders is deliberately widened only for this specialization.
  unsigned derivative_state;
  Scalar coefficient;
};

/** Exact-sized sparse shell-pair expansion through total order four. */
template <unsigned PairOrder, typename Scalar>
struct FourthOrderPairExpansion {
  static_assert(PairOrder <= 4);
  FourthOrderHermiteTerm<Scalar> terms[1U << PairOrder];
};

__device__ inline unsigned fourth_order_derivative_state(int axis) { return 1U << (3 * axis); }

__device__ inline unsigned fourth_order_derivative_total(unsigned state) {
  return (state & 7U) + ((state >> 3U) & 7U) + ((state >> 6U) & 7U);
}

/**
 * Generate the exact Wick expansion of one shell pair through order four.
 *
 * Base subset terms represent uncontracted angular quanta. Every same-axis
 * pair adds one 1/(2p) contraction times the uncontracted remaining factors;
 * order four additionally admits the three possible disjoint pairings. All
 * contributions merge into the existing 2^N subset slots, so no generic
 * recurrence workspace is required.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, typename Scalar>
__device__ inline FourthOrderPairExpansion<FirstShellAngular + SecondShellAngular, Scalar>
make_fourth_order_pair_expansion(double exponent, const Vec3<Scalar>& product,
                                 const Vec3<Scalar>& first, const Angular& angular_first,
                                 const Vec3<Scalar>& second, const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 4);
  constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  FourthOrderPairExpansion<PairOrder, Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[QuantumStorage];
    Scalar shifts[QuantumStorage];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = fourth_order_derivative_state(axis);
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
    } else if constexpr (PairOrder == 4) {
      for (unsigned first_quantum = 0; first_quantum < 4; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 4; ++second_quantum) {
          if (derivative_states[first_quantum] != derivative_states[second_quantum]) {
            continue;
          }
          unsigned remaining[2];
          unsigned remaining_count = 0;
          for (unsigned quantum = 0; quantum < 4; ++quantum) {
            if (quantum != first_quantum && quantum != second_quantum) {
              remaining[remaining_count++] = quantum;
            }
          }
          const unsigned first_remaining = remaining[0];
          const unsigned second_remaining = remaining[1];
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              inverse_two_exponent * shifts[first_remaining] * shifts[second_remaining];
          expansion.terms[1U << first_remaining].coefficient =
              expansion.terms[1U << first_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent) *
                  shifts[second_remaining];
          expansion.terms[1U << second_remaining].coefficient =
              expansion.terms[1U << second_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent) * shifts[first_remaining];
          const unsigned both_remaining = (1U << first_remaining) | (1U << second_remaining);
          expansion.terms[both_remaining].coefficient =
              expansion.terms[both_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent * inverse_two_exponent);
        }
      }

      constexpr unsigned Pairings[3][4] = {
          {0, 1, 2, 3},
          {0, 2, 1, 3},
          {0, 3, 1, 2},
      };
      for (unsigned pairing = 0; pairing < 3; ++pairing) {
        if (derivative_states[Pairings[pairing][0]] == derivative_states[Pairings[pairing][1]] &&
            derivative_states[Pairings[pairing][2]] == derivative_states[Pairings[pairing][3]]) {
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most four. */
template <typename Scalar>
__device__ inline Scalar fourth_order_coulomb(unsigned derivative_state, double rho,
                                              const Vec3<Scalar>& product_difference,
                                              const Scalar* boys) {
  const unsigned x_order = derivative_state & 7U;
  const unsigned y_order = (derivative_state >> 3U) & 7U;
  const unsigned z_order = (derivative_state >> 6U) & 7U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 4) {
    const unsigned lower_order_state = x_order | (y_order << 2U) | (z_order << 4U);
    return third_order_coulomb(lower_order_state, rho, product_difference, boys);
  }

  const double fourth_order_factor = 16.0 * rho * rho * rho * rho;
  const double third_order_factor = -8.0 * rho * rho * rho;
  const double second_order_factor = 4.0 * rho * rho;
  if (x_order == 4 || y_order == 4 || z_order == 4) {
    const Scalar coordinate = x_order == 4
                                  ? product_difference.x
                                  : (y_order == 4 ? product_difference.y : product_difference.z);
    const Scalar coordinate_squared = coordinate * coordinate;
    return fourth_order_factor * coordinate_squared * coordinate_squared * boys[4] +
           (6.0 * third_order_factor) * coordinate_squared * boys[3] +
           (3.0 * second_order_factor) * boys[2];
  }

  if (x_order == 3 || y_order == 3 || z_order == 3) {
    const Scalar repeated_coordinate =
        x_order == 3 ? product_difference.x
                     : (y_order == 3 ? product_difference.y : product_difference.z);
    const Scalar single_coordinate =
        x_order == 1 ? product_difference.x
                     : (y_order == 1 ? product_difference.y : product_difference.z);
    return fourth_order_factor * repeated_coordinate * repeated_coordinate * repeated_coordinate *
               single_coordinate * boys[4] +
           (3.0 * third_order_factor) * repeated_coordinate * single_coordinate * boys[3];
  }

  if ((x_order == 2 && y_order == 2) || (x_order == 2 && z_order == 2) ||
      (y_order == 2 && z_order == 2)) {
    Scalar first_coordinate = product_difference.x;
    Scalar second_coordinate = product_difference.y;
    if (x_order == 0) {
      first_coordinate = product_difference.y;
      second_coordinate = product_difference.z;
    } else if (y_order == 0) {
      second_coordinate = product_difference.z;
    }
    const Scalar first_squared = first_coordinate * first_coordinate;
    const Scalar second_squared = second_coordinate * second_coordinate;
    return fourth_order_factor * first_squared * second_squared * boys[4] +
           third_order_factor * (first_squared + second_squared) * boys[3] +
           second_order_factor * boys[2];
  }

  const Scalar repeated_coordinate =
      x_order == 2 ? product_difference.x
                   : (y_order == 2 ? product_difference.y : product_difference.z);
  Scalar single_product = scalar<Scalar>(1.0);
  if (x_order == 1) single_product = single_product * product_difference.x;
  if (y_order == 1) single_product = single_product * product_difference.y;
  if (z_order == 1) single_product = single_product * product_difference.z;
  return fourth_order_factor * repeated_coordinate * repeated_coordinate * single_product *
             boys[4] +
         third_order_factor * single_product * boys[3];
}

/**
 * Closed order-4 contraction for canonical (f p|s s), (d d|s s),
 * (f s|p s), (d p|p s), (d s|d s), (d s|p p), and (p p|p p) quartets.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ inline Scalar primitive_eri_order4(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder = FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder = ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 4);
  static_assert(FirstPairOrder <= 4 && SecondPairOrder <= 4);
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const FourthOrderPairExpansion<FirstPairOrder, Scalar> first_expansion =
      make_fourth_order_pair_expansion<FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const FourthOrderPairExpansion<SecondPairOrder, Scalar> second_expansion =
      make_fourth_order_pair_expansion<ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[5];
  boys_values<4>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
      const FourthOrderHermiteTerm<Scalar>& first_item = first_expansion.terms[first_term];
      const FourthOrderHermiteTerm<Scalar>& second_item = second_expansion.terms[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      value = value +
              sign * first_item.coefficient * second_item.coefficient *
                  fourth_order_coulomb(first_item.derivative_state + second_item.derivative_state,
                                       rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

}  // namespace vibeqc::scf::cuda_execution
""",
    "generated_direct_shell_class.cuh": r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_cartesian.cuh"
#include "generated_direct_eri_order2.cuh"
#include "generated_direct_eri_order3.cuh"
#include "generated_direct_eri_order4.cuh"
#include "generated_direct_shell_pair_hermite.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned Direct recurrence lowering.
// Do not edit this build artifact: change
// python/vibeqc_compiler/integral/direct_recurrence_cuda.py instead.
// Arithmetic/workspace order is preserved; host plans and queue policy remain native.
namespace vibeqc::scf::cuda_execution {

/**
 * Evaluate one Cartesian primitive quartet with exact shell-pair workspaces.
 *
 * Axis powers remain AO-component data, while the enclosing shell angular
 * momenta bound every Hermite dimension at compile time. This avoids charging
 * an s/p/d task for the generic f/f pair workspace.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ inline Scalar primitive_eri_cartesian_shell_class(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned MaximumAngular =
      FirstShellAngular + SecondShellAngular + ThirdShellAngular + FourthShellAngular;
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  if constexpr (FirstShellAngular == 1 && SecondShellAngular == 0 && ThirdShellAngular == 0 &&
                FourthShellAngular == 0) {
    const int axis = angular_first.x == 1 ? 0 : (angular_first.y == 1 ? 1 : 2);
    return primitive_eri_psss(axis, alpha, first, beta, second, gamma, third, delta, fourth);
  } else if constexpr (MaximumAngular == 2) {
    return primitive_eri_order2<FirstShellAngular, SecondShellAngular, ThirdShellAngular,
                                FourthShellAngular>(alpha, first, angular_first, beta, second,
                                                    angular_second, gamma, third, angular_third,
                                                    delta, fourth, angular_fourth);
  } else if constexpr (MaximumAngular == 3) {
    return primitive_eri_order3<FirstShellAngular, SecondShellAngular, ThirdShellAngular,
                                FourthShellAngular>(alpha, first, angular_first, beta, second,
                                                    angular_second, gamma, third, angular_third,
                                                    delta, fourth, angular_fourth);
  } else if constexpr (MaximumAngular == 4) {
    return primitive_eri_order4<FirstShellAngular, SecondShellAngular, ThirdShellAngular,
                                FourthShellAngular>(alpha, first, angular_first, beta, second,
                                                    angular_second, gamma, third, angular_third,
                                                    delta, fourth, angular_fourth);
  } else {
    using Real = EvaluationReal<Scalar>;
    const Real alpha_value{alpha};
    const Real beta_value{beta};
    const Real gamma_value{gamma};
    const Real delta_value{delta};
    const Real p = alpha_value + beta_value;
    const Real q = gamma_value + delta_value;
    const Real rho = p * q / (p + q);
    const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
    const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
    ShellPairHermiteCoefficients<Scalar, FirstShellAngular, SecondShellAngular>
        first_coefficients[3];
    ShellPairHermiteCoefficients<Scalar, ThirdShellAngular, FourthShellAngular>
        second_coefficients[3];
    for (int axis = 0; axis < 3; ++axis) {
      fill_shell_pair_hermite<FirstShellAngular, SecondShellAngular>(
          angular_axis(angular_first, axis), angular_axis(angular_second, axis),
          vec_axis(product_p, axis), vec_axis(first, axis), vec_axis(second, axis), alpha, beta,
          first_coefficients[axis]);
      fill_shell_pair_hermite<ThirdShellAngular, FourthShellAngular>(
          angular_axis(angular_third, axis), angular_axis(angular_fourth, axis),
          vec_axis(product_q, axis), vec_axis(third, axis), vec_axis(fourth, axis), gamma, delta,
          second_coefficients[axis]);
    }
    return eri_cartesian_value<MaximumAngular>(p, q, rho, product_p, product_q, angular_first,
                                               angular_second, angular_third, angular_fourth,
                                               first_coefficients, second_coefficients);
  }
}

}  // namespace vibeqc::scf::cuda_execution
""",
}


def emit_direct_recurrence_headers() -> dict[str, str]:
    """Return generated Direct recurrence headers keyed by output filename."""

    return dict(_SOURCES)
