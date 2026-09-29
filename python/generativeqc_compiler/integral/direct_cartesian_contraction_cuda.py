"""Compiler-owned emission of Direct Cartesian and generic contraction support.

These generated headers preserve the qualified Cartesian primitive evaluation and
generic contracted-ERI arithmetic while removing maintained native CUDA formula
owners. Runtime scheduling, screening, selectors and numerical policy remain
outside this module.
"""

from __future__ import annotations

_SOURCES = {
    "generated_direct_cartesian.cuh": r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/hermite_recurrence.cuh"
#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned Direct Cartesian lowering.
// Do not edit this build artifact: change
// python/generativeqc_compiler/integral/direct_cartesian_contraction_cuda.py instead.
// Arithmetic/workspace order is preserved; host plans and queue policy remain native.
namespace generativeqc::scf::cuda_execution {

template <typename Scalar>
__device__ inline Scalar primitive_eri(double alpha, const Vec3<Scalar>& first, double beta,
                                       const Vec3<Scalar>& second, double gamma,
                                       const Vec3<Scalar>& third, double delta,
                                       const Vec3<Scalar>& fourth) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const Vec3<Scalar> center_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> center_q = product_center(gamma, third, delta, fourth);
  const double rho = p * q / (p + q);
  const double prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));
  return prefactor *
         qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth)) *
         boys0(rho * distance_squared(center_p, center_q));
}

template <unsigned MaximumAngular, typename Scalar, typename FirstCoefficients,
          typename SecondCoefficients>
__device__ inline __noinline__ Scalar eri_cartesian_value(
    EvaluationReal<Scalar> p, EvaluationReal<Scalar> q, EvaluationReal<Scalar> rho,
    const Vec3<Scalar>& product_p, const Vec3<Scalar>& product_q, const Angular& angular_first,
    const Angular& angular_second, const Angular& angular_third, const Angular& angular_fourth,
    const FirstCoefficients* first_coefficients, const SecondCoefficients* second_coefficients,
    generativeqc::integrals::CoulombRange range = generativeqc::integrals::CoulombRange::Full,
    double omega = 0.0) {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  CoulombAuxiliary<Scalar, MaximumAngular> auxiliary;
  if constexpr (!std::is_same_v<Scalar, MixedPrecisionFloat>) {
    if (range == generativeqc::integrals::CoulombRange::Full)
      fill_coulomb<MaximumAngular>(rho, product_p, product_q, auxiliary);
    else if (!fill_range_coulomb<MaximumAngular>(rho, product_p, product_q, range, omega,
                                                 auxiliary))
      return scalar<Scalar>(NAN);
  } else {
    if (range != generativeqc::integrals::CoulombRange::Full) return scalar<Scalar>(NAN);
    fill_coulomb<MaximumAngular>(rho, product_p, product_q, auxiliary);
  }

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned t = 0; t <= angular_first.x + angular_second.x; ++t) {
    for (unsigned u = 0; u <= angular_first.y + angular_second.y; ++u) {
      for (unsigned v = 0; v <= angular_first.z + angular_second.z; ++v) {
        const Scalar first_value = first_coefficients[0].at(angular_first.x, angular_second.x, t) *
                                   first_coefficients[1].at(angular_first.y, angular_second.y, u) *
                                   first_coefficients[2].at(angular_first.z, angular_second.z, v);
        for (unsigned tau = 0; tau <= angular_third.x + angular_fourth.x; ++tau) {
          for (unsigned nu = 0; nu <= angular_third.y + angular_fourth.y; ++nu) {
            for (unsigned phi = 0; phi <= angular_third.z + angular_fourth.z; ++phi) {
              const double sign = ((tau + nu + phi) & 1U) == 0 ? 1.0 : -1.0;
              value =
                  value + sign * first_value *
                              second_coefficients[0].at(angular_third.x, angular_fourth.x, tau) *
                              second_coefficients[1].at(angular_third.y, angular_fourth.y, nu) *
                              second_coefficients[2].at(angular_third.z, angular_fourth.z, phi) *
                              auxiliary.at(0, t + tau, u + nu, v + phi);
            }
          }
        }
      }
    }
  }
  const EvaluationReal<Scalar> prefactor =
      EvaluationReal<Scalar>{2.0 * pow(kPi, 2.5)} / (p * q * qsqrt(p + q));
  return prefactor * value;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ inline Scalar primitive_eri_cartesian(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth,
    generativeqc::integrals::CoulombRange range = generativeqc::integrals::CoulombRange::Full,
    double omega = 0.0) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  HermiteCoefficients<Scalar> first_coefficients[3];
  HermiteCoefficients<Scalar> second_coefficients[3];
  for (int axis = 0; axis < 3; ++axis) {
    fill_hermite(angular_axis(angular_first, axis), angular_axis(angular_second, axis),
                 vec_axis(product_p, axis), vec_axis(first, axis), vec_axis(second, axis), alpha,
                 beta, first_coefficients[axis]);
    fill_hermite(angular_axis(angular_third, axis), angular_axis(angular_fourth, axis),
                 vec_axis(product_q, axis), vec_axis(third, axis), vec_axis(fourth, axis), gamma,
                 delta, second_coefficients[axis]);
  }
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  return eri_cartesian_value<MaximumAngular>(p, q, rho, product_p, product_q, angular_first,
                                             angular_second, angular_third, angular_fourth,
                                             first_coefficients, second_coefficients, range, omega);
}

/**
 * Closed first-order Hermite contraction for canonical (p s | s s).
 *
 * The exact order-1 shell class has only one Cartesian component on the first
 * center. Generating its two reachable Hermite terms directly avoids all six
 * coefficient workspaces and the generic six-deep component contraction.
 */
template <typename Scalar>
__device__ inline Scalar primitive_eri_psss(int axis, double alpha, const Vec3<Scalar>& first,
                                            double beta, const Vec3<Scalar>& second, double gamma,
                                            const Vec3<Scalar>& third, double delta,
                                            const Vec3<Scalar>& fourth) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  Scalar boys[2];
  boys_values<1>(rho * distance_squared(product_p, product_q), boys);
  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  const Scalar pa = vec_axis(product_p, axis) - vec_axis(first, axis);
  const Scalar pq = vec_axis(product_p, axis) - vec_axis(product_q, axis);
  const Scalar value = pa * boys[0] - (rho / p) * pq * boys[1];
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

}  // namespace generativeqc::scf::cuda_execution
""",
    "generated_direct_contraction.cuh": r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "generated_direct_cartesian.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned Direct contraction lowering.
// Do not edit this build artifact: change
// python/generativeqc_compiler/integral/direct_cartesian_contraction_cuda.py instead.
// Arithmetic/reduction order is preserved; host plans and queue policy remain native.
namespace generativeqc::scf::cuda_execution {

template <unsigned MaximumAngular, typename Scalar>
__device__ inline __noinline__ Scalar contracted_eri_cartesian(
    const DeviceBatch& batch, std::int64_t ao_i, std::int64_t ao_j, std::int64_t ao_k,
    std::int64_t ao_l, std::int32_t shell_i, std::int32_t shell_j, std::int32_t shell_k,
    std::int32_t shell_l, std::int64_t derivative_coordinate,
    generativeqc::integrals::CoulombRange range = generativeqc::integrals::CoulombRange::Full,
    double omega = 0.0) {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  const Vec3<Scalar> first =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_i], derivative_coordinate);
  const Vec3<Scalar> second =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_j], derivative_coordinate);
  const Vec3<Scalar> third =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_k], derivative_coordinate);
  const Vec3<Scalar> fourth =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_l], derivative_coordinate);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const unsigned third_terms = batch.ao_term_counts[ao_k];
  const unsigned fourth_terms = batch.ao_term_counts[ao_l];

  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[shell_k];
           c < batch.shell_primitive_offsets[shell_k + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[shell_l];
             d < batch.shell_primitive_offsets[shell_l + 1]; ++d) {
          const double weight = batch.primitive_coefficients[a] * batch.primitive_coefficients[b] *
                                batch.primitive_coefficients[c] * batch.primitive_coefficients[d];
          for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
            const Angular first_angular = ao_angular(batch, ao_i, first_term);
            const double first_coefficient = ao_term_coefficient(batch, ao_i, first_term);
            for (unsigned second_term = 0; second_term < second_terms; ++second_term) {
              const Angular second_angular = ao_angular(batch, ao_j, second_term);
              const double second_coefficient = ao_term_coefficient(batch, ao_j, second_term);
              for (unsigned third_term = 0; third_term < third_terms; ++third_term) {
                const Angular third_angular = ao_angular(batch, ao_k, third_term);
                const double third_coefficient = ao_term_coefficient(batch, ao_k, third_term);
                for (unsigned fourth_term = 0; fourth_term < fourth_terms; ++fourth_term) {
                  result = result + weight * first_coefficient * second_coefficient *
                                        third_coefficient *
                                        ao_term_coefficient(batch, ao_l, fourth_term) *
                                        primitive_eri_cartesian<MaximumAngular>(
                                            batch.primitive_exponents[a], first, first_angular,
                                            batch.primitive_exponents[b], second, second_angular,
                                            batch.primitive_exponents[c], third, third_angular,
                                            batch.primitive_exponents[d], fourth,
                                            ao_angular(batch, ao_l, fourth_term), range, omega);
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ inline Scalar contracted_eri_order(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l, std::int64_t derivative_coordinate,
    generativeqc::integrals::CoulombRange range = generativeqc::integrals::CoulombRange::Full,
    double omega = 0.0) {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.nbf;
  const std::int64_t ao_i = base + i;
  const std::int64_t ao_j = base + j;
  const std::int64_t ao_k = base + k;
  const std::int64_t ao_l = base + l;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const std::int32_t shell_k = batch.ao_shells[ao_k];
  const std::int32_t shell_l = batch.ao_shells[ao_l];
  if (range != generativeqc::integrals::CoulombRange::Full) {
    if constexpr (std::is_same_v<Scalar, double>)
      return contracted_eri_cartesian<MaximumAngular, Scalar>(batch, ao_i, ao_j, ao_k, ao_l,
                                                              shell_i, shell_j, shell_k, shell_l,
                                                              derivative_coordinate, range, omega);
    else
      return scalar<Scalar>(NAN);
  }
  if constexpr (MaximumAngular == 0) {
    const Vec3<Scalar> first =
        atom_position<Scalar>(batch, batch.shell_atoms[shell_i], derivative_coordinate);
    const Vec3<Scalar> second =
        atom_position<Scalar>(batch, batch.shell_atoms[shell_j], derivative_coordinate);
    const Vec3<Scalar> third =
        atom_position<Scalar>(batch, batch.shell_atoms[shell_k], derivative_coordinate);
    const Vec3<Scalar> fourth =
        atom_position<Scalar>(batch, batch.shell_atoms[shell_l], derivative_coordinate);
    Scalar result = scalar<Scalar>(0.0);
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        for (std::int64_t c = batch.shell_primitive_offsets[shell_k];
             c < batch.shell_primitive_offsets[shell_k + 1]; ++c) {
          for (std::int64_t d = batch.shell_primitive_offsets[shell_l];
               d < batch.shell_primitive_offsets[shell_l + 1]; ++d) {
            const double weight = batch.primitive_coefficients[a] *
                                  batch.primitive_coefficients[b] *
                                  batch.primitive_coefficients[c] * batch.primitive_coefficients[d];
            result =
                result +
                weight * ao_term_coefficient(batch, ao_i, 0) * ao_term_coefficient(batch, ao_j, 0) *
                    ao_term_coefficient(batch, ao_k, 0) * ao_term_coefficient(batch, ao_l, 0) *
                    primitive_eri(batch.primitive_exponents[a], first, batch.primitive_exponents[b],
                                  second, batch.primitive_exponents[c], third,
                                  batch.primitive_exponents[d], fourth);
          }
        }
      }
    }
    return result;
  } else {
    return contracted_eri_cartesian<MaximumAngular, Scalar>(batch, ao_i, ao_j, ao_k, ao_l, shell_i,
                                                            shell_j, shell_k, shell_l,
                                                            derivative_coordinate, range, omega);
  }
}

template <typename Scalar>
__device__ inline Scalar contracted_eri(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l, std::int64_t derivative_coordinate,
    generativeqc::integrals::CoulombRange range = generativeqc::integrals::CoulombRange::Full,
    double omega = 0.0) {
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.nbf;
  const std::int32_t shell_i = batch.ao_shells[base + i];
  const std::int32_t shell_j = batch.ao_shells[base + j];
  const std::int32_t shell_k = batch.ao_shells[base + k];
  const std::int32_t shell_l = batch.ao_shells[base + l];
  // Shell angular momentum is invariant across Cartesian expansion terms, so
  // one contracted-quartet dispatch covers every primitive and sparse
  // spherical term below it.
  const unsigned maximum = batch.shell_angular[shell_i] + batch.shell_angular[shell_j] +
                           batch.shell_angular[shell_k] + batch.shell_angular[shell_l];
  switch (maximum) {
    case 0:
      return contracted_eri_order<0, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 1:
      return contracted_eri_order<1, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 2:
      return contracted_eri_order<2, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 3:
      return contracted_eri_order<3, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 4:
      return contracted_eri_order<4, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 5:
      return contracted_eri_order<5, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 6:
      return contracted_eri_order<6, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 7:
      return contracted_eri_order<7, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 8:
      return contracted_eri_order<8, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 9:
      return contracted_eri_order<9, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                             range, omega);
    case 10:
      return contracted_eri_order<10, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                              range, omega);
    case 11:
      return contracted_eri_order<11, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                              range, omega);
    case 12:
      return contracted_eri_order<12, Scalar>(batch, system, i, j, k, l, derivative_coordinate,
                                              range, omega);
  }
  return scalar<Scalar>(0.0);
}

}  // namespace generativeqc::scf::cuda_execution
""",
}


def emit_direct_cartesian_contraction_headers() -> dict[str, str]:
    """Return generated Direct Cartesian/contraction headers by filename."""

    return dict(_SOURCES)
