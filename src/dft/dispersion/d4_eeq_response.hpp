#pragma once

#include "dft/dispersion/d4_eeq.hpp"

#if defined(__CUDACC__)
#define GENERATIVEQC_EEQ_RESPONSE_HD __host__ __device__
#else
#define GENERATIVEQC_EEQ_RESPONSE_HD
#endif

namespace generativeqc::dft::dispersion {

/** One LU factorization, four augmented vectors, and linear response scratch.
 * No Cartesian-by-charge Jacobian or three-index derivative is retained. */
GENERATIVEQC_EEQ_RESPONSE_HD inline std::size_t eeq_direction_workspace_elements(int atoms) {
  if (atoms < 1 || atoms > kEEQMaximumAtoms) return 0;
  const auto count = static_cast<std::size_t>(atoms);
  const auto extent = count + 1;
  return extent * extent + 4 * extent + 10 * count;
}

namespace eeq_detail {
struct DirectionalPair {
  double separation[3], motion[3], distance, distance_dot, gamma;
};

GENERATIVEQC_EEQ_RESPONSE_HD inline DirectionalPair directional_pair(int first, int second,
                                                                     const std::int32_t* numbers,
                                                                     const double* xyz,
                                                                     const double* direction,
                                                                     EEQTables tables) {
  DirectionalPair pair{};
  pair.distance = sqrt(d4_detail::distance2(xyz, first, second, pair.separation));
  for (int axis = 0; axis < 3; ++axis) {
    pair.motion[axis] = direction[3 * first + axis] - direction[3 * second + axis];
    pair.distance_dot += pair.separation[axis] * pair.motion[axis];
  }
  pair.distance_dot /= pair.distance;
  const double radius_a = tables.charge_elements[numbers[first] - 1].radius;
  const double radius_b = tables.charge_elements[numbers[second] - 1].radius;
  pair.gamma = 1.0 / sqrt(radius_a * radius_a + radius_b * radius_b);
  return pair;
}
}  // namespace eeq_detail

/** Differentiate only the charge-mediated pullback of E(R, q(R)).
 * SeedProvider(q, q_dot, seed, seed_dot) supplies dE/dq and its *total*
 * directional derivative using borrowed, atom-length arrays. The provider
 * must not retain these views. This function publishes only the geometry
 * terms J_q^T seed and their tangents; the fixed-charge D4 owner supplies
 * E_R and its tangent separately. All numeric arrays must be disjoint. */
template <typename SeedProvider>
GENERATIVEQC_EEQ_RESPONSE_HD D4Status eeq_directional_pullback(
    int atoms, const std::int32_t* numbers, const double* xyz, double total_charge,
    EEQTables tables, const double* direction, double* workspace, std::size_t workspace_elements,
    SeedProvider seeds, double* gradient, double* action) {
  using namespace eeq_detail;
  namespace scalar = ::generativeqc::generated::dispersion;
  if (atoms < 1 || atoms > kEEQMaximumAtoms || !tables.elements || !tables.charge_elements ||
      tables.element_count != kD4TableElementCount)
    return D4Status::unsupported;
  const auto capacity = eeq_direction_workspace_elements(atoms);
  if (workspace_elements < capacity || !finite(total_charge)) return D4Status::invalid_argument;
  const auto count = static_cast<std::size_t>(atoms);
  const auto cartesian_bytes = 3 * count * sizeof(double);
  const void* buffers[] = {numbers,  xyz,    direction,       workspace,
                           gradient, action, tables.elements, tables.charge_elements};
  const std::size_t sizes[] = {count * sizeof(*numbers),
                               cartesian_bytes,
                               cartesian_bytes,
                               capacity * sizeof(double),
                               cartesian_bytes,
                               cartesian_bytes,
                               tables.element_count * sizeof(data::D4ElementData),
                               tables.element_count * sizeof(D4EEQChargeElementData)};
  d4_detail::Range ranges[8];
  for (int index = 0; index < 8; ++index) {
    if (!d4_detail::range(buffers[index], sizes[index], ranges[index]) ||
        reinterpret_cast<std::uintptr_t>(buffers[index]) %
            (index == 0 ? alignof(std::int32_t) : alignof(double)))
      return D4Status::invalid_argument;
    for (int other = 0; other < index; ++other)
      if (d4_detail::overlaps(ranges[index], ranges[other])) return D4Status::invalid_argument;
  }
  for (int atom = 0; atom < atoms; ++atom) {
    if (numbers[atom] < 1 || numbers[atom] > static_cast<int>(tables.element_count))
      return D4Status::unsupported;
    for (int axis = 0; axis < 3; ++axis)
      if (!finite(xyz[3 * atom + axis]) || !finite(direction[3 * atom + axis]))
        return D4Status::invalid_argument;
  }

  const int extent = atoms + 1;
  double* lu = workspace;
  double* charges = lu + static_cast<std::size_t>(extent) * extent;
  double* charge_dot = charges + extent;
  double* adjoint = charge_dot + extent;
  double* adjoint_dot = adjoint + extent;
  double* raw_cn = adjoint_dot + extent;
  double* rhs_slope = raw_cn + atoms;
  double* cn_dot = rhs_slope + atoms;
  double* slope_dot = cn_dot + atoms;
  double* candidate_gradient = slope_dot + atoms;
  double* candidate_action = candidate_gradient + 3 * atoms;
  int pivots[kEEQMaximumAtoms + 1]{};
  auto status = factor_charge_system(atoms, numbers, xyz, total_charge, tables, raw_cn, rhs_slope,
                                     lu, charges, pivots);
  if (status != D4Status::success) return status;

  for (int atom = 0; atom < atoms; ++atom) cn_dot[atom] = charge_dot[atom] = 0;
  charge_dot[atoms] = 0;
  for (int first = 1; first < atoms; ++first)
    for (int second = 0; second < first; ++second) {
      const auto pair = directional_pair(first, second, numbers, xyz, direction, tables);
      // There is no Hessian exactly at a hard coordination membership change.
      if (pair.distance == kEEQCutoff) return D4Status::unsupported;
      if (pair.distance < kEEQCutoff) {
        double pair_count, slope;
        eeq_cn_pair(tables.elements[numbers[first] - 1], tables.elements[numbers[second] - 1],
                    pair.distance, pair_count, slope);
        cn_dot[first] += slope * pair.distance_dot;
        cn_dot[second] += slope * pair.distance_dot;
      }
      double coulomb[2];
      scalar::eeq_coulomb(pair.distance, pair.gamma, coulomb);
      const double matrix_dot = coulomb[1] * pair.distance * pair.distance_dot;
      charge_dot[first] -= matrix_dot * charges[second];
      charge_dot[second] -= matrix_dot * charges[first];
    }
  for (int atom = 0; atom < atoms; ++atom) {
    const auto parameter = tables.charge_elements[numbers[atom] - 1];
    double rhs[3];
    scalar::eeq_rhs_second(raw_cn[atom], parameter.chi, parameter.kcnchi, rhs);
    slope_dot[atom] = rhs[2] * cn_dot[atom];
    charge_dot[atom] += rhs_slope[atom] * cn_dot[atom];
  }
  // q_dot = A^-1 (rhs_dot - A_dot q). With a fixed total charge its last RHS is zero.
  if (!lu_solve(lu, extent, pivots, charge_dot)) return D4Status::numerical_failure;
  status = seeds(charges, charge_dot, adjoint, adjoint_dot);
  if (status != D4Status::success) return status;
  adjoint[atoms] = adjoint_dot[atoms] = 0;
  // A is symmetric even though its pivoted factors are not. Reuse the same solve
  // for lambda = A^-T seed and lambda_dot = A^-T (seed_dot - A_dot lambda).
  if (!lu_solve(lu, extent, pivots, adjoint)) return D4Status::numerical_failure;
  for (int first = 1; first < atoms; ++first)
    for (int second = 0; second < first; ++second) {
      const auto pair = directional_pair(first, second, numbers, xyz, direction, tables);
      double coulomb[2];
      scalar::eeq_coulomb(pair.distance, pair.gamma, coulomb);
      const double matrix_dot = coulomb[1] * pair.distance * pair.distance_dot;
      adjoint_dot[first] -= matrix_dot * adjoint[second];
      adjoint_dot[second] -= matrix_dot * adjoint[first];
    }
  if (!lu_solve(lu, extent, pivots, adjoint_dot)) return D4Status::numerical_failure;

  for (int coordinate = 0; coordinate < 3 * atoms; ++coordinate)
    candidate_gradient[coordinate] = candidate_action[coordinate] = 0;
  for (int first = 1; first < atoms; ++first)
    for (int second = 0; second < first; ++second) {
      const auto pair = directional_pair(first, second, numbers, xyz, direction, tables);
      double coulomb[3];
      scalar::eeq_coulomb_second(pair.distance, pair.gamma, coulomb);
      const double matrix_weight =
          adjoint[first] * charges[second] + adjoint[second] * charges[first];
      const double matrix_weight_dot =
          adjoint_dot[first] * charges[second] + adjoint_dot[second] * charges[first] +
          adjoint[first] * charge_dot[second] + adjoint[second] * charge_dot[first];
      double scatter = -matrix_weight * coulomb[1];
      double scatter_dot =
          -matrix_weight_dot * coulomb[1] - matrix_weight * coulomb[2] * pair.distance_dot;
      if (pair.distance < kEEQCutoff) {
        const double radius = tables.elements[numbers[first] - 1].covalent_radius +
                              tables.elements[numbers[second] - 1].covalent_radius;
        double coordination[3];
        scalar::d4_coordination_second(pair.distance, radius, 1., coordination);
        const double cn_scatter = coordination[1] / pair.distance;
        const double cn_scatter_dot =
            (coordination[2] - cn_scatter) * pair.distance_dot / pair.distance;
        const double cn_weight =
            adjoint[first] * rhs_slope[first] + adjoint[second] * rhs_slope[second];
        const double cn_weight_dot =
            adjoint_dot[first] * rhs_slope[first] + adjoint_dot[second] * rhs_slope[second] +
            adjoint[first] * slope_dot[first] + adjoint[second] * slope_dot[second];
        scatter += cn_weight * cn_scatter;
        scatter_dot += cn_weight_dot * cn_scatter + cn_weight * cn_scatter_dot;
      }
      for (int axis = 0; axis < 3; ++axis) {
        const double contribution = scatter * pair.separation[axis];
        const double tangent = scatter_dot * pair.separation[axis] + scatter * pair.motion[axis];
        candidate_gradient[3 * first + axis] += contribution;
        candidate_gradient[3 * second + axis] -= contribution;
        candidate_action[3 * first + axis] += tangent;
        candidate_action[3 * second + axis] -= tangent;
      }
    }
  for (int coordinate = 0; coordinate < 3 * atoms; ++coordinate)
    if (!finite(candidate_gradient[coordinate]) || !finite(candidate_action[coordinate]))
      return D4Status::numerical_failure;
  for (int coordinate = 0; coordinate < 3 * atoms; ++coordinate) {
    gradient[coordinate] = candidate_gradient[coordinate];
    action[coordinate] = candidate_action[coordinate];
  }
  return D4Status::success;
}

}  // namespace generativeqc::dft::dispersion

#undef GENERATIVEQC_EEQ_RESPONSE_HD
