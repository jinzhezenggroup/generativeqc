#pragma once

#include "dft/dispersion/d4_reference.hpp"

#if defined(__CUDACC__)
#define GENERATIVEQC_D4_RESPONSE_HD __host__ __device__
#else
#define GENERATIVEQC_D4_RESPONSE_HD
#endif

namespace generativeqc::dft::dispersion {

/** Fixed-charge gradient/seed response needs only atom and reference panels. */
GENERATIVEQC_D4_RESPONSE_HD inline std::size_t d4_direction_workspace_elements(int atoms) {
  if (atoms < 1 || atoms > kD4MaximumAtoms) return 0;
  return (12 + 6 * kD4MaximumReferences) * static_cast<std::size_t>(atoms);
}

namespace d4_detail {
namespace direction_scalar = ::generativeqc::generated::dispersion;

struct EdgeDirection {
  double separation[3], motion[3], squared, squared_dot, distance, distance_dot;
};

GENERATIVEQC_D4_RESPONSE_HD inline EdgeDirection edge_direction(const double* xyz,
                                                                const double* direction, int first,
                                                                int second) {
  EdgeDirection edge{};
  edge.squared = distance2(xyz, first, second, edge.separation);
  edge.distance = sqrt(edge.squared);
  for (int axis = 0; axis < 3; ++axis) {
    edge.motion[axis] = direction[3 * first + axis] - direction[3 * second + axis];
    edge.squared_dot += 2 * edge.separation[axis] * edge.motion[axis];
  }
  edge.distance_dot = 0.5 * edge.squared_dot / edge.distance;
  return edge;
}

GENERATIVEQC_D4_RESPONSE_HD inline void scatter_direction(int first, int second,
                                                          const EdgeDirection& edge,
                                                          double coefficient,
                                                          double coefficient_dot, double* gradient,
                                                          double* action) {
  for (int axis = 0; axis < 3; ++axis) {
    const double value = coefficient * edge.separation[axis];
    const double tangent =
        coefficient_dot * edge.separation[axis] + coefficient * edge.motion[axis];
    gradient[3 * first + axis] += value;
    gradient[3 * second + axis] -= value;
    action[3 * first + axis] += tangent;
    action[3 * second + axis] -= tangent;
  }
}

struct WeightDirections {
  double *value, *cn, *charge, *value_dot, *cn_dot, *charge_dot;
};

/** Push the total CN/charge direction through the physical reference weights.
 * The constant max-reference fallback differentiates to zero with respect to CN. */
GENERATIVEQC_D4_RESPONSE_HD inline D4Status reference_weight_directions(
    int atoms, const std::int32_t* numbers, const double* cn, const double* cn_dot,
    const double* charge, const double* charge_dot, const D4Parameters& parameters, D4Tables tables,
    WeightDirections weights) {
  for (int atom = 0; atom < atoms; ++atom) {
    const auto element = tables.elements[numbers[atom] - 1];
    double numerator[kD4MaximumReferences][3]{}, normalization[3]{};
    double max_reference_cn = -DBL_MAX;
    for (int ref = 0; ref < element.reference_count; ++ref) {
      const auto reference = tables.references[element.reference_offset + ref];
      max_reference_cn = fmax(max_reference_cn, reference.coordination_number);
      for (int gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
        double jet[3];
        direction_scalar::gaussian_second(cn[atom], reference.coordination_number,
                                          gaussian * math::kReferenceWeightFactor, jet);
        for (int leg = 0; leg < 3; ++leg) {
          numerator[ref][leg] += jet[leg];
          normalization[leg] += jet[leg];
        }
      }
    }
    if (normalization[0] == math::kMinimumWeightNorm) return D4Status::unsupported;
    const double inverse = normalization[0] > math::kMinimumWeightNorm ? 1 / normalization[0] : 0;
    const double qmod = element.effective_charge + (charge ? charge[atom] : 0);
    if (charge && qmod == 0) return D4Status::unsupported;
    const double qdot = charge_dot ? charge_dot[atom] : 0;
    for (int ref = 0; ref < element.reference_count; ++ref) {
      const auto reference = tables.references[element.reference_offset + ref];
      double normalized = fabs(max_reference_cn - reference.coordination_number) < 1e-12 ? 1 : 0;
      double normalized_slope = 0, normalized_curvature = 0;
      if (inverse) {
        normalized = numerator[ref][0] * inverse;
        double jet[2];
        direction_scalar::normalize(numerator[ref][0], normalization[0], numerator[ref][1],
                                    normalization[1], jet);
        normalized_slope = jet[1];
        // Differentiate N = normalized * normalization twice. Using the
        // quotient identity avoids high inverse powers near the norm floor.
        normalized_curvature = inverse * (numerator[ref][2] - normalized * normalization[2] -
                                          2 * normalized_slope * normalization[1]);
      }
      double scale[3]{exp(parameters.ga), 0, 0};
      if (qmod > 0)
        direction_scalar::d4_charge_second(parameters.ga, parameters.gc * element.hardness,
                                           reference.charge + element.effective_charge, qmod,
                                           scale);
      const double cn_motion = normalized_slope * cn_dot[atom];
      const double charge_motion = scale[1] * qdot;
      const auto index = kD4MaximumReferences * atom + ref;
      weights.value[index] = normalized * scale[0];
      weights.cn[index] = normalized_slope * scale[0];
      weights.value_dot[index] = cn_motion * scale[0] + normalized * charge_motion;
      weights.cn_dot[index] =
          normalized_curvature * cn_dot[atom] * scale[0] + normalized_slope * charge_motion;
      weights.charge[index] = charge ? normalized * scale[1] : 0;
      weights.charge_dot[index] = charge ? cn_motion * scale[1] + normalized * scale[2] * qdot : 0;
    }
  }
  return D4Status::success;
}

struct CoefficientDirection {
  math::Coefficient value, tangent;
};

GENERATIVEQC_D4_RESPONSE_HD inline CoefficientDirection coefficient_direction(
    int first, int second, const std::int32_t* numbers, D4Tables tables, WeightDirections weights) {
  const auto lhs = tables.elements[numbers[first] - 1];
  const auto rhs = tables.elements[numbers[second] - 1];
  const auto lhs_offset = first * kD4MaximumReferences;
  const auto rhs_offset = second * kD4MaximumReferences;
  const math::PackedReferenceC6 references{tables.reference_c6};
  const auto value =
      math::coefficient(lhs, rhs, references, weights.value + lhs_offset, weights.cn + lhs_offset,
                        weights.charge + lhs_offset, weights.value + rhs_offset,
                        weights.cn + rhs_offset, weights.charge + rhs_offset);
  const auto first_dot = math::coefficient(
      lhs, rhs, references, weights.value_dot + lhs_offset, weights.cn_dot + lhs_offset,
      weights.charge_dot + lhs_offset, weights.value + rhs_offset, weights.cn + rhs_offset,
      weights.charge + rhs_offset);
  const auto second_dot =
      math::coefficient(lhs, rhs, references, weights.value + lhs_offset, weights.cn + lhs_offset,
                        weights.charge + lhs_offset, weights.value_dot + rhs_offset,
                        weights.cn_dot + rhs_offset, weights.charge_dot + rhs_offset);
  return {
      value,
      {first_dot.c6 + second_dot.c6, first_dot.first_cn + second_dot.first_cn,
       first_dot.second_cn + second_dot.second_cn, first_dot.first_charge + second_dot.first_charge,
       first_dot.second_charge + second_dot.second_charge}};
}
}  // namespace d4_detail

/** Apply the joint (R,q) Hessian of the D4 reference model. `action` is the
 * tangent of the partial Cartesian gradient; `seed_dot` is the tangent of
 * the partial charge gradient. Charge motion is an explicit input and may
 * come from the EEQ implicit solve, but there is no implicit VJP here. */
GENERATIVEQC_D4_RESPONSE_HD inline D4Status d4_fixed_charge_direction(
    int atoms, const std::int32_t* numbers, const double* xyz, const double* charges,
    const double* direction, const double* charge_dot, const D4Parameters& parameters,
    D4Tables tables, double* workspace, std::size_t workspace_elements, double* gradient,
    double* action, double* seed, double* seed_dot) {
  using namespace d4_detail;
  const auto required = d4_direction_workspace_elements(atoms);
  if (!required || tables.reference_model != parameters.reference_model ||
      fabs(tables.ga - parameters.ga) > 1e-15 || fabs(tables.gc - parameters.gc) > 1e-15 ||
      tables.element_count != kD4TableElementCount ||
      tables.reference_count != kD4TableReferenceCount ||
      tables.reference_c6_count != kD4PackedReferenceC6Count)
    return D4Status::unsupported;
  if (workspace_elements < required || !valid_parameters(parameters))
    return D4Status::invalid_argument;
  const auto atom_bytes = static_cast<std::size_t>(atoms) * sizeof(double);
  const void* buffers[] = {numbers,
                           xyz,
                           charges,
                           direction,
                           charge_dot,
                           workspace,
                           gradient,
                           action,
                           seed,
                           seed_dot,
                           tables.elements,
                           tables.references,
                           tables.reference_c6,
                           &parameters};
  const std::size_t sizes[] = {static_cast<std::size_t>(atoms) * sizeof(*numbers),
                               3 * atom_bytes,
                               atom_bytes,
                               3 * atom_bytes,
                               atom_bytes,
                               required * sizeof(double),
                               3 * atom_bytes,
                               3 * atom_bytes,
                               atom_bytes,
                               atom_bytes,
                               tables.element_count * sizeof(*tables.elements),
                               tables.reference_count * sizeof(*tables.references),
                               tables.reference_c6_count * sizeof(double),
                               sizeof(parameters)};
  Range ranges[14];
  for (int index = 0; index < 14; ++index) {
    if (!range(buffers[index], sizes[index], ranges[index]) ||
        reinterpret_cast<std::uintptr_t>(buffers[index]) %
            (index ? alignof(double) : alignof(std::int32_t)))
      return D4Status::invalid_argument;
    for (int other = 0; other < index; ++other)
      if (overlaps(ranges[index], ranges[other])) return D4Status::invalid_argument;
  }
  for (int atom = 0; atom < atoms; ++atom) {
    if (numbers[atom] < 1 || numbers[atom] > static_cast<int>(tables.element_count))
      return D4Status::unsupported;
    if (!finite(charges[atom]) || !finite(charge_dot[atom])) return D4Status::invalid_argument;
    for (int axis = 0; axis < 3; ++axis)
      if (!finite(xyz[3 * atom + axis]) || !finite(direction[3 * atom + axis]))
        return D4Status::invalid_argument;
  }
  double* cn = workspace;
  double* cn_dot = cn + atoms;
  double* adj = cn_dot + atoms;
  double* adj_dot = adj + atoms;
  double* candidate_g = adj_dot + atoms;
  double* candidate_action = candidate_g + 3 * atoms;
  double* candidate_seed = candidate_action + 3 * atoms;
  double* candidate_seed_dot = candidate_seed + atoms;
  for (int index = 0; index < 12 * atoms; ++index) workspace[index] = 0;
  double* ref_weights = candidate_seed_dot + atoms;
  const auto panel = kD4MaximumReferences * atoms;
  WeightDirections weights{ref_weights,
                           ref_weights + panel,
                           ref_weights + 2 * panel,
                           ref_weights + 3 * panel,
                           ref_weights + 4 * panel,
                           ref_weights + 5 * panel};
  // Fix the hard pair/triple membership at the reference geometry. At a cutoff
  // equality the piecewise deployment cannot promise a differentiable Hessian.
  for (int first = 1; first < atoms; ++first)
    for (int second = 0; second < first; ++second) {
      const auto edge = edge_direction(xyz, direction, first, second);
      if (!finite(edge.squared) || edge.squared < 1e-12) return D4Status::invalid_argument;
      if (edge.distance == parameters.cn_cutoff || edge.distance == parameters.pair_cutoff ||
          (parameters.s9 != 0 && edge.distance == parameters.atm_cutoff))
        return D4Status::unsupported;
      if (edge.distance > parameters.cn_cutoff) continue;
      double value, slope;
      cn_pair(first, second, numbers, tables, edge.distance, value, slope);
      cn[first] += value;
      cn[second] += value;
      cn_dot[first] += slope * edge.distance_dot;
      cn_dot[second] += slope * edge.distance_dot;
    }

  auto status = reference_weight_directions(atoms, numbers, cn, cn_dot, charges, charge_dot,
                                            parameters, tables, weights);
  if (status != D4Status::success) return status;
  for (int first = 1; first < atoms; ++first)
    for (int second = 0; second < first; ++second) {
      const auto edge = edge_direction(xyz, direction, first, second);
      if (edge.distance > parameters.pair_cutoff) continue;
      const auto coefficient = coefficient_direction(first, second, numbers, tables, weights);
      const auto a = tables.elements[numbers[first] - 1], b = tables.elements[numbers[second] - 1];
      double damping[3];
      direction_scalar::bj_damping_second(edge.squared, math::pair_rr(a, b),
                                          math::damping_radius(a, b, parameters.a1, parameters.a2),
                                          parameters.s6, parameters.s8, damping);
      const double damping_dot = 0.5 * damping[1] * edge.squared_dot;
      const auto& value = coefficient.value;
      const auto& tangent = coefficient.tangent;
      candidate_seed[first] -= value.first_charge * damping[0];
      candidate_seed[second] -= value.second_charge * damping[0];
      candidate_seed_dot[first] -=
          tangent.first_charge * damping[0] + value.first_charge * damping_dot;
      candidate_seed_dot[second] -=
          tangent.second_charge * damping[0] + value.second_charge * damping_dot;
      adj[first] -= value.first_cn * damping[0];
      adj[second] -= value.second_cn * damping[0];
      adj_dot[first] -= tangent.first_cn * damping[0] + value.first_cn * damping_dot;
      adj_dot[second] -= tangent.second_cn * damping[0] + value.second_cn * damping_dot;
      scatter_direction(first, second, edge, -value.c6 * damping[1],
                        -tangent.c6 * damping[1] - value.c6 * damping[2] * edge.squared_dot,
                        candidate_g, candidate_action);
    }

  // ATM uses neutral reference weights: only the CN derivative survives.
  if (parameters.s9 != 0) {
    status = reference_weight_directions(atoms, numbers, cn, cn_dot, nullptr, nullptr, parameters,
                                         tables, weights);
    if (status != D4Status::success) return status;
    for (int first = 2; first < atoms; ++first)
      for (int second = 1; second < first; ++second) {
        const auto first_edge = edge_direction(xyz, direction, first, second);
        if (first_edge.distance > parameters.atm_cutoff) continue;
        const auto first_c6 = coefficient_direction(first, second, numbers, tables, weights);
        for (int third = 0; third < second; ++third) {
          const auto second_edge = edge_direction(xyz, direction, first, third);
          const auto third_edge = edge_direction(xyz, direction, second, third);
          if (second_edge.distance > parameters.atm_cutoff ||
              third_edge.distance > parameters.atm_cutoff)
            continue;
          const auto second_c6 = coefficient_direction(first, third, numbers, tables, weights);
          const auto third_c6 = coefficient_direction(second, third, numbers, tables, weights);
          if (!(first_c6.value.c6 > 0 && second_c6.value.c6 > 0 && third_c6.value.c6 > 0))
            return D4Status::numerical_failure;
          const double radius_product = radius(first, second, numbers, tables, parameters) *
                                        radius(first, third, numbers, tables, parameters) *
                                        radius(second, third, numbers, tables, parameters);
          double triplet[14];
          direction_scalar::atm_direction(
              first_edge.squared, second_edge.squared, third_edge.squared, radius_product,
              first_c6.value.c6, second_c6.value.c6, third_c6.value.c6, parameters.s9, 16.0,
              first_edge.squared_dot, second_edge.squared_dot, third_edge.squared_dot,
              first_c6.tangent.c6, second_c6.tangent.c6, third_c6.tangent.c6, triplet);
          const int from[] = {first, first, second}, to[] = {second, third, third};
          const EdgeDirection edges[] = {first_edge, second_edge, third_edge};
          const CoefficientDirection coefficients[] = {first_c6, second_c6, third_c6};
          for (int leg = 0; leg < 3; ++leg) {
            scatter_direction(from[leg], to[leg], edges[leg], triplet[1 + leg], triplet[8 + leg],
                              candidate_g, candidate_action);
            const auto& c = coefficients[leg];
            adj[from[leg]] += triplet[4 + leg] * c.value.first_cn;
            adj[to[leg]] += triplet[4 + leg] * c.value.second_cn;
            adj_dot[from[leg]] +=
                triplet[11 + leg] * c.value.first_cn + triplet[4 + leg] * c.tangent.first_cn;
            adj_dot[to[leg]] +=
                triplet[11 + leg] * c.value.second_cn + triplet[4 + leg] * c.tangent.second_cn;
          }
        }
      }
  }

  for (int first = 1; first < atoms; ++first)
    for (int second = 0; second < first; ++second) {
      const auto edge = edge_direction(xyz, direction, first, second);
      if (edge.distance > parameters.cn_cutoff) continue;
      const auto p = math::coordination_parameters(tables.elements[numbers[first] - 1],
                                                   tables.elements[numbers[second] - 1]);
      double jet[3];
      direction_scalar::d4_coordination_second(edge.distance, p.radius, p.electronegativity_factor,
                                               jet);
      const double inverse = 1 / edge.distance;
      const double coefficient = jet[1] * inverse;
      const double coefficient_dot = (jet[2] - coefficient) * edge.distance_dot * inverse;
      const double weight = adj[first] + adj[second];
      const double weight_dot = adj_dot[first] + adj_dot[second];
      scatter_direction(first, second, edge, weight * coefficient,
                        weight_dot * coefficient + weight * coefficient_dot, candidate_g,
                        candidate_action);
    }
  for (int index = 0; index < 3 * atoms; ++index)
    if (!finite(candidate_g[index]) || !finite(candidate_action[index]))
      return D4Status::numerical_failure;
  for (int index = 0; index < atoms; ++index)
    if (!finite(candidate_seed[index]) || !finite(candidate_seed_dot[index]))
      return D4Status::numerical_failure;
  for (int index = 0; index < 3 * atoms; ++index) {
    gradient[index] = candidate_g[index];
    action[index] = candidate_action[index];
  }
  for (int index = 0; index < atoms; ++index) {
    seed[index] = candidate_seed[index];
    seed_dot[index] = candidate_seed_dot[index];
  }
  return D4Status::success;
}

}  // namespace generativeqc::dft::dispersion

#undef GENERATIVEQC_D4_RESPONSE_HD
