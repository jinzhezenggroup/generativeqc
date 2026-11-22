#pragma once

#include "dft/dispersion/d3_model.hpp"

#if defined(__CUDACC__)
#define GENERATIVEQC_D3_RESPONSE_HD __host__ __device__
#else
#define GENERATIVEQC_D3_RESPONSE_HD
#endif

namespace generativeqc::dft::dispersion {

/** Linear workspace; no pair or triple matrix is retained. The first phase
 * writes per-atom reference jets, the second writes local CN cotangents and
 * direct forces, and the third applies the mixed CN pullback. */
GENERATIVEQC_D3_RESPONSE_HD inline std::size_t d3_hessian_workspace_elements(std::size_t atoms) {
  return atoms <= kD3MaximumAtomsPerSystem ? 24 * atoms : 0;
}

/** The native KS owner shares one CN graph between its pair and optional ATM
 * terms. The stand-alone reference can request different CN cutoffs. */
GENERATIVEQC_D3_RESPONSE_HD inline bool d3_hessian_supported(const D3ModelParameters& model) {
  return valid_d3_model(model) && (!model.atm_enabled || model.atm.cn_cutoff == model.bj.cn_cutoff);
}

namespace d3_response {
namespace scalar = ::generativeqc::generated::dispersion;
using d3_detail::finite;

struct Jet {
  double value{}, dot{};
  GENERATIVEQC_D3_RESPONSE_HD Jet& operator+=(Jet other) {
    value += other.value;
    dot += other.dot;
    return *this;
  }
};
GENERATIVEQC_D3_RESPONSE_HD inline Jet operator+(Jet first, Jet second) {
  return {first.value + second.value, first.dot + second.dot};
}
GENERATIVEQC_D3_RESPONSE_HD inline Jet operator*(Jet first, Jet second) {
  return {first.value * second.value, first.dot * second.value + first.value * second.dot};
}
GENERATIVEQC_D3_RESPONSE_HD inline Jet operator-(Jet term) { return {-term.value, -term.dot}; }

struct Work {
  double *weights, *first_cn, *second_cn, *cn_dot, *adjoint, *adjoint_dot;
};
GENERATIVEQC_D3_RESPONSE_HD inline Work workspace(double* data, std::size_t atoms) {
  return {data,
          data + 7 * atoms,
          data + 14 * atoms,
          data + 21 * atoms,
          data + 22 * atoms,
          data + 23 * atoms};
}

struct Edge {
  double delta[3]{}, velocity[3]{}, r2{}, r2_dot{}, radius{}, radius_dot{};
};
GENERATIVEQC_D3_RESPONSE_HD inline Edge edge(std::size_t first, std::size_t second,
                                             const double* xyz, const double* direction) {
  Edge out;
  for (int axis = 0; axis < 3; ++axis) {
    out.delta[axis] = xyz[3 * first + axis] - xyz[3 * second + axis];
    out.velocity[axis] = direction[3 * first + axis] - direction[3 * second + axis];
    out.r2 += out.delta[axis] * out.delta[axis];
    out.r2_dot += 2.0 * out.delta[axis] * out.velocity[axis];
  }
  out.radius = sqrt(out.r2);
  out.radius_dot = out.r2_dot / (2 * out.radius);
  return out;
}

GENERATIVEQC_D3_RESPONSE_HD inline bool outside(const Edge& bond, double cutoff) {
  return cutoff > 0 && bond.r2 > cutoff * cutoff;
}
GENERATIVEQC_D3_RESPONSE_HD inline bool hard_boundary(const Edge& bond, double cutoff,
                                                      double width = 0) {
  return cutoff > 0 && width == 0 && bond.r2 == cutoff * cutoff;
}
GENERATIVEQC_D3_RESPONSE_HD inline double cn_cutoff(const D3ModelParameters& model) {
  return model.damping == D3Damping::bj ? model.bj.cn_cutoff : model.zero.cn_cutoff;
}

struct Radial {
  Jet energy, scatter;
};
GENERATIVEQC_D3_RESPONSE_HD inline Radial weight(const Edge& bond, double cutoff, double width) {
  if (outside(bond, cutoff)) return {};
  if (cutoff == 0 || width == 0 || bond.radius <= cutoff - width) return {{1, 0}, {}};
  if (bond.radius >= cutoff) return {};
  double jet[3];
  scalar::smooth_switch_second(bond.radius, cutoff, width, jet);
  const auto scatter = jet[1] / bond.radius;
  return {{jet[0], jet[1] * bond.radius_dot},
          {scatter, (jet[2] - scatter) * bond.radius_dot / bond.radius}};
}

/** Construct per-atom first and second CN weight derivatives without forming
 * inverse powers of a tiny Gaussian norm. When the reciprocal norm overflows,
 * the primal max-reference fallback is locally constant. */
GENERATIVEQC_D3_RESPONSE_HD inline D3Status weights_for_atom(std::size_t atoms, std::size_t atom,
                                                             const std::int32_t* numbers,
                                                             const double* xyz,
                                                             const double* direction, double cutoff,
                                                             D3Tables tables, Work work) {
  if (numbers[atom] < 1 || numbers[atom] > 86) return D3Status::unsupported;
  const auto element = tables.elements[numbers[atom] - 1];
  if (!element.reference_count || element.reference_count > 7) return D3Status::unsupported;
  for (int axis = 0; axis < 3; ++axis)
    if (!finite(xyz[3 * atom + axis]) || !finite(direction[3 * atom + axis]))
      return D3Status::invalid_argument;
  Jet coordination{};
  for (std::size_t other = 0; other < atoms; ++other) {
    if (other == atom) continue;
    if (numbers[other] < 1 || numbers[other] > 86) return D3Status::unsupported;
    const auto bond = edge(atom, other, xyz, direction);
    if (!finite(bond.r2) || bond.r2 < 1e-12 || !finite(bond.r2_dot))
      return D3Status::numerical_failure;
    if (hard_boundary(bond, cutoff)) return D3Status::unsupported;
    if (outside(bond, cutoff)) continue;
    const auto radius =
        element.covalent_radius + tables.elements[numbers[other] - 1].covalent_radius;
    double jet[3];
    scalar::d3_coordination_second(bond.radius, radius, jet);
    coordination += {jet[0], jet[1] * (0.5 * bond.r2_dot)};
  }
  if (!finite(coordination.value) || !finite(coordination.dot)) return D3Status::numerical_failure;
  work.cn_dot[atom] = coordination.dot;
  double gaussian[7][3]{}, norm[3]{}, maximum = -DBL_MAX;
  for (int ref = 0; ref < element.reference_count; ++ref) {
    const auto center = tables.reference_cn[element.reference_offset + ref];
    maximum = fmax(maximum, center);
    scalar::gaussian_second(coordination.value, center, 4.0, gaussian[ref]);
    for (int slot = 0; slot < 3; ++slot) norm[slot] += gaussian[ref][slot];
  }
  const bool use_max_reference = norm[0] == 0 || !finite(1.0 / norm[0]);
  for (int ref = 0; ref < element.reference_count; ++ref) {
    const auto index = 7 * atom + ref;
    if (use_max_reference) {
      work.weights[index] = tables.reference_cn[element.reference_offset + ref] == maximum;
      work.first_cn[index] = work.second_cn[index] = 0;
      continue;
    }
    double normalized[2];
    scalar::normalize(gaussian[ref][0], norm[0], gaussian[ref][1], norm[1], normalized);
    work.weights[index] = normalized[0];
    work.first_cn[index] = normalized[1];
    work.second_cn[index] =
        (gaussian[ref][2] - normalized[0] * norm[2] - 2 * normalized[1] * norm[1]) / norm[0];
    if (!finite(work.weights[index]) || !finite(work.first_cn[index]) ||
        !finite(work.second_cn[index]))
      return D3Status::numerical_failure;
  }
  return D3Status::success;
}

struct Coefficient {
  Jet total, first_cn;
};
GENERATIVEQC_D3_RESPONSE_HD inline Coefficient coefficient(std::size_t first, std::size_t second,
                                                           const std::int32_t* numbers,
                                                           D3Tables tables, Work work) {
  const auto first_element = tables.elements[numbers[first] - 1];
  const auto second_element = tables.elements[numbers[second] - 1];
  Coefficient result{};
  for (int left = 0; left < first_element.reference_count; ++left) {
    const auto i = 7 * first + left;
    const Jet wi{work.weights[i], work.first_cn[i] * work.cn_dot[first]};
    const Jet dwi{work.first_cn[i], work.second_cn[i] * work.cn_dot[first]};
    for (int right = 0; right < second_element.reference_count; ++right) {
      const auto j = 7 * second + right;
      const auto reference =
          d3_detail::reference_c6(tables, numbers[first], numbers[second], left, right);
      const Jet wj{work.weights[j], work.first_cn[j] * work.cn_dot[second]};
      result.total += wi * wj * Jet{reference, 0};
      result.first_cn += dwi * wj * Jet{reference, 0};
    }
  }
  return result;
}

GENERATIVEQC_D3_RESPONSE_HD inline void scatter(const Edge& edge, Jet force, double* action) {
  for (int axis = 0; axis < 3; ++axis)
    action[axis] += force.dot * edge.delta[axis] + force.value * edge.velocity[axis];
}

GENERATIVEQC_D3_RESPONSE_HD inline Radial pair_radial(const Edge& bond, double rr, double r0,
                                                      const D3ModelParameters& model) {
  double jet[3];
  Radial raw{}, damping{};
  if (model.damping == D3Damping::bj) {
    const auto& p = model.bj;
    scalar::bj_damping_second(bond.r2, rr, p.a1 * sqrt(rr) + p.a2, p.s6, p.s8, jet);
    raw = {{jet[0], 0.5 * jet[1] * bond.r2_dot}, {jet[1], jet[2] * bond.r2_dot}};
    damping = weight(bond, p.pair_cutoff, p.pair_switch_width);
  } else {
    const auto& p = model.zero;
    double term8[3];
    scalar::zero_damping_second(bond.radius, p.rs6 * r0, p.alpha6, 6, jet);
    scalar::zero_damping_second(bond.radius, p.rs8 * r0, p.alpha6 + 2, 8, term8);
    for (int slot = 0; slot < 3; ++slot) jet[slot] = p.s6 * jet[slot] + p.s8 * rr * term8[slot];
    raw = {{jet[0], 0.5 * jet[1] * bond.r2_dot}, {jet[1], jet[2] * bond.radius_dot}};
    damping = weight(bond, p.pair_cutoff, p.pair_switch_width);
  }
  return {raw.energy * damping.energy, raw.scatter * damping.energy + raw.energy * damping.scatter};
}

/** Each atom owns only its incident gradient legs and CN cotangent. An ATM
 * triangle is visited once per vertex; all three edge tangents are needed even
 * when the opposite edge supplies no gradient or CN derivative to this atom. */
GENERATIVEQC_D3_RESPONSE_HD inline D3Status local_response(
    std::size_t atoms, std::size_t atom, const std::int32_t* numbers, const double* xyz,
    const double* direction, const D3ModelParameters& model, D3Tables tables, Work work,
    double* action) {
  Jet adjoint{};
  for (int axis = 0; axis < 3; ++axis) action[axis] = 0;
  const bool bj = model.damping == D3Damping::bj;
  const auto cutoff = bj ? model.bj.pair_cutoff : model.zero.pair_cutoff;
  const auto width = bj ? model.bj.pair_switch_width : model.zero.pair_switch_width;
  const auto element = tables.elements[numbers[atom] - 1];
  for (std::size_t other = 0; other < atoms; ++other) {
    if (other == atom) continue;
    const auto first = edge(atom, other, xyz, direction);
    if (hard_boundary(first, cutoff, width)) return D3Status::unsupported;
    if (!outside(first, cutoff)) {
      const auto other_element = tables.elements[numbers[other] - 1];
      const auto rr = 3 * element.r4r2 * other_element.r4r2;
      const auto r0 = tables.pairs[d3_detail::pair_index(numbers[atom], numbers[other])].vdw_radius;
      const auto radial = pair_radial(first, rr, r0, model);
      const auto c6 = coefficient(atom, other, numbers, tables, work);
      adjoint += -(c6.first_cn * radial.energy);
      scatter(first, -(c6.total * radial.scatter), action);
    }
    if (!model.atm_enabled || atoms < 3) continue;
    const auto& atm = model.atm;
    if (outside(first, atm.atm_cutoff)) continue;
    for (std::size_t third = 0; third < other; ++third) {
      if (third == atom) continue;
      const Edge edges[]{first, edge(atom, third, xyz, direction),
                         edge(other, third, xyz, direction)};
      if (outside(edges[1], atm.atm_cutoff) || outside(edges[2], atm.atm_cutoff)) continue;
      for (const auto& edge : edges)
        if (hard_boundary(edge, atm.atm_cutoff, atm.atm_switch_width)) return D3Status::unsupported;
      const Coefficient c6[]{coefficient(atom, other, numbers, tables, work),
                             coefficient(atom, third, numbers, tables, work),
                             coefficient(other, third, numbers, tables, work)};
      const int labels[][2]{{numbers[atom], numbers[other]},
                            {numbers[atom], numbers[third]},
                            {numbers[other], numbers[third]}};
      Radial switches[3];
      double base_radius = 1;
      for (int side = 0; side < 3; ++side) {
        if (!(c6[side].total.value > 0)) return D3Status::numerical_failure;
        const auto pair = tables.pairs[d3_detail::pair_index(labels[side][0], labels[side][1])];
        base_radius *= kD3BjAtmRs9 * pair.vdw_radius;
        switches[side] = weight(edges[side], atm.atm_cutoff, atm.atm_switch_width);
      }
      double response[14];
      scalar::atm_direction(edges[0].r2, edges[1].r2, edges[2].r2, base_radius, c6[0].total.value,
                            c6[1].total.value, c6[2].total.value, atm.s9, kD3BjAtmAlpha,
                            edges[0].r2_dot, edges[1].r2_dot, edges[2].r2_dot, c6[0].total.dot,
                            c6[1].total.dot, c6[2].total.dot, response);
      const Jet product = switches[0].energy * switches[1].energy * switches[2].energy;
      const Jet energy{response[0], response[7]};
      for (int side = 0; side < 2; ++side) {
        const Jet radial{response[1 + side], response[8 + side]};
        const Jet partial_c6{response[4 + side], response[11 + side]};
        const Jet switch_scatter =
            switches[side].scatter * switches[1 - side].energy * switches[2].energy;
        adjoint += product * partial_c6 * c6[side].first_cn;
        scatter(edges[side], radial * product + energy * switch_scatter, action);
      }
    }
  }
  work.adjoint[atom] = adjoint.value;
  work.adjoint_dot[atom] = adjoint.dot;
  return finite(adjoint.value) && finite(adjoint.dot) ? D3Status::success
                                                      : D3Status::numerical_failure;
}

GENERATIVEQC_D3_RESPONSE_HD inline D3Status cn_pullback(std::size_t atoms, std::size_t atom,
                                                        const std::int32_t* numbers,
                                                        const double* xyz, const double* direction,
                                                        double cutoff, D3Tables tables, Work work,
                                                        double* action) {
  const auto covalent = tables.elements[numbers[atom] - 1].covalent_radius;
  for (std::size_t other = 0; other < atoms; ++other) {
    if (other == atom) continue;
    const auto bond = edge(atom, other, xyz, direction);
    if (outside(bond, cutoff)) continue;
    double jet[3];
    scalar::d3_coordination_second(
        bond.radius, covalent + tables.elements[numbers[other] - 1].covalent_radius, jet);
    const Jet adjoint{work.adjoint[atom] + work.adjoint[other],
                      work.adjoint_dot[atom] + work.adjoint_dot[other]};
    scatter(bond, adjoint * Jet{jet[1], jet[2] * bond.radius_dot}, action);
  }
  for (int axis = 0; axis < 3; ++axis)
    if (!finite(action[axis])) return D3Status::numerical_failure;
  return D3Status::success;
}
}  // namespace d3_response
}  // namespace generativeqc::dft::dispersion

#undef GENERATIVEQC_D3_RESPONSE_HD
