#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "core/types.hpp"

namespace generativeqc::integrals {

/** Bounds on the Coulomb norms of a contracted Cartesian AO pair and its
 * first/mixed nuclear tangents. Valid for unit Cartesian nuclear coordinates
 * (including repeated centers), normalized real s/p shell coefficients and
 * round-to-nearest IEEE binary64. No integrals or reference code run here. */
struct CoulombDerivativeEnvelope {
  using Polynomial = std::array<double, 4>;
  using Orders = std::array<double, 3>;
  static_assert(std::numeric_limits<double>::is_iec559);

  static double upper(double value) noexcept { return std::nextafter(value, INFINITY); }
  static double lower(double value) noexcept { return std::max(0.0, std::nextafter(value, 0.0)); }

  /** The returned bound counts all ordered AO components in both shell pairs. */
  static std::optional<Orders> pair(const core::Shell& first, const core::Shell& second,
                                   const std::array<double, 3>& center_a,
                                   const std::array<double, 3>& center_b) noexcept {
    if (first.angular_momentum > 1 || second.angular_momentum > 1 ||
        first.primitives.empty() || second.primitives.empty())
      return std::nullopt;
    double distance_squared_lower{}, distance_squared_upper{};
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (!(std::abs(center_a[axis]) <= 0x1p20 && std::abs(center_b[axis]) <= 0x1p20))
        return std::nullopt;
      const auto distance = std::abs(center_a[axis] - center_b[axis]);
      const auto lo = lower(distance), hi = upper(distance);
      distance_squared_lower = lower(distance_squared_lower + lower(lo * lo));
      distance_squared_upper = upper(distance_squared_upper + upper(hi * hi));
    }
    const auto distance = upper(std::sqrt(distance_squared_upper));
    Orders bounds{};
    for (const auto& primitive_a : first.primitives)
      for (const auto& primitive_b : second.primitives) {
        const auto alpha = primitive_a.exponent, beta = primitive_b.exponent;
        const auto ca = std::abs(primitive_a.coefficient), cb = std::abs(primitive_b.coefficient);
        if (!(alpha >= 0x1p-20 && alpha <= 0x1p20 && beta >= 0x1p-20 && beta <= 0x1p20 &&
              ca <= 0x1p20 && cb <= 0x1p20))
          return std::nullopt;
        const auto total = alpha + beta;
        const auto total_lower = lower(total);
        const auto shift_a = upper(upper(beta / total_lower) * distance);
        const auto shift_b = upper(upper(alpha / total_lower) * distance);
        const auto decay = exp_negative_upper(lower(lower(lower(alpha * beta) / upper(total)) *
                                                     distance_squared_lower));
        const auto scale = upper(upper(ca * cb) * decay);
        const auto left = orbital_polynomials(first.angular_momentum, alpha, shift_a);
        const auto right = orbital_polynomials(second.angular_momentum, beta, shift_b);
        const auto denominator = lower(lower(total_lower * total_lower) *
                                       lower(std::sqrt(lower(2 * total_lower))));
        constexpr double twice_pi_power_upper = 34.986836656;
        const auto norm = upper(std::sqrt(upper(twice_pi_power_upper / denominator)));
        // Max squared Coulomb norms of Cartesian monomials of each degree,
        // relative to the constant monomial at exponent one. Integrating the
        // Coulomb Gaussian representation leaves a uniform u in [0,1], with
        // diagonal variance (2-u^2)/4 and covariance u^2/4. For degrees <=4,
        // the largest integrated Wick moment has all powers along one axis.
        constexpr std::array squared_ratios{1.0, 1.0 / 12, 49.0 / 240,
                                            243.0 / 2240, 11449.0 / 26880};
        std::array<double, 5> moments;
        double exponent_power = 1;
        for (unsigned degree = 0; degree < moments.size(); ++degree) {
          // Use the triangle inequality on signed monomials, not a radial
          // pointwise envelope. The coefficients above bound absolute sums
          // at each degree, and Gaussian scaling contributes p^(-degree/2).
          moments[degree] = upper(norm * upper(std::sqrt(upper(
              upper(squared_ratios[degree]) / exponent_power))));
          exponent_power = lower(exponent_power * total_lower);
        }
        for (unsigned order = 0; order < bounds.size(); ++order) {
          double sum{};
          for (unsigned leg = 0; leg <= order; ++leg) {
            const auto left_degree = first.angular_momentum + leg;
            const auto right_degree = second.angular_momentum + order - leg;
            for (unsigned ali = 0; ali <= left_degree; ++ali)
              for (unsigned bli = 0; bli <= right_degree; ++bli) {
                auto term = upper(upper(left[leg][ali] * right[order - leg][bli]) *
                                  moments[ali + bli]);
                if (order == 2 && leg == 1) term = upper(2 * term);
                sum = upper(sum + term);
              }
          }
          bounds[order] = upper(bounds[order] + upper(scale * sum));
        }
      }
    const unsigned components = (2 * first.angular_momentum + 1) * (2 * second.angular_momentum + 1);
    for (double& bound : bounds) bound = upper(bound * components);
    return bounds;
  }

  /** Bound the sum of absolute omitted derivatives for a canonical source's
   * complete eightfold orbit. Coulomb Cauchy--Schwarz and the product rule
   * apply to the separate left/right pair tangents; use the larger order. */
  static double quartet(const Orders& left, const Orders& right) noexcept {
    const auto first = upper(upper(left[1] * right[0]) + upper(left[0] * right[1]));
    const auto mixed = upper(upper(upper(left[2] * right[0]) + upper(2 * upper(left[1] * right[1]))) +
                             upper(left[0] * right[2]));
    return upper(8 * std::max(first, mixed));
  }

 private:
  static double exp_negative_upper(double exponent_lower) noexcept {
    // A positive partial sum bounds exp from below without assuming libm exp
    // rounds in a particular direction. Clamping weakens, never tightens it.
    const auto argument = std::min(128.0, exponent_lower);
    double term = 1, sum = 1;
    for (unsigned order = 1; order <= 64; ++order) {
      term = lower(lower(term * argument) / order);
      sum = lower(sum + term);
    }
    return upper(1 / sum);
  }

  static std::array<Polynomial, 3> orbital_polynomials(unsigned angular, double exponent,
                                                        double shift) noexcept {
    std::array<Polynomial, 4> powers{{{1}}};
    for (unsigned order = 1; order < powers.size(); ++order)
      for (unsigned degree = 0; degree <= order; ++degree) {
        const auto constant = upper(powers[order - 1][degree] * shift);
        powers[order][degree] = degree ? upper(constant + powers[order - 1][degree - 1]) : constant;
      }
    std::array<Polynomial, 3> coefficients{};
    coefficients[0] = powers[angular];
    const auto twice = upper(2 * exponent), square = upper(twice * twice);
    const auto lower_order = upper((2 + 4 * angular) * exponent);
    for (unsigned degree = 0; degree <= angular + 2; ++degree) {
      coefficients[1][degree] = upper(twice * powers[angular + 1][degree]);
      coefficients[2][degree] = upper(upper(square * powers[angular + 2][degree]) +
                                      upper(lower_order * powers[angular][degree]));
    }
    if (angular) coefficients[1][0] = upper(coefficients[1][0] + 1);
    return coefficients;
  }
};

using CoulombSource = std::array<std::size_t, 4>;

struct CoulombDerivativeOmissions {
  std::vector<CoulombSource> sources;
  double bound{};
  std::size_t candidate_checks{};
  // Guard the density weights on device; this is the largest unweighted
  // discarded derivative bound over individual frozen unit columns.
  static constexpr double max_bound = 1e-14;
  static constexpr double max_density_element = 4;
  static constexpr double max_response_element = 16;
};

/** Select whole canonical orbits. Use a separate absolute ledger for each
 * right atom: an orbit has no derivative in columns owned by other atoms.
 * Each ledger covers all three Cartesian axes and all left rows before
 * density contraction. The candidate limit bounds pair storage and the
 * rank/individual-quartet checks independently of the theoretical census. */
inline CoulombDerivativeOmissions coulomb_derivative_omissions(const core::System& system,
                                                                double budget,
                                                                std::size_t candidate_limit) {
  using Envelope = CoulombDerivativeEnvelope;
  CoulombDerivativeOmissions result;
  const auto shell_count = system.shells.size();
  if (system.basis_representation != core::kCartesianBasis || !shell_count ||
      !std::isfinite(budget) || budget <= 0 || shell_count == SIZE_MAX ||
      shell_count > SIZE_MAX / (shell_count + 1))
    return result;
  const auto pair_count = shell_count * (shell_count + 1) / 2;
  if (pair_count > candidate_limit || system.atoms.size() > candidate_limit) return result;
  struct Pair {
    std::array<std::size_t, 2> shells;
    Envelope::Orders bounds;
  };
  const auto& shells = system.shells;
  std::array<std::vector<Pair>, 3> by_dominant_order;
  for (std::size_t first = 0; first < shell_count; ++first)
    for (std::size_t second = first; second < shell_count; ++second) {
      const auto& left = shells[first];
      const auto& right = shells[second];
      if (left.primitives.size() > 16 || right.primitives.size() > 16 ||
          left.atom_index >= system.atoms.size() || right.atom_index >= system.atoms.size())
        return {};
      const auto bound = Envelope::pair(left, right, system.atoms[left.atom_index].position,
                                        system.atoms[right.atom_index].position);
      if (!bound) return {};
      if (!std::all_of(bound->begin(), bound->end(), [](double value) {
            return std::isfinite(value);
          }))
        return {};
      const auto order = std::max_element(bound->begin(), bound->end()) - bound->begin();
      by_dominant_order[order].push_back({{first, second}, *bound});
    }
  for (unsigned order = 0; order < by_dominant_order.size(); ++order) {
    auto& pairs = by_dominant_order[order];
    std::sort(pairs.begin(), pairs.end(), [order](const Pair& left, const Pair& right) {
      if (left.bounds[order] != right.bounds[order])
        return left.bounds[order] < right.bounds[order];
      return left.shells < right.shells;
    });
  }
  struct Candidate {
    double bound;
    CoulombSource source;
  };
  std::vector<Candidate> candidates;
  const auto charge = [&](std::size_t checks) {
    if (checks > candidate_limit - result.candidate_checks) return false;
    result.candidate_checks += checks;
    return true;
  };
  for (unsigned left_order = 0; left_order < by_dominant_order.size(); ++left_order) {
    const auto& left_pairs = by_dominant_order[left_order];
    for (std::size_t index = 0; index < left_pairs.size(); ++index) {
      const auto& left = left_pairs[index];
      for (unsigned right_order = left_order; right_order < by_dominant_order.size();
           ++right_order) {
        const auto& right_pairs = by_dominant_order[right_order];
        const auto begin = right_pairs.begin() + (left_order == right_order ? index : 0);
        if (!charge(std::bit_width(static_cast<std::size_t>(right_pairs.end() - begin))))
          return {};
        // The dominant coordinate contributes at least one positive term to
        // the outward quartet envelope. Its monotone product rules out the
        // suffix without evaluating the other terms. The central 2* term is
        // intentionally not needed for this necessary condition.
        const auto coefficient = right_order == 0 ? std::max(left.bounds[1], left.bounds[2])
                               : right_order == 1 ? std::max(left.bounds[0], left.bounds[1])
                                                  : left.bounds[0];
        const auto eligible = std::partition_point(begin, right_pairs.end(), [&](const Pair& right) {
          return Envelope::upper(8 * Envelope::upper(coefficient * right.bounds[right_order])) <=
                 budget;
        });
        if (!charge(static_cast<std::size_t>(eligible - begin))) return {};
        for (auto iter = begin; iter != eligible; ++iter) {
          const auto& first = left.shells < iter->shells ? left : *iter;
          const auto& second = left.shells < iter->shells ? *iter : left;
          const auto bound = Envelope::quartet(first.bounds, second.bounds);
          if (bound <= budget)
            candidates.push_back({bound, {first.shells[0], first.shells[1],
                                         second.shells[0], second.shells[1]}});
        }
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
    if (left.bound != right.bound) return left.bound < right.bound;
    return left.source < right.source;
  });
  std::vector<double> by_atom(system.atoms.size());
  for (const auto& candidate : candidates) {
    std::array<std::size_t, 4> atoms;
    for (unsigned slot = 0; slot < atoms.size(); ++slot)
      atoms[slot] = shells[candidate.source[slot]].atom_index;
    std::sort(atoms.begin(), atoms.end());
    const auto end = std::unique(atoms.begin(), atoms.end());
    const auto fits = [&](std::size_t atom) {
      return Envelope::upper(by_atom[atom] + candidate.bound) <= budget;
    };
    if (!std::all_of(atoms.begin(), end, fits)) continue;
    for (auto iter = atoms.begin(); iter != end; ++iter) {
      auto& total = by_atom[*iter];
      total = Envelope::upper(total + candidate.bound);
      result.bound = std::max(result.bound, total);
    }
    result.sources.push_back(candidate.source);
  }
  std::sort(result.sources.begin(), result.sources.end());
  return result;
}

}  // namespace generativeqc::integrals
