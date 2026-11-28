#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
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
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53 &&
                std::numeric_limits<double>::max_exponent == 1024);
  static constexpr auto infinity_bits =
      std::bit_cast<std::uint64_t>(std::numeric_limits<double>::infinity());

  // Nonnegative finite binary64 encodings increase by one per ULP, including
  // across the subnormal boundary. Keep the library semantics for other inputs.
  static double upper(double value) noexcept {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    if (bits < infinity_bits) return std::bit_cast<double>(bits + 1);
    return std::nextafter(value, INFINITY);
  }
  static double lower(double value) noexcept {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    if (bits <= infinity_bits) return std::bit_cast<double>(bits - (bits != 0));
    return std::max(0.0, std::nextafter(value, 0.0));
  }

  /** The returned bound counts all ordered AO components in both shell pairs. */
  static std::optional<Orders> pair(const core::Shell& first, const core::Shell& second,
                                    const std::array<double, 3>& center_a,
                                    const std::array<double, 3>& center_b) noexcept {
    if (first.angular_momentum > 1 || second.angular_momentum > 1 || first.primitives.empty() ||
        second.primitives.empty())
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
        const auto decay = exp_negative_upper(
            lower(lower(lower(alpha * beta) / upper(total)) * distance_squared_lower));
        const auto scale = upper(upper(ca * cb) * decay);
        const auto left = orbital_polynomials(first.angular_momentum, alpha, shift_a);
        const auto right = orbital_polynomials(second.angular_momentum, beta, shift_b);
        const auto denominator =
            lower(lower(total_lower * total_lower) * lower(std::sqrt(lower(2 * total_lower))));
        constexpr double twice_pi_power_upper = 34.986836656;
        const auto norm = upper(std::sqrt(upper(twice_pi_power_upper / denominator)));
        // Max squared Coulomb norms of Cartesian monomials of each degree,
        // relative to the constant monomial at exponent one. Integrating the
        // Coulomb Gaussian representation leaves a uniform u in [0,1], with
        // diagonal variance (2-u^2)/4 and covariance u^2/4. For degrees <=4,
        // the largest integrated Wick moment has all powers along one axis.
        constexpr std::array squared_ratios{1.0, 1.0 / 12, 49.0 / 240, 243.0 / 2240,
                                            11449.0 / 26880};
        std::array<double, 5> moments;
        double exponent_power = 1;
        for (unsigned degree = 0; degree < moments.size(); ++degree) {
          // Use the triangle inequality on signed monomials, not a radial
          // pointwise envelope. The coefficients above bound absolute sums
          // at each degree, and Gaussian scaling contributes p^(-degree/2).
          moments[degree] =
              upper(norm * upper(std::sqrt(upper(upper(squared_ratios[degree]) / exponent_power))));
          exponent_power = lower(exponent_power * total_lower);
        }
        for (unsigned order = 0; order < bounds.size(); ++order) {
          double sum{};
          for (unsigned leg = 0; leg <= order; ++leg) {
            const auto left_degree = first.angular_momentum + leg;
            const auto right_degree = second.angular_momentum + order - leg;
            for (unsigned ali = 0; ali <= left_degree; ++ali)
              for (unsigned bli = 0; bli <= right_degree; ++bli) {
                auto term =
                    upper(upper(left[leg][ali] * right[order - leg][bli]) * moments[ali + bli]);
                if (order == 2 && leg == 1) term = upper(2 * term);
                sum = upper(sum + term);
              }
          }
          bounds[order] = upper(bounds[order] + upper(scale * sum));
        }
      }
    const unsigned components =
        (2 * first.angular_momentum + 1) * (2 * second.angular_momentum + 1);
    for (double& bound : bounds) bound = upper(bound * components);
    return bounds;
  }

  struct RightAtomBounds {
    double left, right, both;

    double operator()(bool on_left, bool on_right) const noexcept {
      return on_left ? (on_right ? both : left) : (on_right ? right : 0);
    }
  };

  /** Bound the complete eightfold orbit for an individual frozen column.
   * The arbitrary left coordinate can act on either pair, but a fixed
   * right atom cannot differentiate an AO pair that does not own it. Pair
   * exchange in the orbit only exchanges which factor owns the derivative. */
  static RightAtomBounds quartet(const Orders& left, const Orders& right) noexcept {
    const auto first_left = upper(left[1] * right[0]);
    const auto first_right = upper(left[0] * right[1]);
    const auto cross = upper(left[1] * right[1]);
    const auto mixed_left = upper(upper(left[2] * right[0]) + cross);
    const auto mixed_right = upper(upper(left[0] * right[2]) + cross);
    return {upper(8 * std::max(first_left, mixed_left)),
            upper(8 * std::max(first_right, mixed_right)),
            upper(8 * std::max(upper(first_left + first_right), upper(mixed_left + mixed_right)))};
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
  std::vector<std::uint64_t> omitted_rank_words;
  std::size_t rank_count{}, omitted_rank_count{};
  double bound{};
  std::size_t candidate_checks{};

  [[nodiscard]] std::size_t size() const noexcept {
    return omitted_rank_words.empty() ? sources.size() : omitted_rank_count;
  }
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
 * rank/individual-quartet checks independently of the theoretical census.
 * A bitmap-aware consumer can avoid expanding accepted ranks to shell IDs;
 * bitmap allocation refusal still returns the bounded sorted source list. */
inline CoulombDerivativeOmissions coulomb_derivative_omissions(const core::System& system,
                                                               double budget,
                                                               std::size_t candidate_limit,
                                                               bool prefer_rank_bitmap = false) {
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
    std::size_t ordinal;
  };
  const auto& shells = system.shells;
  std::vector<std::array<std::size_t, 2>> canonical_pairs;
  canonical_pairs.reserve(pair_count);
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
      if (!std::all_of(bound->begin(), bound->end(),
                       [](double value) { return std::isfinite(value); }))
        return {};
      const auto order = std::max_element(bound->begin(), bound->end()) - bound->begin();
      by_dominant_order[order].push_back({{first, second}, *bound, canonical_pairs.size()});
      canonical_pairs.push_back({first, second});
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
    std::array<std::size_t, 2> pairs;
    Envelope::RightAtomBounds columns;
  };
  const auto owners = [&](const Candidate& candidate) {
    const auto& left = canonical_pairs[candidate.pairs[0]];
    const auto& right = canonical_pairs[candidate.pairs[1]];
    return std::array{shells[left[0]].atom_index, shells[left[1]].atom_index,
                      shells[right[0]].atom_index, shells[right[1]].atom_index};
  };
  const auto source = [&](const auto& pairs) -> CoulombSource {
    const auto& left = canonical_pairs[pairs[0]];
    const auto& right = canonical_pairs[pairs[1]];
    return {left[0], left[1], right[0], right[1]};
  };
  // A bounded bitmap publishes stable lexicographic order without sorting
  // accepted orbits. Sparse cases and allocation refusal retain the output sort.
  const bool rank_representable =
      pair_count < SIZE_MAX && pair_count <= SIZE_MAX / (pair_count + 1);
  const auto ranks = rank_representable ? pair_count * (pair_count + 1) / 2 : 0;
  const auto words = ranks ? 1 + (ranks - 1) / 64 : 0;
  constexpr std::size_t max_ranks_per_check = 2;
  std::vector<std::uint64_t> accepted;
  if (words <= candidate_limit / (64 / max_ranks_per_check)) {
    try {
      accepted.resize(words);
    } catch (const std::bad_alloc&) {
      // A missing optional bitmap is not a certificate failure.
    }
  }
  std::size_t accepted_count{};
  const auto accept = [&](std::array<std::size_t, 2> pairs) {
    if (accepted.empty()) {
      result.sources.push_back(source(pairs));
    } else {
      const auto row = pairs[0];
      const auto rank = row * (2 * pair_count - row + 1) / 2 + pairs[1] - row;
      accepted[rank / 64] |= std::uint64_t{1} << (rank % 64);
      ++accepted_count;
    }
  };
  // Charge all tiny sources against every atom at once. A directed upper bound
  // on the number of possible ranks keeps it below 1/16 of the budget;
  // the normal owner-specific ledger still decides the remaining omissions.
  const auto count_upper = Envelope::upper(static_cast<double>(ranks));
  const auto tiny_cutoff =
      accepted.empty() ? 0.0 : Envelope::lower(Envelope::lower(budget / count_upper) * 0x1p-4);
  std::size_t tiny_count{};
  // Nonnegative binary64 bounds in distinct exponent bins have strict
  // order. Keep the original bound/source tie-break within each bin.
  constexpr unsigned exponent_count = 2048;
  std::vector<std::vector<Candidate>> candidates(exponent_count);
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
        if (!charge(std::bit_width(static_cast<std::size_t>(right_pairs.end() - begin)))) return {};
        // At least one right atom is affected by each of these product-rule
        // terms, even if the pairs have disjoint owners. Its monotone bound
        // rules out the suffix before visiting candidate quartets.
        const auto coefficient = right_order == 0   ? std::max(left.bounds[1], left.bounds[2])
                                 : right_order == 1 ? std::max(left.bounds[0], left.bounds[1])
                                                    : left.bounds[0];
        const auto eligible =
            std::partition_point(begin, right_pairs.end(), [&](const Pair& right) {
              return Envelope::upper(
                         8 * Envelope::upper(coefficient * right.bounds[right_order])) <= budget;
            });
        auto iter = begin;
        const auto eligible_count = static_cast<std::size_t>(eligible - begin);
        if (!charge(eligible_count)) return {};
        if (tiny_cutoff && eligible_count >= 32 && charge(std::bit_width(eligible_count))) {
          // Both-owner bounds dominate either single owner. Replacing all
          // three right bounds by their dominant one produces a monotone test
          // in the existing right-pair order, without evaluating each quartet.
          const auto tiny_end = std::partition_point(begin, eligible, [&](const Pair& right) {
            const auto dominant = right.bounds[right_order];
            return Envelope::quartet(left.bounds, {dominant, dominant, dominant}).both <=
                   tiny_cutoff;
          });
          tiny_count += static_cast<std::size_t>(tiny_end - begin);
          for (; iter != tiny_end; ++iter)
            accept({std::min(left.ordinal, iter->ordinal), std::max(left.ordinal, iter->ordinal)});
        }
        for (; iter != eligible; ++iter) {
          const auto& first = left.shells < iter->shells ? left : *iter;
          const auto& second = left.shells < iter->shells ? *iter : left;
          Candidate candidate{
              0, {first.ordinal, second.ordinal}, Envelope::quartet(first.bounds, second.bounds)};
          const auto atoms = owners(candidate);
          const bool shared = atoms[0] == atoms[2] || atoms[0] == atoms[3] ||
                              atoms[1] == atoms[2] || atoms[1] == atoms[3];
          candidate.bound = shared ? candidate.columns.both
                                   : std::max(candidate.columns.left, candidate.columns.right);
          if (candidate.bound <= budget) {
            const auto exponent = std::bit_cast<std::uint64_t>(candidate.bound) >> 52;
            candidates[exponent].push_back(candidate);
          }
        }
      }
    }
  }
  if (tiny_count) {
    result.bound = Envelope::upper(Envelope::upper(static_cast<double>(tiny_count)) * tiny_cutoff);
    if (result.bound > budget) return {};
  }
  std::vector<double> by_atom(system.atoms.size(), result.bound);
  for (auto& bin : candidates) {
    std::sort(bin.begin(), bin.end(), [](const auto& left, const auto& right) {
      if (left.bound != right.bound) return left.bound < right.bound;
      return left.pairs < right.pairs;
    });
    for (const auto& candidate : bin) {
      const auto atoms = owners(candidate);
      std::array<double, 4> next;
      unsigned present{};
      bool fits = true;
      for (unsigned slot = 0; slot < atoms.size(); ++slot) {
        const auto atom = atoms[slot];
        if (std::find(atoms.begin(), atoms.begin() + slot, atom) != atoms.begin() + slot) continue;
        const auto column = candidate.columns(atoms[0] == atom || atoms[1] == atom,
                                              atoms[2] == atom || atoms[3] == atom);
        next[slot] = Envelope::upper(by_atom[atom] + column);
        if (next[slot] > budget) {
          fits = false;
          break;
        }
        present |= 1U << slot;
      }
      if (!fits) continue;
      for (unsigned slot = 0; slot < atoms.size(); ++slot) {
        if (!(present & (1U << slot))) continue;
        by_atom[atoms[slot]] = next[slot];
        result.bound = std::max(result.bound, next[slot]);
      }
      accept(candidate.pairs);
    }
    std::vector<Candidate>{}.swap(bin);
  }
  if (accepted.empty()) {
    std::sort(result.sources.begin(), result.sources.end());
  } else if (prefer_rank_bitmap) {
    result.omitted_rank_words = std::move(accepted);
    result.rank_count = ranks;
    result.omitted_rank_count = accepted_count;
  } else {
    result.sources.reserve(accepted_count);
    std::size_t first{}, row{};
    for (std::size_t word = 0; word < accepted.size(); ++word) {
      for (auto bits = accepted[word]; bits; bits &= bits - 1) {
        const auto rank = 64 * word + std::countr_zero(bits);
        while (rank >= row + pair_count - first) {
          row += pair_count - first;
          ++first;
        }
        result.sources.push_back(source(std::array{first, first + rank - row}));
      }
    }
  }
  return result;
}

}  // namespace generativeqc::integrals
