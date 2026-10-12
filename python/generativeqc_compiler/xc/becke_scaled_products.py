"""Binary-scaled lowering of ordered Becke products.

The canonical switch, ratio AD, normalization objective and geometric gather
remain authoritative. Only the product representation changes: a positive
mantissa and an integer exponent replace an ordered sum of logarithms. Exact
zeros remain separate, including the single-zero excluded-product derivative.

Native source admits this representation only with ordinary phased storage,
center geometry, at most 128 atoms, and no primitive-adjoint scratch alias.
Field 7 holds absolute exponents before normalization and packed relative
exponents until pair reverse finishes, so this lowering must not
be combined with the normalized-adjoint lowering that borrows the same field.
The existing four-word pair panel and all seven phase boundaries are retained.
"""

from __future__ import annotations

_SCALED_SOURCE = r"""
#include <cstdint>
#include <cstring>
#if defined(__CUDACC__)
#define GENERATIVEQC_SCALED_HD __host__ __device__
#else
#define GENERATIVEQC_SCALED_HD
#endif
namespace generativeqc_grid_scaled {
using namespace generativeqc_grid_phased;
constexpr size_t maximum_scaled_atoms = 128;

GENERATIVEQC_SCALED_HD inline std::uint64_t binary64_bits(double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53,
                "binary-scaled products require IEEE binary64");
#if defined(__CUDA_ARCH__)
  return static_cast<std::uint64_t>(__double_as_longlong(value));
#else
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
#endif
}

GENERATIVEQC_SCALED_HD inline double binary64_from_bits(std::uint64_t bits) {
#if defined(__CUDA_ARCH__)
  return __longlong_as_double(static_cast<long long>(bits));
#else
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

/** Carry a signed relative exponent in a finite, private double scratch word.
 * Normalization converts once per atom; reverse reads integer bits instead
 * of converting absolute exponents and the common scale for every pair. */
GENERATIVEQC_SCALED_HD inline double pack_relative_exponent(int exponent) {
  static_assert(std::numeric_limits<int>::max() == 2147483647 &&
                std::numeric_limits<int>::min() == (-2147483647 - 1),
                "packed exponents require a signed 32-bit int");
  return binary64_from_bits(UINT64_C(0x3fe0000000000000) |
      static_cast<std::uint32_t>(exponent));
}

GENERATIVEQC_SCALED_HD inline int unpack_relative_exponent(double value) {
  const auto encoded = static_cast<std::uint32_t>(binary64_bits(value));
  return encoded <= 0x7fffffffu ? static_cast<int>(encoded)
      : static_cast<int>(encoded - 0x80000000u) - 2147483647 - 1;
}

struct ScaledProduct {
  double mantissa{0.5};
  int exponent{1};
  size_t zeros{};

  GENERATIVEQC_SCALED_HD bool multiply(double factor) {
    if (!std::isfinite(factor) || factor < 0 || factor > 1) return false;
    if (factor == 0) {
      ++zeros;
      return true;
    }
    int factor_exponent = 0;
    const double factor_mantissa = std::frexp(factor, &factor_exponent);
    mantissa *= factor_mantissa;
    exponent += factor_exponent;
    if (mantissa < 0.5) {
      mantissa *= 2;
      --exponent;
    }
    return true;
  }

  GENERATIVEQC_SCALED_HD double scaled_value(int scale) const {
    return zeros ? 0 : std::ldexp(mantissa, exponent - scale);
  }

  GENERATIVEQC_SCALED_HD double excluded_value(double factor, int scale) const {
    if (zeros > 1 || (zeros == 1 && factor != 0)) return 0;
    if (zeros == 1) return std::ldexp(mantissa, exponent - scale);
    int factor_exponent = 0;
    const double factor_mantissa = std::frexp(factor, &factor_exponent);
    return std::ldexp(mantissa / factor_mantissa,
                      exponent - factor_exponent - scale);
  }

  /** A normalized, zero-free row has a nonpositive relative exponent. With
   * a normal factor and canonical mantissa, mantissa/factor is below 2^1022;
   * scaling downward cannot overflow. Only this bound permits removing a
   * zero-bar term: single-zero or malformed rows can still produce 0*Inf. */
  GENERATIVEQC_SCALED_HD bool finite_normal_excluded_bound(double factor) const {
    return !zeros && exponent <= 0 && mantissa >= 0.5 && mantissa < 1 &&
           factor >= std::numeric_limits<double>::min() && factor <= 1;
  }

  /** A normal factor gives a finite normal mantissa/factor quotient (< 2^1022).
   * Binary rescaling commutes exactly with that rounded division. Remove the
   * factor before scaling. Canonical normalized rows divide by the factor's
   * exact [1,2) significand, keeping correctly rounded division near unity;
   * carry its power of two into the final scale instead. Changing exponent
   * bits is exact for normal results; ldexp retains underflow/overflow rounding.
   * Unproven encodings retain the original division path. */
  GENERATIVEQC_SCALED_HD double normal_factor_excluded_value(double factor, int scale) const {
    if (!zeros && factor >= std::numeric_limits<double>::min()) {
      double divisor = factor;
      int division_exponent = exponent - scale;
      if (factor <= 1 && mantissa >= 0.5 && mantissa < 1 && exponent <= scale) {
        const std::uint64_t factor_bits = binary64_bits(factor);
        const int factor_exponent = static_cast<int>((factor_bits >> 52) & 0x7ff);
        divisor = binary64_from_bits((factor_bits & UINT64_C(0x000fffffffffffff)) |
                                     UINT64_C(0x3ff0000000000000));
        division_exponent += 1023 - factor_exponent;
      }
      const double quotient = mantissa / divisor;
      std::uint64_t quotient_bits = binary64_bits(quotient);
      const int scaled_exponent = static_cast<int>((quotient_bits >> 52) & 0x7ff) +
                                  division_exponent;
      if (scaled_exponent > 0 && scaled_exponent < 0x7ff) {
        quotient_bits = (quotient_bits & UINT64_C(0x800fffffffffffff)) |
                        (static_cast<std::uint64_t>(scaled_exponent) << 52);
        return binary64_from_bits(quotient_bits);
      }
      return std::ldexp(quotient, division_exponent);
    }
    return excluded_value(factor, scale);
  }
};

/** Renormalize only before the ordered product can leave the normal range.
 * A normal factor with floor exponent -cost is at least 2^-cost. Starting
 * from mantissa >= 0.5, a cost sum <= 1021 keeps every product >= 2^-1022.
 * Binary scaling therefore commutes exactly with each rounded multiply.
 * Smaller factors retain the per-factor fallback, including subnormals.
 * Call finalize before reading a live inherited product representation. */
struct BlockedScaledProduct : ScaledProduct {
  unsigned pending_cost{};

  GENERATIVEQC_SCALED_HD void finalize() {
    if (!pending_cost) return;
    if (zeros > 1) {
      pending_cost = 0;
      return;
    }
    int adjustment = 0;
    mantissa = std::frexp(mantissa, &adjustment);
    exponent += adjustment;
    pending_cost = 0;
  }

  GENERATIVEQC_SCALED_HD bool multiply(double factor) {
    if (!std::isfinite(factor) || factor < 0 || factor > 1) return false;
    if (factor == 0) {
      ++zeros;
      return true;
    }
    // Two exact zeros annihilate the product and every first excluded product.
    // Keep validating later factors and counting zeros, but their encoding is dead.
    if (zeros > 1) return true;
    if (factor == 1) return true;
    if (factor < 0x1p-1021) {
      finalize();
      return ScaledProduct::multiply(factor);
    }
    const auto encoded_exponent = static_cast<unsigned>((binary64_bits(factor) >> 52) & 0x7ffu);
    const unsigned cost = 1023u - encoded_exponent;
    if (pending_cost + cost > 1021u) finalize();
    mantissa *= factor;
    pending_cost += cost;
    return true;
  }
};

GENERATIVEQC_SCALED_HD inline ScaledProduct atom_product(Workspace work,
    size_t point, size_t atom) {
  return {work.field(4, point)[atom], static_cast<int>(work.field(7, point)[atom]),
          work.zero_counts(point)[atom]};
}

GENERATIVEQC_SCALED_HD inline void store_relative_exponent(Workspace work,
    size_t point, size_t atom, double scale) {
  work.field(7, point)[atom] = pack_relative_exponent(
      static_cast<int>(work.field(7, point)[atom] - scale));
}

GENERATIVEQC_SCALED_HD inline ScaledProduct normalized_atom_product(Workspace work,
    size_t point, size_t atom) {
  return {work.field(4, point)[atom], unpack_relative_exponent(work.field(7, point)[atom]),
          work.zero_counts(point)[atom]};
}

template <class Geometry, class Log, class Pair>
GENERATIVEQC_SCALED_HD bool scaled_pair_primal_phase(Workspace work, size_t point,
    size_t first, size_t second, Geometry geometry, Log logarithm, Pair pair) {
  bool valid = true;
  const auto separation = geometry.separation(first, second, valid);
  const auto state = point_pair<false>(work.field(0, point)[first] -
      work.field(0, point)[second], first, second, separation[0], geometry,
      logarithm, pair);
  const size_t index = center_pair_index(first, second);
  // No logarithms occupy these words. Retain the canonical coordinate partials
  // until reverse, rather than evaluating the same geometry graph again.
  work.pair(0, index, point) = state.ratio[0];
  work.pair(1, index, point) = state.ratio[1];
  work.pair(2, index, point) = state.factor[0];
  work.pair(3, index, point) = state.factor[1];
  return valid && std::isfinite(state.factor[0]);
}

template <bool PrimalValidated = false>
GENERATIVEQC_SCALED_HD inline bool scaled_atom_products_phase(Workspace work,
    size_t point, size_t atom) {
  if (!work.atoms || work.atoms > maximum_scaled_atoms) return false;
  BlockedScaledProduct product;
  for (size_t neighbor = 0; neighbor < work.atoms; ++neighbor) {
    if (neighbor == atom) continue;
    const bool upper = neighbor > atom;
    const size_t index = center_pair_index(std::max(atom, neighbor), std::min(atom, neighbor));
    const double factor = work.pair(2, index, point);
    if (!product.multiply(upper ? 1 - factor : factor)) return false;
    // Native first-gradient consumers run only after all primal factors pass.
    // Two zeros make every excluded product zero; their private count may
    // saturate at two. Unvalidated callers still inspect every factor.
    if constexpr (PrimalValidated)
      if (product.zeros > 1) break;
  }
  product.finalize();
  work.field(4, point)[atom] = product.mantissa;
  work.field(7, point)[atom] = product.exponent;
  work.zero_counts(point)[atom] = product.zeros;
  return true;
}

template <class Ratio>
GENERATIVEQC_SCALED_HD bool scaled_point_normalize_phase(Workspace work,
    size_t point, size_t owner, double seed, Ratio ratio) {
  if (!work.atoms || work.atoms > maximum_scaled_atoms || owner >= work.atoms ||
      !std::isfinite(seed)) return false;
  int scale = std::numeric_limits<int>::min();
  for (size_t atom = 0; atom < work.atoms; ++atom) {
    const auto product = atom_product(work, point, atom);
    if (!product.zeros) scale = std::max(scale, product.exponent);
  }
  if (scale == std::numeric_limits<int>::min()) return false;
  work.maximum[point] = scale;
  for (size_t atom = 0; atom < work.atoms; ++atom) {
    work.field(5, point)[atom] = atom_product(work, point, atom).scaled_value(scale);
    store_relative_exponent(work, point, atom, scale);
  }
  const auto objective = normalized_product_objective(work.atoms, owner,
      work.field(5, point), ratio);
  for (size_t atom = 0; atom < work.atoms; ++atom)
    work.field(6, point)[atom] = normalized_product_bar(atom, owner, seed, objective);
  return true;
}

template <class Geometry>
GENERATIVEQC_SCALED_HD bool scaled_pair_pullback_coefficients(Workspace work,
    size_t point, size_t first, size_t second, Geometry geometry,
    PairPullbackCoefficients& result) {
  result = {};
  const size_t index = center_pair_index(first, second);
  const double factor = work.pair(2, index, point);
  const double slope = work.pair(3, index, point);
  if (!std::isfinite(factor) || factor < 0 || factor > 1 || !std::isfinite(slope))
    return false;
  if (slope == 0) return true;
  // A one-zero row has a live excluded derivative only at its zero factor.
  // Test the current factor before geometry, but never discard that live slot.
  const size_t first_zeros = work.zero_counts(point)[first];
  const size_t second_zeros = work.zero_counts(point)[second];
  const bool first_inactive = first_zeros > 1 || (first_zeros == 1 && factor != 0);
  const bool second_inactive = second_zeros > 1 || (second_zeros == 1 && factor != 1);
  if (first_inactive && second_inactive)
    return std::isfinite(work.field(6, point)[first]) &&
           std::isfinite(work.field(6, point)[second]);
  bool valid = true;
  if (slope != 0) {
    result.separation = geometry.separation(first, second, valid);
    double bar_mu = 0;
    for (size_t side = 0; side < 2; ++side) {
      const size_t atom = side ? second : first;
      const double value = side ? 1 - factor : factor;
      const auto product = normalized_atom_product(work, point, atom);
      const double bar = work.field(6, point)[atom];
      if (bar == 0 && product.finite_normal_excluded_bound(value)) continue;
      const double derivative = product.normal_factor_excluded_value(value, 0);
      bar_mu += (side ? -1 : 1) * bar * derivative * slope;
    }
    // Geometry and distances are fixed across the seven phases. Read the
    // primal-owned partials before reverse overwrites the same pair panel.
    result.values = {bar_mu * work.pair(0, index, point),
                     bar_mu * work.pair(1, index, point)};
  }
  for (double value : result.values) valid = valid && std::isfinite(value);
  return valid;
}

template <class Geometry, class Log>
GENERATIVEQC_SCALED_HD bool scaled_pair_reverse_phase(Workspace work, size_t point,
    size_t first, size_t second, Geometry geometry, Log) {
  PairPullbackCoefficients coefficients;
  bool valid = scaled_pair_pullback_coefficients(work, point, first, second,
                                                geometry, coefficients);
  const size_t index = center_pair_index(first, second);
  for (size_t word = 0; word < 4; ++word) {
    const double value = pair_pullback_component(coefficients.values[0],
        coefficients.values[1], word, coefficients.separation[word]);
    work.pair(word, index, point) = value;
    valid = valid && std::isfinite(value);
  }
  return valid;
}
}
#undef GENERATIVEQC_SCALED_HD
"""


def emit_binary_scaled_becke() -> str:
    """Emit an isolated candidate after the canonical adjoint/phased helpers.

    The 128-atom bound keeps every exponent exactly representable by both int
    and the borrowed double panel, even for positive FP64 subnormal factors.
    Ordered blocks amortize decomposition without reassociating multiplication;
    tiny factors flush the block and use per-factor decomposition to avoid
    intermediate underflow. Live encodings match the per-factor baseline;
    dead rows retain exact zero products and excluded derivatives instead.
    Normalization freezes a shared binary exponent and uses canonical ratio AD;
    exponent derivatives cancel in the normalized objective. Reverse removes a
    factor before scaling, rather than dividing an already-underflowed product.
    Saturated switch slopes remain an exact-zero early exit before that work.
    Native first-gradient traversal can stop a dead row only after the complete
    primal factor domain has passed validation. Its zero class saturates at two;
    unvalidated traversal retains complete validation and exact zero counts.
    Freed log words retain canonical geometry partials until reverse consumes
    them, without changing scratch size or floating-point evaluation order.
    No production selection, numerical gate or scratch reservation is changed.
    """
    return _SCALED_SOURCE
