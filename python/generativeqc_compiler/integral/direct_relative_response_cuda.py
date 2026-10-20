"""Compiler-owned fused relative-pair contraction for exact Direct forces."""


def emit_direct_relative_response_support() -> str:
    """Emit bounded pair preparation and a fused nuclear-response contraction.

    Runtime consumers own screening, weights, unique-atom projection and scatter.
    This reusable schedule lends scalar spatial roots and relative Hermite
    responses to every independent atom without recomputing their contractions.
    """
    return _RELATIVE_SUPPORT


_RELATIVE_SUPPORT = r"""
struct DirectRelativePairDerivativeRecurrence {
  using Pair = ShellPairHermiteCoefficients<Dual, 2, 2>;
  Pair bra[3], ket[3];
  CoulombAuxiliary<double, 9> coulomb;
  Vec3<double> first_product;
  double p, q, coefficients[4], product_scales[4], response_scales[3][3];
};
static_assert(sizeof(DirectRelativePairDerivativeRecurrence) <= (12U << 10));
/** Seed one relative pair displacement, independently of atom identity.
 * Repeated atom identities are projected only after the shared contraction.
 * Gaussian decay stays inside the existing Hermite recurrence; no Gaussian
 * factor is divided out, including at underflow and coincident centers. */
__device__ inline void prepare_direct_relative_pair(
    const PrimitivePairData& cached, double alpha, double beta,
    const Vec3<double>& first, const Vec3<double>& second,
    DirectRelativePairDerivativeRecurrence::Pair* coefficients,
    unsigned first_angular, unsigned second_angular, unsigned axis) {
  const Dual first_axis{vec_axis(first, axis), 1.0};
  const Dual second_axis{vec_axis(second, axis), 0.0};
  const Dual product_axis{vec_axis(cached.product_center, axis), alpha / (alpha + beta)};
  const Dual difference = first_axis - second_axis;
  fill_shell_pair_hermite_geometry<2, 2>(first_angular, second_angular, product_axis,
      first_axis, second_axis, cached.exponent_sum,
      qexp(-cached.reduced_exponent * difference * difference), coefficients[axis]);
}

/** Contract a relative-response basis once, rather than once per atom.
 * Each axis convolution shares its primal coefficient with both pair responses.
 * Only nine response sums survive the contraction: bra, ket and P-Q for xyz.
 * Nuclear Jacobians are constant during this sum and are applied by the caller.
 * The packed axis rows require at most eleven triples, never an AO tensor. */
__device__ inline void contract_direct_relative_response_basis(
    const Angular& first, const Angular& second, const Angular& third, const Angular& fourth,
    const DirectRelativePairDerivativeRecurrence& recurrence,
    double (&responses)[3][3]) {
  struct Weight { double value, bra, ket; };
  Weight weights[11];
  unsigned degrees[3], offsets[3], offset = 0U;
  for (unsigned axis = 0; axis < 3U; ++axis) {
    const unsigned first_degree = angular_axis(first, axis);
    const unsigned second_degree = angular_axis(second, axis);
    const unsigned third_degree = angular_axis(third, axis);
    const unsigned fourth_degree = angular_axis(fourth, axis);
    degrees[axis] = first_degree + second_degree + third_degree + fourth_degree;
    offsets[axis] = offset;
    offset += degrees[axis] + 1U;
    auto* row = weights + offsets[axis];
    for (unsigned state = 0; state <= degrees[axis]; ++state) row[state] = {};
    for (unsigned bra = 0; bra <= first_degree + second_degree; ++bra) {
      const auto left = recurrence.bra[axis].at(first_degree, second_degree, bra);
      for (unsigned ket = 0; ket <= third_degree + fourth_degree; ++ket) {
        const auto right = recurrence.ket[axis].at(third_degree, fourth_degree, ket);
        const double sign = (ket & 1U) ? -1.0 : 1.0;
        auto& item = row[bra + ket];
        item.value += sign * left.value * right.value;
        item.bra += sign * left.derivative * right.value;
        item.ket += sign * left.value * right.derivative;
      }
    }
  }
  for (unsigned axis_x = 0; axis_x <= degrees[0]; ++axis_x)
    for (unsigned axis_y = 0; axis_y <= degrees[1]; ++axis_y)
      for (unsigned axis_z = 0; axis_z <= degrees[2]; ++axis_z) {
        const Weight items[3] = {weights[axis_x], weights[offsets[1] + axis_y],
                                weights[offsets[2] + axis_z]};
        const double others[3] = {items[1].value * items[2].value,
            items[0].value * items[2].value, items[0].value * items[1].value};
        const double coefficient = others[2] * items[2].value;
        const double coulomb = recurrence.coulomb.at(0, axis_x, axis_y, axis_z);
        for (unsigned axis = 0; axis < 3U; ++axis) {
          responses[0][axis] += items[axis].bra * others[axis] * coulomb;
          responses[1][axis] += items[axis].ket * others[axis] * coulomb;
          responses[2][axis] += coefficient * recurrence.coulomb.at(0,
              axis_x + (axis == 0U), axis_y + (axis == 1U), axis_z + (axis == 2U));
        }
      }
}

"""
