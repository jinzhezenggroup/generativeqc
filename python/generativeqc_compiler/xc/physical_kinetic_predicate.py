"""Exact physical-input lowering of the canonical kinetic boundary predicate.

Rounded spin dot products need not retain a realizable rank-one equality. Only
the predicate uses the original Cartesian inputs; branch values and already
differentiated work-coordinate coefficients remain unchanged.
"""

from __future__ import annotations

from generativeqc_compiler.integral.expr import Expr, Graph

PREDICATE_VARIABLE = "physical_kinetic_side"


def bind_physical_kinetic_predicate(
    graph: Graph, roots: tuple[Expr, ...]
) -> tuple[tuple[Expr, ...], bool]:
    """Bind only canonical total-sigma/kinetic comparisons in polarized roots.

    The original normal-product guard stays intact. Replacing both energy and
    derivative predicates together retains the inclusive inward tie convention
    without differentiating the work-input floors or changing branch payloads.
    """
    names = {node.payload for node in graph.nodes if node.operation == "variable"}
    required = {"rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb", "tau_a", "tau_b"}
    if not required <= names:
        return roots, False
    sigma_terms = (
        graph.variable("sigma_aa"),
        2 * graph.variable("sigma_ab"),
        graph.variable("sigma_bb"),
    )
    sigma_ids = {
        (sigma_terms[0] + sigma_terms[1] + sigma_terms[2]).identifier,
        graph.add_many(sigma_terms).identifier,
    }
    density = graph.variable("rho_a") + graph.variable("rho_b")
    tau = graph.variable("tau_a") + graph.variable("tau_b")
    cap_ids = {
        (8 * (density * tau)).identifier,
        graph.multiply_many((graph.constant(8), density, tau)).identifier,
    }
    replacements = {}
    for identifier in graph.topological_order(roots):
        node = graph.nodes[identifier]
        if (
            node.operation == "select_le"
            and node.arguments[0] in sigma_ids
            and node.arguments[1] in cap_ids
        ):
            replacements[Expr(graph, identifier)] = graph.select_le(
                graph.variable(PREDICATE_VARIABLE),
                0,
                Expr(graph, node.arguments[2]),
                Expr(graph, node.arguments[3]),
            )
    return graph.replace_subexpressions(roots, replacements), bool(replacements)


def emit_exact_kinetic_comparison() -> str:
    """Emit a bounded, backend-neutral exact binary64 quadratic comparison.

    Outward-rounded intervals certify separated sides cheaply. Ambiguous
    intervals fall back to a signed 4352-bit accumulator, which compares
    8*(rho_a+rho_b)*(tau_a+tau_b)
    against sum_axis((gradient_a+gradient_b)**2), without rounding any sum or
    product. frexp supplies exact integer significands; four 32-bit product
    limbs avoid host-only 128-bit arithmetic. The accumulator covers every
    finite binary64 input, including subnormals, with room for all 13 terms.
    There is no numerical tolerance, oracle, dynamic storage or extended type.
    """
    return """[&]() {
    static_assert(sizeof(unsigned long long) == 8);
    static_assert(sizeof(unsigned) == 4);
    const double infinity = HUGE_VAL;
    const double density_lower = fmax(0.0, nextafter(rho[0] + rho[1], -infinity));
    const double density_upper = nextafter(rho[0] + rho[1], infinity);
    const double tau_lower = fmax(0.0, nextafter(tau[0] + tau[1], -infinity));
    const double tau_upper = nextafter(tau[0] + tau[1], infinity);
    const double cap_lower = fmax(0.0, nextafter(8.0 * fmax(0.0,
        nextafter(density_lower * tau_lower, -infinity)), -infinity));
    const double cap_upper = nextafter(8.0 *
        nextafter(density_upper * tau_upper, infinity), infinity);
    double sigma_lower = 0.0;
    double sigma_upper = 0.0;
    for (unsigned axis = 0; axis < 3; ++axis) {
      const double sum = gradient[0][axis] + gradient[1][axis];
      const double lower = nextafter(sum, -infinity);
      const double upper = nextafter(sum, infinity);
      const double nearest = lower <= 0.0 && upper >= 0.0
          ? 0.0 : fmin(fabs(lower), fabs(upper));
      const double farthest = fmax(fabs(lower), fabs(upper));
      const double squared_lower = fmax(0.0, nextafter(nearest * nearest, -infinity));
      const double squared_upper = nextafter(farthest * farthest, infinity);
      sigma_lower = fmax(0.0, nextafter(sigma_lower + squared_lower, -infinity));
      sigma_upper = nextafter(sigma_upper + squared_upper, infinity);
    }
    if (sigma_upper <= cap_lower) return true;
    if (sigma_lower > cap_upper) return false;
    unsigned words[136]{};
    const auto add_product = [&](double left, double right, unsigned shift,
                                 bool subtract) {
      if (left == 0.0 || right == 0.0) return;
      subtract = subtract != ((left < 0.0) != (right < 0.0));
      int left_exponent = 0;
      int right_exponent = 0;
      const auto left_mantissa = static_cast<unsigned long long>(
          frexp(fabs(left), &left_exponent) * 0x1p53);
      const auto right_mantissa = static_cast<unsigned long long>(
          frexp(fabs(right), &right_exponent) * 0x1p53);
      const unsigned long long mask = 0xffffffffULL;
      const auto low = (left_mantissa & mask) * (right_mantissa & mask);
      const auto cross_left = (left_mantissa & mask) * (right_mantissa >> 32);
      const auto cross_right = (left_mantissa >> 32) * (right_mantissa & mask);
      const auto middle = (low >> 32) + (cross_left & mask) + (cross_right & mask);
      const auto high = (left_mantissa >> 32) * (right_mantissa >> 32) +
          (cross_left >> 32) + (cross_right >> 32) + (middle >> 32);
      const unsigned limbs[4]{static_cast<unsigned>(low),
          static_cast<unsigned>(middle), static_cast<unsigned>(high),
          static_cast<unsigned>(high >> 32)};
      const unsigned offset = static_cast<unsigned>(
          left_exponent + right_exponent - 106 + 2252) + shift;
      for (unsigned limb = 0; limb < 4; ++limb) {
        unsigned index = offset / 32 + limb;
        unsigned long long carry =
            static_cast<unsigned long long>(limbs[limb]) << (offset % 32);
        while (carry != 0 && index < 136) {
          const auto old = static_cast<unsigned long long>(words[index]);
          const auto part = carry & mask;
          if (subtract) {
            words[index] = static_cast<unsigned>(old - part);
            carry = (carry >> 32) + (old < part);
          } else {
            const auto sum = old + part;
            words[index] = static_cast<unsigned>(sum);
            carry = (carry >> 32) + (sum >> 32);
          }
          ++index;
        }
      }
    };
    for (unsigned spin = 0; spin < 2; ++spin)
      for (unsigned other = 0; other < 2; ++other)
        add_product(rho[spin], tau[other], 3, false);
    for (unsigned axis = 0; axis < 3; ++axis) {
      add_product(gradient[0][axis], gradient[0][axis], 0, true);
      add_product(gradient[0][axis], gradient[1][axis], 1, true);
      add_product(gradient[1][axis], gradient[1][axis], 0, true);
    }
    return (words[135] & 0x80000000U) == 0;
  }()"""


def emit_physical_kinetic_side() -> tuple[str, ...]:
    """Keep transformed work inputs on their original canonical comparison.

    Exact raw-feature classification is used only if no floor, FHC clamp or
    cross-spin clamp changed the work inputs. This intentionally does not grant
    a new boundary convention to empty/floored-spin work coordinates.
    """
    return (
        "  const double physical_kinetic_side = [&]() {",
        "    if (work_rho_a != rho_a || work_rho_b != rho_b ||",
        "        work_tau_a != tau_a || work_tau_b != tau_b ||",
        "        work_sigma_aa != sigma_aa || work_sigma_ab != sigma_ab ||",
        "        work_sigma_bb != sigma_bb)",
        "      return work_sigma_aa + 2.0 * work_sigma_ab + work_sigma_bb <=",
        "          8.0 * ((work_rho_a + work_rho_b) * (work_tau_a + work_tau_b))",
        "          ? 0.0 : 1.0;",
        f"    const bool inside = {emit_exact_kinetic_comparison()};",
        "    return inside ? 0.0 : 1.0;",
        "  }();",
    )
