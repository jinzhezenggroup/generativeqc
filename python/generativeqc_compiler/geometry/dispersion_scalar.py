"""Scalar dispersion energies and AD for the bounded native pair schedulers.

The compiler owns differentiation. Native code retains table lookup, cutoff
admission, underflow fallbacks, ragged traversal and accumulation. This lowering
does not promote a new scheduler or change the public method domain.
"""

import math
from fractions import Fraction
from functools import cache

from generativeqc_compiler.integral.expr import Expr, Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter
from generativeqc_compiler.integral.scalar_python import polynomial_terms


class _ProjectionEmitter(ScalarCEmitter):
    """Retain multiply-then-divide scaling in cached norm pullbacks."""

    def _operation_code(self, identifier: int) -> str:
        node = self.graph.nodes[identifier]
        if node.operation == "multiply" and len(node.arguments) == 2:
            for index, operand in enumerate(node.arguments):
                inverse = self.graph.nodes[operand]
                if inverse.operation == "reciprocal":
                    return f"{self._reference(node.arguments[1 - index])} / {self._reference(inverse.arguments[0])}"
        return super()._operation_code(identifier)


class _Log2Emitter(ScalarCEmitter):
    """Preserve independent base-two scaling of tiny energies and responses."""

    def _operation_code(self, identifier: int) -> str:
        node = self.graph.nodes[identifier]
        if node.operation == "exp":
            argument = self.graph.nodes[node.arguments[0]]
            if argument.operation == "multiply" and len(argument.arguments) == 2:
                for index, operand in enumerate(argument.arguments):
                    coefficient = self.graph.nodes[operand]
                    if (
                        coefficient.operation == "constant"
                        and coefficient.payload == math.log(2)
                    ):
                        return f"exp2({self._reference(argument.arguments[1 - index])})"
        return super()._operation_code(identifier)


def _atm_power(graph: Graph) -> Expr:
    """Parameterize positive dynamic powers without quotient-rule work."""
    logarithmic_radius = graph.stable_unary("log", graph.variable("radius_product"))
    logarithmic_lengths = graph.sum(
        graph.stable_unary("log", graph.variable(name))
        for name in ("first_second_r2", "first_third_r2", "second_third_r2")
    )
    return graph.exponential(
        graph.variable("exponent")
        / 3
        * (logarithmic_radius - 0.5 * logarithmic_lengths)
    )


@cache
def dispersion_scalar_program(name: str) -> tuple:
    """Return graph, input names and primal/response roots for a scalar contract.

    D3 coordination and BJ damping responses are Cartesian scatter coefficients
    (radial derivative divided by distance). Other physical responses are plain
    scalar derivatives; normalization and product responses are supplied JVPs.
    """
    graph = Graph()

    def variables(*names: str) -> tuple:
        return tuple(graph.variable(value) for value in names)

    if name in ("radial_projection", "radial_scatter"):
        names = ("distance", "radial_derivative") + (
            ("component",) if name == "radial_projection" else ()
        )
        component, other, third = variables("component", "other", "third")
        squared = component * component + other * other + third * third
        norm = squared.pow(Fraction(1, 2))
        response = graph.differentiate(norm, component)
        (response,) = graph.replace_subexpressions(
            (response,),
            {squared.pow(Fraction(-1, 2)): graph.variable("inverse_distance")},
        )
        graph, (response,) = graph.canonicalize_associative(
            (response,), factor_common=False
        )
        terms = polynomial_terms(graph, response)
        if terms != ((Fraction(1), ("component", "inverse_distance")),):
            raise ValueError(
                "norm AD no longer admits the cached radial scatter schedule"
            )
        value = graph.variable("radial_derivative")
        if name == "radial_projection":
            value = value * graph.variable("component")
        roots = (value / graph.variable("distance"),)
    elif name == "gaussian":
        names = ("coordination", "reference", "factor")
        coordination, reference, factor = variables(*names)
        delta = coordination - reference
        value = graph.exponential(-factor * delta * delta)
        roots = (value, graph.differentiate(value, coordination))
    elif name == "normalize":
        names = ("value", "norm", "dvalue", "dnorm")
        value, norm, dvalue, dnorm = variables(*names)
        normalized = value / norm
        roots = (
            normalized,
            graph.differentiate(
                normalized,
                graph.variable("direction"),
                {"value": dvalue, "norm": dnorm},
            ),
        )
    elif name == "smooth_switch":
        names = ("distance", "cutoff", "width")
        distance, cutoff, width = variables(*names)
        scaled = (cutoff - distance) / width
        value = scaled.pow(3) * (10 + scaled * (-15 + 6 * scaled))
        roots = (value, graph.differentiate(value, distance))
    elif name == "d3_coordination":
        names = ("distance", "radius")
        distance, radius = variables(*names)
        argument = 16 * (radius / distance - 1)
        positive_exp = graph.exponential(argument)
        negative_exp = graph.exponential(-argument)
        value = graph.select_le(
            argument, 0, positive_exp / (1 + positive_exp), 1 / (1 + negative_exp)
        )
        roots = (value, graph.differentiate(value, distance) / distance)
    elif name == "bj_damping":
        names = ("distance_squared", "rr", "radius", "s6", "s8")
        distance_squared, rr, radius, s6, s8 = variables(*names)
        value = s6 / (distance_squared.pow(3) + radius.pow(6)) + s8 * rr / (
            distance_squared.pow(4) + radius.pow(8)
        )
        roots = (value, 2 * graph.differentiate(value, distance_squared))
    elif name == "d4_charge":
        names = ("a", "c", "qref", "qmod")
        steepness, hardness, qref, qmod = variables(*names)
        inner = graph.exponential(hardness * (1 - qref / qmod))
        value = graph.exponential(steepness * (1 - inner))
        derivative = graph.differentiate(value, qmod)
        derivative = graph.select_le(
            inner, 0, 0, graph.select_le(value, 0, 0, derivative)
        )
        roots = (value, derivative)
    elif name == "d4_coordination":
        names = ("distance", "radius", "electronegativity")
        distance, radius, electronegativity = variables(*names)
        argument = 7.5 * (radius - distance) / radius
        value = (
            0.5 * electronegativity * (1 + graph.transcendental_unary("erf", argument))
        )
        roots = (value, graph.differentiate(value, distance))
    elif name == "product_response":
        names = ("first", "second", "dfirst", "dsecond")
        first, second, dfirst, dsecond = variables(*names)
        value = first * second
        roots = (
            value,
            graph.differentiate(
                value, graph.variable("direction"), {"first": dfirst, "second": dsecond}
            ),
        )
    elif name in ("atm", "atm_absolute"):
        names = (
            "first_second_r2",
            "first_third_r2",
            "second_third_r2",
            "radius_product",
            "first_second_c6",
            "first_third_c6",
            "second_third_c6",
            "s9",
            "exponent",
        )
        first, second, third, radius, c6_first, c6_second, c6_third, s9, exponent = (
            variables(*names)
        )
        product = first * second * third
        product_root = product.pow(0.5)
        product_cubed = product * product_root
        ratio_power = _atm_power(graph)
        damping = 1 / (1 + 6 * ratio_power)
        angle = (
            0.375
            * (first + third - second)
            * (first - third + second)
            * (-first + third + second)
            / (product_cubed * product)
            + 1 / product_cubed
        )
        c6_product = c6_first * c6_second * c6_third
        if name == "atm_absolute":
            c6_product = graph.select_le(c6_product, 0, -c6_product, c6_product)
        value = s9 * c6_product.pow(0.5) * angle * damping
        roots = (
            value,
            *(2 * graph.differentiate(value, edge) for edge in (first, second, third)),
            *(
                graph.differentiate(value, coefficient)
                for coefficient in (c6_first, c6_second, c6_third)
            ),
        )
    elif name == "atm_switch":
        names = (
            "energy",
            "first_radial",
            "second_radial",
            "third_radial",
            "first_weight",
            "second_weight",
            "third_weight",
            "first_slope",
            "second_slope",
            "third_slope",
            "first_distance",
            "second_distance",
            "third_distance",
        )
        energy, *rest = variables(*names)
        radial = rest[:3]
        weights, slopes, distances = rest[3:6], rest[6:9], rest[9:12]
        value = energy * weights[0] * weights[1] * weights[2]
        roots = (
            value,
            *(
                graph.differentiate(
                    value,
                    graph.variable("edge"),
                    {
                        "energy": radial[index],
                        names[4 + index]: slopes[index] / distances[index],
                    },
                )
                for index in range(3)
            ),
        )
    elif name == "zero_regular":
        names = ("distance", "scaled_radius", "exponent", "power", "ratio_power")
        distance, radius, exponent, power, _ = variables(*names)
        ratio_power = graph.exponential(
            exponent
            * (graph.stable_unary("log", radius) - graph.stable_unary("log", distance))
        )
        inverse = graph.exponential(-power * graph.stable_unary("log", distance))
        value = inverse / (1 + 6 * ratio_power)
        roots = (value, graph.differentiate(value, distance) / distance)
    elif name == "zero_overflow":
        names = ("log_distance", "log_radius", "exponent", "power")
        distance, radius, exponent, power = variables(*names)
        logarithm = -power * distance - math.log2(6) - exponent * (radius - distance)
        logarithmic_slope = graph.differentiate(logarithm, distance)
        magnitude = graph.select_le(
            logarithmic_slope, 0, -logarithmic_slope, logarithmic_slope
        )
        value = graph.exponential(math.log(2) * logarithm)
        radial_magnitude = graph.exponential(
            math.log(2)
            * (
                logarithm
                - 2 * distance
                + graph.stable_unary("log", magnitude) / math.log(2)
            )
        )
        radial = graph.select_le(
            magnitude,
            0,
            0,
            graph.select_le(logarithmic_slope, 0, -radial_magnitude, radial_magnitude),
        )
        roots = (value, radial)
    elif name == "eeq_rhs":
        names = ("raw_coordination", "chi", "kcnchi")
        raw, chi, kcnchi = variables(*names)
        coordination = math.log1p(math.exp(8)) - graph.stable_unary(
            "log1p", graph.exponential(8 - raw)
        )
        value = -chi + kcnchi * coordination / (coordination + 1e-14).pow(0.5)
        roots = (value, graph.differentiate(value, raw))
    elif name == "eeq_coulomb":
        names = ("distance", "gamma")
        distance, gamma = variables(*names)
        value = graph.transcendental_unary("erf", gamma * distance) / distance
        roots = (value, graph.differentiate(value, distance) / distance)
    else:
        raise ValueError(f"unknown scalar dispersion program: {name}")
    return graph, names, roots


SCALAR_PROGRAMS = (
    "radial_projection",
    "radial_scatter",
    "gaussian",
    "normalize",
    "smooth_switch",
    "d3_coordination",
    "bj_damping",
    "d4_charge",
    "d4_coordination",
    "product_response",
    "atm",
    "atm_absolute",
    "atm_switch",
    "zero_regular",
    "zero_overflow",
    "eeq_rhs",
    "eeq_coulomb",
)


def emit_dispersion_scalar_header() -> str:
    """Emit shared host/device functions with derivative-free value specialization."""
    lines = [
        "// Generated by tools/generate_dispersion_scalar.py; do not edit.",
        "#pragma once",
        "#include <cfloat>",
        "#include <cmath>",
        "#if defined(__CUDACC__)",
        "#define GENERATIVEQC_DISPERSION_HD __host__ __device__",
        "#define GENERATIVEQC_DISPERSION_HOT_INLINE inline",
        "#else",
        "#define GENERATIVEQC_DISPERSION_HD",
        "#if defined(__GNUC__)",
        "#define GENERATIVEQC_DISPERSION_HOT_INLINE inline __attribute__((always_inline))",
        "#else",
        "#define GENERATIVEQC_DISPERSION_HOT_INLINE inline",
        "#endif",
        "#endif",
        "namespace generativeqc::generated::dispersion {",
    ]
    for name in SCALAR_PROGRAMS:
        graph, names, roots = dispersion_scalar_program(name)
        inlining = (
            "GENERATIVEQC_DISPERSION_HOT_INLINE"
            if name in ("atm", "atm_absolute", "d3_coordination")
            else "inline"
        )
        lines.extend(
            [
                "template <bool NeedDerivative = true>",
                f"GENERATIVEQC_DISPERSION_HD {inlining} void {name}("
                + ", ".join(f"double {variable}" for variable in names)
                + ", double* output) {",
            ]
        )
        for derivative in (False, True):
            selected = roots if derivative else roots[:1]
            emission_graph = graph
            prelude = []
            if name in ("atm", "atm_absolute"):
                dynamic_power = _atm_power(graph)
                prelude.append(
                    "    const double atm_power = pow(radius_product / sqrt("
                    "first_second_r2 * first_third_r2 * second_third_r2), exponent / 3.0);"
                )
                selected = graph.replace_subexpressions(
                    selected, {dynamic_power: graph.variable("atm_power")}
                )
                # Positive squared lengths/C6 products admit reciprocal sqrt;
                # avoid generic negative-half pow calls in the triple hot path.
                selected = graph.replace_subexpressions(
                    selected,
                    {
                        Expr(graph, identifier): 1
                        / Expr(graph, graph.nodes[identifier].arguments[0]).pow(0.5)
                        for identifier in graph.topological_order(selected)
                        if graph.nodes[identifier].operation == "power"
                        and graph.nodes[identifier].payload == -0.5
                    },
                )
            elif name == "zero_regular":
                distance, radius, exponent, power = (
                    graph.variable(field) for field in names[:4]
                )
                ratio_power = graph.exponential(
                    exponent
                    * (
                        graph.stable_unary("log", radius)
                        - graph.stable_unary("log", distance)
                    )
                )
                inverse = graph.exponential(
                    -power * graph.stable_unary("log", distance)
                )
                prelude.extend(
                    [
                        "    double inverse;",
                        "    if (power == 6.0 || power == 8.0) {",
                        "      const double reciprocal = 1.0 / distance;",
                        "      const double square = reciprocal * reciprocal;",
                        "      const double fourth = square * square;",
                        "      inverse = power == 6.0 ? fourth * square : fourth * fourth;",
                        "    } else {",
                        "      inverse = 1.0;",
                        "      for (int order = 0; order < static_cast<int>(power); ++order)",
                        "        inverse /= distance;",
                        "    }",
                    ]
                )
                selected = graph.replace_subexpressions(
                    selected,
                    {
                        ratio_power: graph.variable("ratio_power"),
                        inverse: graph.variable("inverse"),
                    },
                )
            if name not in ("zero_overflow", "radial_projection", "radial_scatter"):
                selected = graph.replace_subexpressions(
                    selected,
                    {
                        Expr(graph, identifier): (
                            1 / Expr(graph, graph.nodes[identifier].arguments[0])
                        ).pow(-float(graph.nodes[identifier].payload))
                        for identifier in graph.topological_order(selected)
                        if graph.nodes[identifier].operation == "power"
                        and graph.nodes[identifier].payload
                        in (-1, -2, -3, -4, -5, -6, -7, -8)
                    },
                )
                selected = graph.replace_subexpressions(
                    selected,
                    {
                        Expr(graph, identifier): 1
                        / Expr(graph, graph.nodes[identifier].arguments[0]).pow(0.5)
                        for identifier in graph.topological_order(selected)
                        if graph.nodes[identifier].operation == "power"
                        and graph.nodes[identifier].payload == -0.5
                    },
                )
                if name == "bj_damping":
                    # Preserve the final addition-chain schedule: flattening
                    # after power lowering recreates long serial products.
                    emission_graph, selected = graph.canonicalize_associative(
                        selected, factor_common=True
                    )
                    emission_graph, selected = (
                        emission_graph.lower_small_integer_powers(
                            selected, maximum_absolute_power=8
                        )
                    )
                else:
                    emission_graph, selected = graph.lower_small_integer_powers(
                        selected, maximum_absolute_power=8
                    )
                    emission_graph, selected = emission_graph.canonicalize_associative(
                        selected, factor_common=True
                    )
            emitter_type = (
                _ProjectionEmitter
                if name in ("radial_projection", "radial_scatter")
                else _Log2Emitter
                if name == "zero_overflow"
                else ScalarCEmitter
            )
            emitter = emitter_type(
                emission_graph,
                dict(zip(names, names, strict=True)),
                coalesce_branches=name == "d3_coordination",
            )
            emitter.emit(selected)
            lines.extend(
                [
                    "  if constexpr (NeedDerivative) {"
                    if derivative
                    else "  if constexpr (!NeedDerivative) {",
                    *prelude,
                    *emitter.lines,
                    *(
                        f"    output[{index}] = {emitter.reference(root)};"
                        for index, root in enumerate(selected)
                    ),
                    "  }",
                ]
            )
        lines.append("}")
    lines.extend(
        [
            "GENERATIVEQC_DISPERSION_HD inline void zero_damping(double distance, double scaled_radius,",
            "    double exponent, double power, double* output) {",
            "  const double ratio_power = pow(scaled_radius / distance, exponent);",
            "  if (ratio_power <= DBL_MAX / 6.0) {",
            "    zero_regular(distance, scaled_radius, exponent, power, ratio_power, output);",
            "  } else {",
            "    zero_overflow(log2(distance), log2(scaled_radius), exponent, power, output);",
            "  }",
            "}",
            "}",
            "#undef GENERATIVEQC_DISPERSION_HD",
            "#undef GENERATIVEQC_DISPERSION_HOT_INLINE",
            "",
        ]
    )
    return "\n".join(lines)
