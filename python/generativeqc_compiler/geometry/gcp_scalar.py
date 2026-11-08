"""Slater-overlap and gCP scalar primals with compiler-owned radial AD.

Shell products retain the pinned simple-dftd3 normalization and auxiliary
recurrences. Differentiating the actual finite B series also keeps its small
argument limit smooth instead of differentiating a separately truncated table.
"""

import math
from functools import cache

from generativeqc_compiler.integral.expr import Expr, Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

SHELL_PRODUCTS = {
    1: (3, (1, -1), (2, 0), (0, 2)),
    2: (4, (1, -1, 1, -1), (3, 0, 2, 1), (0, 3, 1, 2)),
    3: (5, (1, -1, 2, -2), (4, 0, 3, 1), (0, 4, 1, 3)),
    4: (5, (1, 1, -2), (4, 0, 2), (0, 4, 2)),
    6: (6, (1, 1, -2, -2, 1, 1), (5, 4, 3, 2, 1, 0), (0, 1, 2, 3, 4, 5)),
    9: (7, (1, -3, 3, -1), (6, 4, 2, 0), (0, 2, 4, 6)),
}


@cache
def overlap_program(shell_product: int, series: bool) -> tuple:
    """Return an overlap primal and its radial derivative for one shell class.

    Mixed shells use increasing principal-shell order for the exponents.
    ``series`` is the established abs(za-zb)<0.1 auxiliary admission policy.
    """
    graph = Graph()
    names = ("distance", "first_exponent", "second_exponent")
    distance, first, second = (graph.variable(name) for name in names)
    order, weights, first_orders, second_orders = SHELL_PRODUCTS[shell_product]
    if shell_product == 1:
        normalization = 0.25 * (first * second).pow(3).pow(0.5)
    elif shell_product == 2:
        normalization = (
            math.sqrt(1 / 3) * (first.pow(3) * second.pow(5)).pow(0.5) * 0.125
        )
    elif shell_product == 3:
        normalization = (
            (first.pow(3) * second.pow(7) / 7.5).pow(0.5) * 0.0625 / math.sqrt(3)
        )
    elif shell_product == 4:
        normalization = (first * second).pow(5).pow(0.5) * 0.0625 / 3
    elif shell_product == 6:
        normalization = (first.pow(5) * second.pow(7) / 7.5).pow(0.5) * 0.03125 / 3
    else:
        normalization = (first * second).pow(7).pow(0.5) / 1440
    first_argument = 0.5 * (first + second) * distance
    second_argument = 0.5 * (second - first) * distance
    exponential = graph.exponential(-first_argument)
    first_auxiliary = [exponential / first_argument]
    for auxiliary_order in range(1, max(first_orders) + 1):
        first_auxiliary.append(
            (auxiliary_order * first_auxiliary[-1] + exponential) / first_argument
        )
    second_auxiliary = []
    if series:
        powers = [graph.constant(1)]
        for term in range(1, 13):
            powers.append(powers[-1] * (-second_argument) / term)
        for auxiliary_order in range(max(second_orders) + 1):
            second_auxiliary.append(
                2
                * sum(
                    (
                        powers[term] / (auxiliary_order + term + 1)
                        for term in range(auxiliary_order % 2, 13, 2)
                    ),
                    graph.constant(0),
                )
            )
    else:
        positive = graph.exponential(second_argument)
        negative = graph.exponential(-second_argument)
        reciprocal = 1 / second_argument
        for auxiliary_order in range(max(second_orders) + 1):
            term = reciprocal
            sign = 1 if auxiliary_order % 2 == 0 else -1
            plus, minus = sign * term, term
            for index in range(1, auxiliary_order + 1):
                term = term * (auxiliary_order - index + 1) * reciprocal
                sign = -sign
                plus = plus + sign * term
                minus = minus + term
            second_auxiliary.append(positive * plus - negative * minus)
    overlap = (
        normalization
        * distance.pow(order)
        * sum(
            (
                weight * first_auxiliary[first_order] * second_auxiliary[second_order]
                for weight, first_order, second_order in zip(
                    weights, first_orders, second_orders, strict=True
                )
            ),
            graph.constant(0),
        )
    )
    return graph, names, (overlap, graph.differentiate(overlap, distance))


@cache
def pair_program() -> tuple:
    """Derive the damped BSSE pair response, including supplied overlap JVP."""
    graph = Graph()
    names = (
        "distance",
        "radius",
        "alpha",
        "beta",
        "damping_scale",
        "damping_exponent",
        "prefactor",
        "overlap",
        "overlap_slope",
    )
    distance, radius, alpha, beta, scale, exponent, prefactor, overlap, slope = (
        graph.variable(name) for name in names
    )
    radial_power = graph.exponential(beta * graph.stable_unary("log", distance))
    damping_power = graph.exponential(
        exponent
        * (graph.stable_unary("log", distance) - graph.stable_unary("log", radius))
    )
    bsse = graph.exponential(-alpha * radial_power) / overlap.pow(0.5)
    value = prefactor * bsse * (1 - 1 / (1 + scale * damping_power))
    return (
        graph,
        names,
        (value, graph.differentiate(value, distance, {"overlap": slope})),
    )


def _emit_function(name: str, program: tuple, *, pair: bool = False) -> list[str]:
    graph, names, roots = program
    lines = [
        "template <bool NeedDerivative>",
        f"GENERATIVEQC_GCP_SCALAR_HD inline void {name}("
        + ", ".join(f"double {field}" for field in names)
        + ", double* output) {",
    ]
    for derivative in (False, True):
        selected = roots if derivative else roots[:1]
        lines.append(
            "  if constexpr (NeedDerivative) {"
            if derivative
            else "  if constexpr (!NeedDerivative) {"
        )
        if pair:
            distance, radius, _, beta, _, exponent = (
                graph.variable(field) for field in names[:6]
            )
            selected = graph.replace_subexpressions(
                selected,
                {
                    graph.exponential(
                        beta * graph.stable_unary("log", distance)
                    ): graph.variable("radial_power"),
                    graph.exponential(
                        exponent
                        * (
                            graph.stable_unary("log", distance)
                            - graph.stable_unary("log", radius)
                        )
                    ): graph.variable("damping_power"),
                },
            )
            lines.extend(
                [
                    "    const double radial_power = pow(distance, beta);",
                    "    const double damping_power = pow(distance / radius, damping_exponent);",
                ]
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
        emission_graph, selected = graph.lower_small_integer_powers(
            selected, maximum_absolute_power=8
        )
        emission_graph, selected = emission_graph.canonicalize_associative(
            selected, factor_common=True
        )
        emitter = ScalarCEmitter(emission_graph, dict(zip(names, names, strict=True)))
        emitter.emit(selected)
        lines.extend(emitter.lines)
        lines.extend(
            f"    output[{index}] = {emitter.reference(root)};"
            for index, root in enumerate(selected)
        )
        lines.append("  }")
    lines.append("}")
    return lines


def emit_gcp_scalar_header() -> str:
    """Emit bounded shell-specialized host/device primals and AD responses."""
    lines = [
        "// Generated by tools/generate_gcp_scalar.py; do not edit.",
        "#pragma once",
        "#include <cmath>",
        "#if defined(__CUDACC__)",
        "#define GENERATIVEQC_GCP_SCALAR_HD __host__ __device__",
        "#else",
        "#define GENERATIVEQC_GCP_SCALAR_HD",
        "#endif",
        "namespace generativeqc::generated::dispersion {",
    ]
    lines.extend(_emit_function("gcp_pair", pair_program(), pair=True))
    for shell_product in SHELL_PRODUCTS:
        for series in (False, True):
            name = (
                f"gcp_overlap_{shell_product}_{'series' if series else 'exponential'}"
            )
            lines.extend(_emit_function(name, overlap_program(shell_product, series)))
    lines.extend(
        [
            "template <bool NeedDerivative>",
            "GENERATIVEQC_GCP_SCALAR_HD inline bool gcp_overlap(int shell_product, bool series,",
            "    double distance, double first_exponent, double second_exponent, double* output) {",
            "  switch (shell_product) {",
        ]
    )
    for shell_product in SHELL_PRODUCTS:
        lines.extend(
            [
                f"  case {shell_product}:",
                f"    if (series) gcp_overlap_{shell_product}_series<NeedDerivative>(",
                "        distance, first_exponent, second_exponent, output);",
                f"    else gcp_overlap_{shell_product}_exponential<NeedDerivative>(",
                "        distance, first_exponent, second_exponent, output);",
                "    return true;",
            ]
        )
    lines.extend(
        [
            "  default: return false;",
            "  }",
            "}",
            "}",
            "#undef GENERATIVEQC_GCP_SCALAR_HD",
            "",
        ]
    )
    return "\n".join(lines)
