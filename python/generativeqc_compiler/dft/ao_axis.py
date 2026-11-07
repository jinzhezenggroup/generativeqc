"""Gaussian Cartesian axis primals and automatically differentiated jets.

CPU coefficient lowering retains zero-coefficient IEEE operations. CUDA lowers
the same primal to radial-free scalar expressions. Neither schedule evaluates
another exponential or divides by a potentially underflowed Gaussian.
"""

from fractions import Fraction
from functools import cache

from generativeqc_compiler.integral.expr import AlgebraForm, Expr, Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter


def _gaussian_primal(power: int) -> tuple[Graph, Expr, Expr, Expr]:
    graph = Graph()
    coordinate, exponent = graph.variable("x"), graph.variable("a")
    gaussian = graph.exponential(-exponent * coordinate * coordinate)
    polynomial = graph.constant(1)
    for _ in range(power):
        polynomial = polynomial * coordinate
    return graph, coordinate, gaussian, polynomial * gaussian


def axis_expression(power: int, derivative: int) -> tuple[Graph, Expr]:
    """Return exp(+a*x*x) d^d[x^l exp(-a*x*x)], without factorials."""
    if (
        type(power) is not int
        or type(derivative) is not int
        or not (0 <= power <= 3 and 0 <= derivative <= 3)
    ):
        raise ValueError("AO axis domain is l,d in [0,3]")
    graph, coordinate, gaussian, root = _gaussian_primal(power)
    for _ in range(derivative):
        root = graph.differentiate(root, coordinate)
    (root,) = graph.replace_subexpressions((root,), {gaussian: graph.constant(1)})
    return graph, root


def _polynomial_terms(power: int) -> tuple[Graph, dict[int, Expr]]:
    """Collect powers of x after AD of one complete Gaussian monomial."""
    graph, coordinate, gaussian, primal = _gaussian_primal(power)
    response = graph.differentiate(primal, coordinate)
    terms: dict[int, dict[int, Expr]] = {}
    for identifier in graph.topological_order((response,)):
        node = graph.nodes[identifier]
        if identifier == gaussian.identifier:
            current = {0: graph.constant(1)}
        elif identifier == coordinate.identifier:
            current = {1: graph.constant(1)}
        elif node.operation in ("variable", "constant"):
            current = {0: Expr(graph, identifier)}
        elif node.operation == "add":
            current = {}
            for child in node.arguments:
                for degree, coefficient in terms[child].items():
                    current[degree] = (
                        current.get(degree, graph.constant(0)) + coefficient
                    )
        elif node.operation == "multiply":
            current = {0: graph.constant(1)}
            for child in node.arguments:
                product = {}
                for first_degree, first in current.items():
                    for second_degree, second in terms[child].items():
                        degree = first_degree + second_degree
                        product[degree] = (
                            product.get(degree, graph.constant(0)) + first * second
                        )
                current = product
        elif node.operation == "exp":
            raise ValueError("AO derivative acquired a second Gaussian")
        else:
            raise ValueError(f"nonpolynomial AO factor {node.operation}")
        terms[identifier] = current
    coefficients = terms[response.identifier]
    degrees = sorted(coefficients)
    graph, roots = graph.apply_algebra_form(
        tuple(coefficients[degree] for degree in degrees), AlgebraForm.FACTORED_NARY
    )
    return graph, dict(zip(degrees, roots, strict=True))


def _signed_coefficient(graph: Graph, coefficient: Expr) -> tuple[str, Expr]:
    """Expose a constant sign without changing the coefficient's IEEE product."""
    node = graph.node(coefficient)
    if node.operation == "constant" and node.payload < 0:
        return "-=", graph.constant(-node.payload)
    if node.operation == "multiply":
        for child in node.arguments:
            factor = graph.nodes[child]
            if factor.operation == "constant" and factor.payload < 0:
                return "-=", graph.multiply_many(
                    graph.constant(-factor.payload)
                    if argument == child
                    else Expr(graph, argument)
                    for argument in node.arguments
                )
    return "+=", coefficient


def _coefficient_monomial(graph: Graph, root: Expr) -> tuple[Fraction, int]:
    node = graph.node(root)
    if node.operation == "constant":
        return node.payload, 0
    if node.operation == "variable" and node.payload == "a":
        return Fraction(1), 1
    if node.operation == "multiply":
        factor, exponent = Fraction(1), 0
        for child in node.arguments:
            child_factor, child_exponent = _coefficient_monomial(
                graph, Expr(graph, child)
            )
            factor *= child_factor
            exponent += child_exponent
        return factor, exponent
    raise ValueError("nonmonomial Gaussian coefficient in bound lowering")


@cache
def axis_bound_transitions(
    maximum_degree: int,
) -> tuple[tuple[tuple[int, float, int], ...], ...]:
    """Sparse absolute AD operator for Gaussian polynomial coefficients.

    Each column contains (target degree, positive factor, alpha exponent).
    Applying the operator with upward-rounded arithmetic bounds every AD
    contribution before cancellation. It is deliberately not the absolute
    value of a fully simplified higher derivative, which can be a smaller
    bound and change the retained AO domain.
    """
    if type(maximum_degree) is not int or maximum_degree < 0:
        raise ValueError("maximum AO bound degree must be a nonnegative integer")
    columns = []
    for degree in range(maximum_degree + 1):
        graph, coefficients = _polynomial_terms(degree)

        column = []
        for target, coefficient in sorted(coefficients.items()):
            factor, exponent = _coefficient_monomial(graph, coefficient)
            if factor:
                column.append((target, float(abs(factor)), exponent))
        columns.append(tuple(column))
    return tuple(columns)


def emit_axis_header() -> str:
    """Emit bounded coefficient schedules, preserving legacy nonfinite admission.

    Keeping all intermediate coefficients, including zeros, preserves Inf*0
    rejection for extreme finite exponents. Only the differentiation and
    coefficient collection are symbolic; Horner evaluation stays operational.
    """
    lines = [
        "// Generated by tools/generate_ao_axis.py; do not edit.",
        "#pragma once",
        "#include <array>",
        "#include <cmath>",
        "#if defined(__GNUC__)",
        "#define GENERATIVEQC_AO_AXIS_NOINLINE __attribute__((noinline))",
        "#else",
        "#define GENERATIVEQC_AO_AXIS_NOINLINE",
        "#endif",
        "namespace generativeqc::dft::generated_ao {",
    ]
    for angular in range(7):
        for derivative in range(1, 7 - angular):
            lines.extend(
                [
                    f"static inline double axis_{angular}_{derivative}(double a, double x) {{",
                    "  std::array<double, 7> coefficients0{};",
                    f"  coefficients0[{angular}] = 1.0;",
                ]
            )
            for stage in range(derivative):
                lines.append(f"  std::array<double, 7> coefficients{stage + 1}{{}};")
                for power in range(angular + stage + 1):
                    graph, terms = _polynomial_terms(power)
                    signed = tuple(
                        (degree, *_signed_coefficient(graph, root))
                        for degree, root in sorted(terms.items())
                    )
                    emitter = ScalarCEmitter(graph, {})
                    emitter.emit(tuple(root for _, _, root in signed))
                    lines.extend(["  {", *emitter.lines])
                    for degree, sign, root in signed:
                        lines.append(
                            f"    coefficients{stage + 1}[{degree}] {sign} {emitter.reference(root)} * coefficients{stage}[{power}];"
                        )
                    lines.append("  }")
            lines.append(
                f"  double result = coefficients{derivative}[{angular + derivative}];"
            )
            for degree in reversed(range(angular + derivative)):
                lines.append(
                    f"  result = result * x + coefficients{derivative}[{degree}];"
                )
            lines.extend(["  return result;", "}"])
    lines.extend(
        [
            "static GENERATIVEQC_AO_AXIS_NOINLINE double axis_internal_nonzero(unsigned angular, unsigned derivative, double a, double x) {",
            "  switch (7 * angular + derivative) {",
        ]
    )
    for angular in range(7):
        for derivative in range(1, 7 - angular):
            if angular <= 3 and derivative <= 3:
                continue
            lines.append(
                f"    case {7 * angular + derivative}: return axis_{angular}_{derivative}(a, x);"
            )
    lines.extend(
        [
            "  }",
            "  return NAN;",
            "}",
            "static inline double axis_nonzero(unsigned angular, unsigned derivative, double a, double x) {",
            "  if (angular <= 3 && derivative <= 3) {",
            "    switch (3 * angular + derivative - 1) {",
        ]
    )
    for angular in range(4):
        for derivative in range(1, 4):
            lines.append(
                f"      case {3 * angular + derivative - 1}: return axis_{angular}_{derivative}(a, x);"
            )
    lines.extend(
        [
            "    }",
            "  }",
            "  return axis_internal_nonzero(angular, derivative, a, x);",
            "}",
            "}",
            "#undef GENERATIVEQC_AO_AXIS_NOINLINE",
            "",
        ]
    )
    return "\n".join(lines)
