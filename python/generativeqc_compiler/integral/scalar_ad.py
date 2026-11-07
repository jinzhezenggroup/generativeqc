"""Graph-derived first channels for bounded native recurrence interpreters.

The runtime retains storage, recurrence traversal and special-function value
evaluation. These small schedules own the differentiated arithmetic. Invariant
frontiers are evaluated once per operation, not once per Cartesian channel.
"""

from fractions import Fraction
from typing import cast

from .expr import Expr, Graph
from .scalar_c import ScalarCEmitter


class _Emitter(ScalarCEmitter):
    """Preserve native division values rather than reciprocal multiplication."""

    def _operation_code(self, identifier: int) -> str:
        node = self.graph.nodes[identifier]
        if node.operation == "multiply" and len(node.arguments) == 2:
            for index, operand in enumerate(node.arguments):
                inverse = self.graph.nodes[operand]
                if inverse.operation == "reciprocal":
                    return (
                        f"{self._reference(node.arguments[1 - index])} / "
                        f"{self._reference(inverse.arguments[0])}"
                    )
        return super()._operation_code(identifier)


def scalar_program(name: str) -> tuple[Graph, Expr, Expr]:
    """Differentiate scalar primals, with implicit division/sqrt schedules."""
    graph = Graph()
    left, right = graph.variable("left"), graph.variable("right")
    direction = graph.variable("direction")
    leaves = {field: graph.variable("d" + field) for field in ("left", "right")}
    if name == "add":
        primal = left + right
    elif name == "subtract":
        primal = left - right
    elif name == "multiply":
        primal = left * right
    elif name == "divide":
        primal = left / right
    elif name == "exp":
        primal = graph.exponential(left)
    elif name == "sqrt":
        primal = left.pow(Fraction(1, 2))
    elif name == "erf":
        primal = graph.transcendental_unary("erf", left)
    else:
        raise ValueError(name)
    if name in ("divide", "sqrt"):
        value = graph.variable("value")
        residual = value * right - left if name == "divide" else value * value - left
        response = -graph.differentiate(
            residual, direction, leaves
        ) / graph.differentiate(residual, value)
    else:
        response = graph.differentiate(primal, direction, leaves)
        (response,) = graph.replace_subexpressions(
            (response,), {primal: graph.variable("value")}
        )
    graph, (primal, response) = graph.lower_small_integer_powers((primal, response))
    return graph, primal, response


def _schedule(name: str) -> str:
    graph, primal, response = scalar_program(name)
    dependent, frontier = {}, set()
    for identifier in graph.topological_order((response,)):
        node = graph.nodes[identifier]
        dependent[identifier] = (
            node.operation == "variable" and node.payload in ("dleft", "dright")
        ) or any(dependent[child] for child in node.arguments)
        if dependent[identifier]:
            frontier.update(
                child
                for child in node.arguments
                if not dependent[child] and graph.nodes[child].operation != "constant"
            )
    invariants = tuple(Expr(graph, identifier) for identifier in sorted(frontier))
    fields = tuple(f"invariant{index}" for index in range(len(invariants)))
    (response,) = graph.replace_subexpressions(
        (response,),
        {
            root: graph.variable(field)
            for root, field in zip(invariants, fields, strict=True)
        },
    )
    constructor = _Emitter(graph, {"left": "left", "right": "right"})
    constructor.emit((primal,))
    constructor.names[graph.variable("value").identifier] = "value"
    previous_lines = len(constructor.lines)
    constructor.emit(invariants)
    tangent = ScalarCEmitter(graph, {})
    tangent.emit((response,))
    arguments = "double left, double right" if name in BINARY else "double left"
    tangent_arguments = (
        "double dleft, double dright" if name in BINARY else "double dleft"
    )
    title = name.title()
    value_emitter = _Emitter(graph, {"left": "left", "right": "right"})
    value_emitter.emit((primal,))
    return "\n".join(
        [
            f"struct {title} {{",
            "  double value;",
            *(f"  double {field};" for field in fields),
            f"  GENERATIVEQC_SCALAR_AD_HD static double primal({arguments}) {{",
            *value_emitter.lines,
            f"    return {value_emitter.reference(primal)};",
            "  }",
            f"  GENERATIVEQC_SCALAR_AD_HD explicit {title}({arguments}) {{",
            *constructor.lines[:previous_lines],
            f"    value = {constructor.reference(primal)};",
            *constructor.lines[previous_lines:],
            *(
                f"    {field} = {constructor.reference(root)};"
                for root, field in zip(invariants, fields, strict=True)
            ),
            "  }",
            f"  GENERATIVEQC_SCALAR_AD_HD double tangent({tangent_arguments}) const {{",
            *tangent.lines,
            f"    return {tangent.reference(response)};",
            "  }",
            "};",
        ]
    )


def _moment_response() -> str:
    """Integrate an AD-derived Gaussian monomial, without differentiating values.

    M_n(T) integrates u^(2n) exp(-T u^2) on fixed radial boundaries. AD of
    the integrand supplies a coefficient and an extra even power. Integration
    lowers that monomial to the next value moment, for full/LR/SR alike.
    """
    graph = Graph()
    argument, radial = graph.variable("argument"), graph.variable("radial")
    integrand = graph.exponential(-argument * radial * radial)
    differentiated = graph.differentiate(integrand, argument)

    def monomial(root: Expr) -> tuple[Fraction, int, int]:
        node = graph.node(root)
        if root.identifier == radial.identifier:
            return Fraction(1), 1, 0
        if root.identifier == integrand.identifier:
            return Fraction(1), 0, 1
        if node.operation == "constant":
            return Fraction(cast("Fraction | float", node.payload)), 0, 0
        if node.operation == "multiply":
            coefficient, power, gaussian = Fraction(1), 0, 0
            for child in node.arguments:
                factor, degree, count = monomial(Expr(graph, child))
                coefficient *= factor
                power += degree
                gaussian += count
            return coefficient, power, gaussian
        raise ValueError("Gaussian moment derivative is not an integrable monomial")

    coefficient, power, gaussian = monomial(differentiated)
    if power != 2 or gaussian != 1:
        raise ValueError("Gaussian moment differentiation changed its radial domain")
    result = coefficient * graph.variable("next_moment") * graph.variable("direction")
    emitter = ScalarCEmitter(graph, {})
    emitter.emit((result,))
    return "\n".join(
        [
            "GENERATIVEQC_SCALAR_AD_HD inline double moment_tangent(double next_moment, double direction) {",
            *emitter.lines,
            f"  return {emitter.reference(result)};",
            "}",
        ]
    )


BINARY = ("add", "subtract", "multiply", "divide")
UNARY = ("exp", "sqrt", "erf")


def emit_scalar_ad_header() -> str:
    """Emit shared host/device recurrence arithmetic from the same primals."""
    return "\n".join(
        [
            "// Generated by tools/generate_integral_scalar_ad.py; do not edit.",
            "#pragma once",
            "#include <cmath>",
            "#if defined(__CUDACC__)",
            "#define GENERATIVEQC_SCALAR_AD_HD __host__ __device__",
            "#else",
            "#define GENERATIVEQC_SCALAR_AD_HD",
            "#endif",
            "namespace generativeqc::integrals::generated_scalar_ad {",
            *(_schedule(name) for name in BINARY + UNARY),
            _moment_response(),
            "}",
            "#undef GENERATIVEQC_SCALAR_AD_HD",
            "",
        ]
    )
