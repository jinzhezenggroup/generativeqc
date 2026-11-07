"""Graph-derived arithmetic for the scaled host/device XC point evaluator.

The native evaluator owns numerical scaling, admission and primal branches.
This compiler owns first and mixed directional differentiation, including the
quotient's implicit residual schedule that never squares a tiny denominator.
"""

from fractions import Fraction

from generativeqc_compiler.integral.expr import Expr, Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter, format_constant

POWERS = ((1, 2), (1, 6), (2, 3), (4, 3), (4, 1), (7, 3))


class _Emitter(ScalarCEmitter):
    """Emit typed arithmetic without promoting FP32 point intermediates."""

    target_type = "Arithmetic"

    def _negative_operand(self, identifier: int) -> tuple[int, ...] | None:
        """Recognize sign-only algebra without multiplying a dense constant Jet."""
        node = self.graph.nodes[identifier]
        if node.operation == "multiply":
            for index, operand in enumerate(node.arguments):
                factor = self.graph.nodes[operand]
                if factor.operation == "constant" and factor.payload == -1:
                    return node.arguments[:index] + node.arguments[index + 1 :]
        return None

    def _operation_code(self, identifier: int) -> str:
        node = self.graph.nodes[identifier]
        if node.operation == "add" and len(node.arguments) == 2:
            for index, operand in enumerate(node.arguments):
                positive = self._negative_operand(operand)
                if positive is not None:
                    product = " * ".join(self._reference(value) for value in positive)
                    return f"{self._reference(node.arguments[1 - index])} - ({product})"
        positive = self._negative_operand(identifier)
        if positive is not None:
            product = " * ".join(self._reference(value) for value in positive)
            return f"(-({product}))"
        if node.operation == "reciprocal":
            return f"{self.target_type}{{1.0}} / {self._reference(node.arguments[0])}"
        if node.operation == "multiply" and len(node.arguments) == 2:
            for index, operand in enumerate(node.arguments):
                inverse = self.graph.nodes[operand]
                if inverse.operation == "reciprocal":
                    return (
                        f"{self._reference(node.arguments[1 - index])} / "
                        f"{self._reference(inverse.arguments[0])}"
                    )
        if node.operation == "power":
            return (
                f"point_power({self._reference(node.arguments[0])}, "
                f"{format_constant(node.payload)})"
            )
        if node.operation in ("exp", "log", "log1p", "expm1"):
            return f"point_{node.operation}({self._reference(node.arguments[0])})"
        return super()._operation_code(identifier)


def _emit(
    graph: Graph, roots: tuple, bindings: dict, targets: tuple, scalar: str
) -> str:
    emitter = _Emitter(graph, bindings)
    emitter.target_type = scalar
    for identifier in graph.topological_order(roots):
        node = graph.nodes[identifier]
        if node.operation == "constant":
            emitter.names[identifier] = f"{scalar}{{{format_constant(node.payload)}}}"
    emitter.emit(roots)
    return "\n".join(
        [line.replace("const double ", f"const {scalar} ") for line in emitter.lines]
        + [
            f"  {target} = {emitter.reference(root)};"
            for root, target in zip(roots, targets, strict=True)
        ]
    )


def _program(name: str, exponent: Fraction | None = None) -> tuple:
    graph = Graph()
    left, right = graph.variable("left"), graph.variable("right")
    if name == "add":
        primal = left + right
    elif name == "subtract":
        primal = left - right
    elif name == "multiply":
        primal = left * right
    elif name == "divide":
        primal = left / right
    elif name == "power":
        primal = left.pow(float(exponent))
    elif name in ("log1p", "expm1"):
        primal = graph.stable_unary(name, left)
    elif name == "cube_root":
        primal = graph.exponential(graph.stable_unary("log", left) / 3)
    else:
        raise ValueError(name)
    direction = graph.variable("direction")
    leaves = {key: graph.variable("d" + key) for key in ("left", "right")}
    if name == "divide":
        # Differentiate left-y*right=0 with y held fixed, then solve for dy.
        # The division primitive and this residual describe the same primal;
        # this schedule retains a representable JVP in subnormal XC tails.
        value = graph.variable("value")
        residual = left - value * right
        response = graph.differentiate(
            residual, direction, leaves
        ) / -graph.differentiate(residual, value)
    else:
        response = graph.differentiate(primal, direction, leaves)
    lowered, roots = graph.canonicalize_associative(
        (primal, response), factor_common=False
    )
    return lowered, *roots


def _r2scan_tail() -> str:
    """Differentiate the vacuum continuation and its energy composition.

    Reflect the same quintic primal about its midpoint so its derivative does
    not subtract O(1) terms near the high-density endpoint. The physical width
    is fixed while differentiating; neither spin channel gets a density floor.
    """
    graph = Graph()
    density = graph.variable("density")
    lower, upper = 1.0e-56, 1.0e-52
    coordinate = (density - lower) / (upper - lower)

    def quintic(argument: Expr) -> Expr:
        return argument.pow(3) * (10 + argument * (-15 + 6 * argument))

    weight = graph.select_le(
        coordinate, 0.5, quintic(coordinate), 1 - quintic(1 - coordinate)
    )
    slope = graph.differentiate(weight, density)
    graph, (weight, slope) = graph.lower_small_integer_powers((weight, slope))
    emitter = ScalarCEmitter(graph, {}, coalesce_branches=True)
    emitter.emit((weight, slope))

    composition = Graph()
    energy, scale = (
        composition.variable("unscaled_energy"),
        composition.variable("scale"),
    )
    primal = energy * scale
    response = composition.differentiate(
        primal,
        composition.variable("direction"),
        {
            "unscaled_energy": composition.variable("raw_derivative"),
            "scale": composition.variable("slope"),
        },
    )
    other = composition.differentiate(
        primal,
        composition.variable("direction"),
        {"unscaled_energy": composition.variable("raw_derivative")},
    )
    bindings = {"raw_derivative": "derivative[index]"}
    return "\n".join(
        [
            "/** C2 vacuum continuation, differentiated from its reflected quintic primal. */",
            "GENERATIVEQC_POINT_AD_HD inline void apply_r2scan_tail(double density, double& energy, double* derivative) {",
            *emitter.lines,
            f"  const double scale = {emitter.reference(weight)};",
            f"  const double slope = {emitter.reference(slope)};",
            "  const double unscaled_energy = energy;",
            "  for (unsigned index = 0; index < 2; ++index) {",
            _emit(composition, (response,), bindings, ("derivative[index]",), "double"),
            "  }",
            "  for (unsigned index = 2; index < 7; ++index) {",
            _emit(composition, (other,), bindings, ("derivative[index]",), "double"),
            "  }",
            "  {",
            _emit(composition, (primal,), bindings, ("energy",), "double"),
            "  }",
            "}",
        ]
    )


def _first(name: str, exponent: Fraction | None = None) -> str:
    graph, primal, response = _program(name, exponent)
    bindings = {
        "left": "left.v",
        "right": "right.v",
        "dleft": "left.d[index]",
        "dright": "right.d[index]",
        "value": "result.v",
    }
    if name == "cube_root":
        head = "  result.v = point_cbrt(left.v);"
    else:
        head = _emit(graph, (primal,), bindings, ("result.v",), "Arithmetic")
    # Bind a primal already evaluated by the same expression or its stable
    # cbrt lowering; never reconstruct exp(log(x)/3) for the value path.
    (response,) = graph.replace_subexpressions(
        (response,), {primal: graph.variable("value")}
    )
    emitter = _Emitter(graph, bindings)
    for identifier in graph.topological_order((response,)):
        node = graph.nodes[identifier]
        if node.operation == "constant":
            emitter.names[identifier] = f"Arithmetic{{{format_constant(node.payload)}}}"
    depends_on_channel = {}
    invariant_frontier = set()
    for identifier in graph.topological_order((response,)):
        node = graph.nodes[identifier]
        depends_on_channel[identifier] = (
            node.operation == "variable" and node.payload in ("dleft", "dright")
        ) or any(depends_on_channel[operand] for operand in node.arguments)
        if depends_on_channel[identifier]:
            invariant_frontier.update(
                operand
                for operand in node.arguments
                if not depends_on_channel[operand]
                and graph.nodes[operand].operation not in ("constant", "variable")
            )
    emitter.emit(
        tuple(Expr(graph, identifier) for identifier in sorted(invariant_frontier))
    )
    invariant_lines = [
        line.replace("const double ", "const Arithmetic ") for line in emitter.lines
    ]
    emitter.lines.clear()
    emitter.emit((response,))
    body = "\n".join(
        [line.replace("const double ", "const Arithmetic ") for line in emitter.lines]
        + [f"  result.d[index] = {emitter.reference(response)};"]
    )
    return (
        head
        + "\n  if constexpr (DerivativeCount > 0) {\n"
        + ("  if (left.v == Arithmetic{0}) return result;\n" if name == "power" else "")
        + "\n".join(invariant_lines)
        + "\n  for (unsigned index = 0; index < DerivativeCount; ++index) {\n"
        + body
        + "\n  }\n  }"
    )


def _nested(name: str, exponent: Fraction | None = None) -> str:
    """Emit mixed scalar lanes directly, sharing nonlinear work across channels.

    A nested C++ Jet expression hides invariant powers behind dense temporary
    objects. Scalar AD exposes one point-local DAG instead. Implicit residuals
    keep division/logarithm responses representable without tiny squared
    denominators; their channel equations are differentiated before solving.
    """
    graph, primal, response = _program(name, exponent)
    channel_names = ("cleft", "cright", "cdleft", "cdright", "cvalue")
    channel_seeds = {
        key: graph.variable("c" + key)
        for key in ("left", "right", "dleft", "dright", "value")
    }
    (first,) = graph.replace_subexpressions(
        (response,),
        {
            graph.variable("dleft"): channel_seeds["left"],
            graph.variable("dright"): channel_seeds["right"],
        },
    )
    if name in ("divide", "log1p"):
        tangent = graph.variable("tangent_value")
        response_node = graph.nodes[response.identifier]
        fractions = [
            (index, graph.nodes[operand])
            for index, operand in enumerate(response_node.arguments)
            if graph.nodes[operand].operation == "reciprocal"
        ]
        if (
            response_node.operation != "multiply"
            or len(response_node.arguments) != 2
            or len(fractions) != 1
        ):
            raise ValueError(
                "point response no longer admits the qualified implicit quotient schedule"
            )
        inverse_index, inverse = fractions[0]
        numerator = Expr(graph, response_node.arguments[1 - inverse_index])
        denominator = Expr(graph, inverse.arguments[0])
        residual = numerator - tangent * denominator
        mixed = graph.differentiate(
            residual, graph.variable("channel"), channel_seeds
        ) / -graph.differentiate(residual, tangent)
    else:
        mixed = graph.differentiate(response, graph.variable("channel"), channel_seeds)
    graph, (primal, response, first, mixed) = graph.canonicalize_associative(
        (primal, response, first, mixed), factor_common=False
    )
    bindings = {
        "left": "left.v",
        "right": "right.v",
        "dleft": "left.tangent.v",
        "dright": "right.tangent.v",
        "value": "result.v",
        "tangent_value": "result.tangent.v",
        "cleft": "left.d[index]",
        "cright": "right.d[index]",
        "cdleft": "left.tangent.d[index]",
        "cdright": "right.tangent.d[index]",
        "cvalue": "result.d[index]",
    }
    emitter = _Emitter(graph, bindings)
    emitter.target_type = "double"
    lines = []
    for root, target in ((primal, "result.v"), (response, "result.tangent.v")):
        emitter.emit((root,))
        lines.extend(emitter.lines)
        lines.append(f"  {target} = {emitter.reference(root)};")
        emitter.lines.clear()
        if name == "power" and root == primal:
            lines.append("  if (left.v == 0.0) return result;")
    dependent = {}
    frontier = set()
    for identifier in graph.topological_order((first, mixed)):
        node = graph.nodes[identifier]
        dependent[identifier] = (
            node.operation == "variable" and node.payload in channel_names
        ) or any(dependent[child] for child in node.arguments)
        if dependent[identifier]:
            frontier.update(
                child
                for child in node.arguments
                if not dependent[child]
                and graph.nodes[child].operation not in ("constant", "variable")
            )
    emitter.emit(tuple(Expr(graph, identifier) for identifier in sorted(frontier)))
    lines.extend(emitter.lines)
    emitter.lines.clear()
    lines.append("  for (unsigned index = 0; index < 8; ++index) {")
    for root, target in (
        (first, "result.d[index]"),
        (mixed, "result.tangent.d[index]"),
    ):
        emitter.emit((root,))
        lines.extend(emitter.lines)
        lines.append(f"  {target} = {emitter.reference(root)};")
        emitter.lines.clear()
    lines.append("  }")
    return "\n".join(lines)


def _point_primitives() -> str:
    functions = []
    for name in ("exp", "log", "log1p", "expm1", "cbrt"):
        functions.append(f"""
template<class Arithmetic> GENERATIVEQC_POINT_AD_HD inline Arithmetic point_{name}(Arithmetic input) {{
  if constexpr (std::is_same_v<Arithmetic, float>) return ::{name}f(input);
  else return ::{name}(input);
}}
""")
    functions.append("""
template<class Arithmetic> GENERATIVEQC_POINT_AD_HD inline Arithmetic point_power(Arithmetic input, double exponent) {
  if constexpr (std::is_same_v<Arithmetic, float>) return ::powf(input, static_cast<float>(exponent));
  else return ::pow(input, exponent);
}
""")
    return "".join(functions)


def _subnormal_gradient() -> str:
    """Differentiate a ratio in local, representable perturbation coordinates."""
    graph = Graph()
    coordinate = graph.variable("coordinate")
    ratio, radius = graph.variable("ratio"), graph.variable("radius")
    direction_gradient, direction_rho, direction_radius = (
        graph.variable(name)
        for name in ("direction_gradient", "direction_rho", "direction_radius")
    )
    primal = (
        (ratio + coordinate * direction_gradient)
        / (1 + coordinate * direction_rho)
        / (1 + coordinate * direction_radius)
        / radius
    )
    derivative = graph.differentiate(primal, coordinate)
    (response,) = graph.replace_subexpressions(
        (derivative,), {coordinate: graph.constant(0)}
    )
    bindings = {
        "ratio": "ratio",
        "radius": "rho13.v",
        "direction_gradient": "(gradient.d[index] / rho.v)",
        "direction_rho": "(rho.d[index] / rho.v)",
        "direction_radius": "(rho13.d[index] / rho13.v)",
    }
    body = _emit(graph, (response,), bindings, ("result.d[index]",), "Arithmetic")
    return (
        """
/** Fixed local scales keep each tangent representable before subtraction. */
template<unsigned DerivativeCount, class Arithmetic>
GENERATIVEQC_POINT_AD_HD inline PointJet<DerivativeCount, Arithmetic> subnormal_reduced_gradient(
    const PointJet<DerivativeCount, Arithmetic>& gradient,
    const PointJet<DerivativeCount, Arithmetic>& rho,
    const PointJet<DerivativeCount, Arithmetic>& rho13) {
  const Arithmetic ratio = gradient.v / rho.v;
  PointJet<DerivativeCount, Arithmetic> result;
  result.v = ratio / rho13.v;
  for (unsigned index = 0; index < DerivativeCount; ++index) {
"""
        + body
        + "\n  }\n  return result;\n}\n"
    )


def _scaled_exchange(name: str) -> str:
    """AD in dimensionless physical coordinates, then restore fixed scales.

    Density perturbations are logarithmic and gradient perturbations are scaled
    by the reference rho^(4/3) (low branch) or |gradient| (high branch). Thus the
    density potential remains representable when the energy scale underflows.
    """
    graph = Graph()
    density = graph.variable("density_coordinate")
    norm = graph.variable("norm_coordinate")
    cx, kappa, mu = (graph.variable(key) for key in ("cx", "kappa", "mu"))
    inputs = tuple(graph.variable(f"component_{axis}") for axis in range(3))
    density_scale = graph.exponential(4 * density / 3)
    if name == "low":
        squared = graph.sum(value * value for value in inputs)
        reduced_squared = squared * graph.exponential(-8 * density / 3)
        denominator = kappa + mu * reduced_squared
        enhancement = 1 + kappa - kappa * kappa / denominator
        stable_value = 1 + kappa * mu * reduced_squared / denominator
    elif name == "high":
        inverse = graph.variable("inverse") * graph.exponential(4 * density / 3 - norm)
        squared = inverse * inverse
        denominator = kappa * squared + mu
        enhancement = 1 + kappa * mu / denominator
        stable_value = 1 + (kappa - kappa * kappa * squared / denominator)
    elif name == "lda":
        enhancement = graph.constant(1)
        stable_value = enhancement
    else:
        raise ValueError(name)
    normalized = -cx * density_scale * enhancement
    energy = graph.variable("rho43") * normalized
    potential = graph.variable("rho13") * graph.differentiate(normalized, density)
    if name == "low":
        gradients = tuple(graph.differentiate(normalized, value) for value in inputs)
    elif name == "high":
        radial = graph.variable("inverse") * graph.differentiate(normalized, norm)
        gradients = tuple(radial * value for value in inputs)
    else:
        gradients = (graph.constant(0),) * 3
    # Differentiate a reciprocal form without quotient-rule cancellation, then
    # lower its shared primal back to the native branch's stable value form.
    roots = graph.replace_subexpressions(
        (energy, potential, *gradients), {enhancement: stable_value}
    )
    roots = graph.replace_subexpressions(
        roots,
        {density: graph.constant(0), norm: graph.constant(0)},
    )
    graph, roots = graph.lower_small_integer_powers(roots, maximum_absolute_power=4)
    graph, roots = graph.canonicalize_associative(roots, factor_common=True)
    bindings = {key: key for key in ("cx", "kappa", "mu", "rho13", "rho43", "inverse")}
    bindings.update({f"component_{axis}": f"components[{axis}]" for axis in range(3)})
    body = _emit(
        graph,
        roots,
        bindings,
        tuple(f"output[{index}]" for index in range(5)),
        "Scalar",
    )
    return f"""
/** Physical {name}-branch exchange energy and first potentials from one primal. */
template<class Scalar>
GENERATIVEQC_POINT_AD_HD inline void generated_exchange_{name}(
    Scalar rho13, Scalar rho43, const Scalar components[3], Scalar inverse,
    Scalar cx, Scalar kappa, Scalar mu, Scalar output[5]) {{
{body}
}}
"""


def emit_point_ad_header() -> str:
    """Generate point arithmetic from primal Graphs, not derivative templates."""
    lines = [
        "// Generated by tools/generate_xc_point_ad.py; do not edit.",
        "#pragma once",
        "#include <cmath>",
        "#include <type_traits>",
        "#if defined(__CUDACC__)",
        "#define GENERATIVEQC_POINT_AD_HD __host__ __device__",
        "#else",
        "#define GENERATIVEQC_POINT_AD_HD",
        "#endif",
        "namespace generativeqc::dft::point::detail {",
        _point_primitives(),
        """
/** Physical first channels with fixed numerical scales; no density floor. */
template<unsigned DerivativeCount, class Arithmetic = double> struct PointJet {
  static constexpr bool second_order = false;
  Arithmetic v{};
  Arithmetic d[DerivativeCount ? DerivativeCount : 1]{};
  GENERATIVEQC_POINT_AD_HD PointJet() {}
  GENERATIVEQC_POINT_AD_HD PointJet(double value) : v(value) {}
  GENERATIVEQC_POINT_AD_HD static PointJet variable(double value, unsigned index) {
    PointJet result(value);
    result.d[index] = Arithmetic{1};
    return result;
  }
""",
    ]
    for name, symbol in (
        ("add", "+"),
        ("subtract", "-"),
        ("multiply", "*"),
        ("divide", "/"),
    ):
        lines.extend(
            [
                f"GENERATIVEQC_POINT_AD_HD friend PointJet operator{symbol}(const PointJet& left, const PointJet& right) {{",
                "  PointJet result;",
                _first(name),
                "  return result;",
                "}",
            ]
        )
    lines.extend(
        [
            "GENERATIVEQC_POINT_AD_HD friend PointJet operator-(const PointJet& value) { return PointJet(0.0) - value; }",
            "};",
            "using Jet = PointJet<8>;",
        ]
    )
    for name in ("log1p", "expm1", "cube_root"):
        lines.extend(
            [
                "template<unsigned DerivativeCount, class Arithmetic>",
                f"GENERATIVEQC_POINT_AD_HD inline PointJet<DerivativeCount, Arithmetic> {name}(const PointJet<DerivativeCount, Arithmetic>& left) {{",
                "  PointJet<DerivativeCount, Arithmetic> result;",
                _first(name),
                "  return result;",
                "}",
            ]
        )
    lines.extend(
        [
            "template<int Numerator, int Denominator = 1, unsigned DerivativeCount, class Arithmetic>",
            "GENERATIVEQC_POINT_AD_HD inline PointJet<DerivativeCount, Arithmetic> power(const PointJet<DerivativeCount, Arithmetic>& left) {",
            "  PointJet<DerivativeCount, Arithmetic> result;",
        ]
    )
    for index, (numerator, denominator) in enumerate(POWERS):
        lines.extend(
            [
                ("  if constexpr" if index == 0 else "  else if constexpr")
                + f" (Numerator == {numerator} && Denominator == {denominator}) {{",
                _first("power", Fraction(numerator, denominator)),
                "  }",
            ]
        )
    lines.extend(
        [
            '  else static_assert(sizeof(Arithmetic) == 0, "unregistered XC point power");',
            "  return result;",
            "}",
            """
/** Mixed first-channel/directional response without a stored point Hessian. */
struct ResponseJet : Jet {
  static constexpr bool second_order = true;
  Jet tangent;
  GENERATIVEQC_POINT_AD_HD ResponseJet(double value = 0.0) : Jet(value) {}
  GENERATIVEQC_POINT_AD_HD ResponseJet(const Jet& value, const Jet& direction) : Jet(value), tangent(direction) {}
  GENERATIVEQC_POINT_AD_HD const Jet& base() const { return *this; }
  GENERATIVEQC_POINT_AD_HD static ResponseJet variable(double value, unsigned index, double direction) {
    return {Jet::variable(value, index), Jet(direction)};
  }
""",
        ]
    )
    for name, symbol in (
        ("add", "+"),
        ("subtract", "-"),
        ("multiply", "*"),
        ("divide", "/"),
    ):
        lines.extend(
            [
                f"GENERATIVEQC_POINT_AD_HD friend ResponseJet operator{symbol}(const ResponseJet& left, const ResponseJet& right) {{",
                "  ResponseJet result;",
                _nested(name),
                "  return result;",
                "}",
            ]
        )
    lines.extend(
        [
            "GENERATIVEQC_POINT_AD_HD friend ResponseJet operator-(const ResponseJet& value) { return ResponseJet(0.0) - value; }",
            "};",
        ]
    )
    for name in ("log1p", "expm1"):
        lines.extend(
            [
                f"GENERATIVEQC_POINT_AD_HD inline ResponseJet {name}(const ResponseJet& left) {{",
                "  ResponseJet result;",
                _nested(name),
                "  return result;",
                "}",
            ]
        )
    lines.extend(
        [
            "template<int Numerator, int Denominator = 1>",
            "GENERATIVEQC_POINT_AD_HD inline ResponseJet power(const ResponseJet& left) {",
            "  ResponseJet result;",
        ]
    )
    for index, (numerator, denominator) in enumerate(POWERS):
        lines.extend(
            [
                ("  if constexpr" if index == 0 else "  else if constexpr")
                + f" (Numerator == {numerator} && Denominator == {denominator}) {{",
                _nested("power", Fraction(numerator, denominator)),
                "  }",
            ]
        )
    lines.extend(
        [
            '  else static_assert(Numerator != Numerator, "unregistered XC response power");',
            "  return result;",
            "}",
            _subnormal_gradient(),
            *(_scaled_exchange(name) for name in ("low", "high", "lda")),
            _r2scan_tail(),
            "}  // namespace generativeqc::dft::point::detail",
            "#undef GENERATIVEQC_POINT_AD_HD",
            "",
        ]
    )
    return "\n".join(lines)
