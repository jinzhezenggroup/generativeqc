"""Stable scalar AD schedules for nonlocal correlation pair contractions.

Logarithmic factorization is a representation of the positive denominator,
not a supplied pair derivative. Differentiate that primal with the shared
Graph engine, then bind its signed value to the already evaluated energy.
No logarithm or exponential survives in the generated pair arithmetic.
"""

from fractions import Fraction
from typing import cast

from generativeqc_compiler.integral.expr import AlgebraForm, Expr, Graph
from generativeqc_compiler.integral.scalar_ad import _Emitter
from generativeqc_compiler.tensor.ir import (
    Node,
    add,
    constant,
    divide,
    input_tensor,
    multiply,
    power,
)
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import TensorSpec

CONTRACTION_INPUTS = {
    "pair_feature_values": (
        "sum_domega",
        "sum_dkappa",
        "domega_drho",
        "domega_dsigma",
        "dkappa_drho",
    ),
    "row_sum_values": ("sum_phi", "sum_rho", "sum_sigma", "rho", "beta", "coefficient"),
    "weight_feature_value": ("sum_phi", "rho", "beta", "coefficient"),
}


def _rational_products(
    graph: Graph, root: Expr, *, expand_numerator_powers: bool
) -> Expr:
    """Normalize rational product regions of a positive-domain scalar DAG.

    This opt-in lowering gathers exact coefficients and integer powers without
    distributing sums. It is not the general IEEE-preserving Graph default:
    nonlocal scale policies explicitly qualify this representation against
    their raw-domain admission, including overflow, underflow and signed zero.
    """

    def normalized(expression: Expr) -> Expr:
        node = graph.node(expression)
        if node.operation == "add":
            return graph.sum(normalized(Expr(graph, child)) for child in node.arguments)
        if node.operation not in ("multiply", "reciprocal", "power"):
            return expression
        coefficient = Fraction(1)
        powers: dict[int, int] = {}

        def collect(value: Expr, exponent: int) -> None:
            nonlocal coefficient
            item = graph.node(value)
            if item.operation == "multiply":
                for child in item.arguments:
                    collect(Expr(graph, child), exponent)
            elif item.operation == "reciprocal":
                collect(Expr(graph, item.arguments[0]), -exponent)
            elif item.operation == "constant":
                coefficient *= (
                    Fraction(cast("Fraction | float", item.payload)) ** exponent
                )
            elif (
                item.operation == "power"
                and float(cast("Fraction | float", item.payload)).is_integer()
            ):
                collect(
                    Expr(graph, item.arguments[0]),
                    exponent * int(cast("Fraction | float", item.payload)),
                )
            else:
                atom = normalized(value) if item.operation == "add" else value
                powers[atom.identifier] = powers.get(atom.identifier, 0) + exponent

        collect(expression, 1)
        numerator, denominator = (
            graph.constant(coefficient.numerator),
            graph.constant(coefficient.denominator),
        )
        for identifier in sorted(
            powers, key=lambda identifier: str(graph.nodes[identifier].payload)
        ):
            exponent = powers[identifier]
            if exponent > 0:
                if expand_numerator_powers:
                    for _ in range(exponent):
                        numerator = numerator * Expr(graph, identifier)
                else:
                    numerator = numerator * Expr(graph, identifier).pow(exponent)
            elif exponent < 0:
                denominator = denominator * Expr(graph, identifier).pow(-exponent)
        return numerator / denominator

    return normalized(root)


def _local_scale_primal(cuda: bool) -> tuple[Graph, Expr, Expr]:
    graph = Graph()
    rho, sigma, damping, gradient_coefficient, pi, plasma = (
        graph.variable(name)
        for name in ("rho", "sigma", "b", "c", "pi", "four_pi_over_three")
    )
    ratio = sigma / (rho * rho)
    omega_squared = (
        gradient_coefficient * ratio * ratio
        if cuda
        else gradient_coefficient * (sigma * sigma) / ((rho * rho) * (rho * rho))
    ) + plasma * rho
    kappa = damping * 1.5 * pi * (rho / (9 * pi)).pow(1 / 6)
    return graph, omega_squared, kappa


def local_scale_primal_cpp(*, cuda: bool) -> str:
    """Emit local values from the same backend primal differentiated below."""
    graph, omega_squared, kappa = _local_scale_primal(cuda)
    rho, sigma = graph.variable("rho"), graph.variable("sigma")
    emitter = _Emitter(graph, {})
    if cuda:
        emitter.names[(sigma / (rho * rho)).identifier] = "ratio"
    else:
        emitter.names[((rho * rho) * (rho * rho)).identifier] = "rho4"
        emitter.names[(sigma * sigma).identifier] = "sigma2"
    omega = omega_squared.pow(0.5)
    emitter.emit((omega, kappa))
    return "\n".join(
        [
            *emitter.lines,
            f"  out.omega = {emitter.reference(omega)};",
            f"  out.kappa = {emitter.reference(kappa)};",
        ]
    )


def local_scale_response_cpp(*, cuda: bool) -> str:
    """Differentiate local-scale primals via their positive implicit residuals.

    The sixth-root kappa relation specifies the physical rational exponent,
    independently of the value evaluator's rounded pow exponent. CPU and CUDA
    retain their qualified positive-power schedules and their original raw
    failure domains. No derivative expression is supplied by a C++ template.
    """
    graph, omega_squared, _ = _local_scale_primal(cuda)
    rho, sigma, omega, kappa = (
        graph.variable(name) for name in ("rho", "sigma", "omega", "kappa")
    )
    omega_residual = omega * omega - omega_squared
    roots = [
        -graph.differentiate(omega_residual, coordinate)
        / graph.differentiate(omega_residual, omega)
        for coordinate in (rho, sigma)
    ]
    kappa_residual = _log_primal(graph, kappa.pow(6)) - graph.stable_unary("log", rho)
    roots.append(
        -graph.differentiate(kappa_residual, rho)
        / graph.differentiate(kappa_residual, kappa)
    )
    graph, roots = graph.apply_algebra_form(tuple(roots), AlgebraForm.FACTORED_NARY)
    roots = tuple(
        _rational_products(graph, root, expand_numerator_powers=cuda) for root in roots
    )
    rho, sigma = graph.variable("rho"), graph.variable("sigma")
    if not cuda:
        roots = graph.replace_subexpressions(
            roots,
            {
                rho.pow(4): graph.variable("rho4"),
                rho.pow(5): graph.variable("rho5"),
                sigma.pow(2): graph.variable("sigma2"),
            },
        )
    emitter = _Emitter(graph, {"omega": "out.omega", "kappa": "out.kappa"})
    emitter.emit(roots)
    lines = emitter.lines
    for target, root in zip(
        ("domega_drho", "domega_dsigma", "dkappa_drho"), roots, strict=True
    ):
        lines.append(f"  out.{target} = {emitter.reference(root)};")
    return "\n".join(lines).replace("std::pow(", "::pow(")


def row_feature_responses(bindings: dict[str, Node]) -> dict[str, Node]:
    """Differentiate the symmetric ordered-row energy and compose local jets.

    Pair parameter sums are already adjoints. Their contraction is a linear
    primal in the local scales. AD composes that primal with the row's density
    factor; no row product/chain rule is supplied separately.
    """
    graph = Graph()
    variables = {name: graph.variable(name) for name in bindings}
    omega, kappa = graph.variable("omega"), graph.variable("kappa")
    linear_pair = variables["sum_domega"] * omega + variables["sum_dkappa"] * kappa
    rho = variables["rho"]
    partials = []
    for coordinate, omega_seed, kappa_seed in (
        (rho, variables["domega_drho"], variables["dkappa_drho"]),
        (graph.variable("sigma"), variables["domega_dsigma"], graph.constant(0)),
    ):
        pair_response = graph.differentiate(
            linear_pair, coordinate, {"omega": omega_seed, "kappa": kappa_seed}
        )
        row_energy = rho * (variables["beta"] + variables["sum_phi"])
        partials.append(
            variables["coefficient"]
            * graph.differentiate(row_energy, coordinate, {"sum_phi": pair_response})
        )
    return {
        name: _tensor_root(graph, root, bindings)
        for name, root in zip(("vrho", "vsigma"), partials, strict=True)
    }


def build_nonlocal_contraction_program(name: str) -> Program:
    """Emit demanded raw-row/weight pullbacks without moving traversal or sums."""
    graph = Graph()
    values = {field: graph.variable(field) for field in CONTRACTION_INPUTS[name]}
    if name == "pair_feature_values":
        omega, kappa = graph.variable("omega"), graph.variable("kappa")
        primal = values["sum_domega"] * omega + values["sum_dkappa"] * kappa
        roots = {
            "dphi_drho": graph.differentiate(
                primal,
                graph.variable("rho"),
                {"omega": values["domega_drho"], "kappa": values["dkappa_drho"]},
            ),
            "dphi_dsigma": graph.differentiate(
                primal, graph.variable("sigma"), {"omega": values["domega_dsigma"]}
            ),
        }
    elif name == "row_sum_values":
        rho = values["rho"]
        primal = rho * (values["beta"] + values["sum_phi"])
        roots = {
            "vrho": values["coefficient"]
            * graph.differentiate(primal, rho, {"sum_phi": values["sum_rho"]}),
            "vsigma": graph.differentiate(
                values["coefficient"] * primal,
                graph.variable("sigma"),
                {"sum_phi": values["sum_sigma"]},
            ),
        }
    elif name == "weight_feature_value":
        weight = graph.variable("weight")
        primal = (
            values["coefficient"]
            * values["rho"]
            * weight
            * (values["beta"] + values["sum_phi"])
        )
        roots = {"weight_derivative": graph.differentiate(primal, weight)}
    else:
        raise ValueError(f"unknown nonlocal contraction {name}")
    bindings = {
        field: input_tensor(field, TensorSpec((), role="input", differentiable=True))
        for field in CONTRACTION_INPUTS[name]
    }
    return Program(
        {field: _tensor_root(graph, root, bindings) for field, root in roots.items()},
        provenance={
            "kind": "nonlocal-correlation-generated-pullback",
            "closure": name,
            "lowering_version": "nonlocal-serial-row-ad-fp64-v1",
        },
    )


def _log_primal(graph: Graph, root: Expr) -> Expr:
    node = graph.node(root)
    if node.operation == "multiply":
        return graph.sum(
            _log_primal(graph, Expr(graph, child)) for child in node.arguments
        )
    if node.operation == "reciprocal":
        return -_log_primal(graph, Expr(graph, node.arguments[0]))
    if node.operation == "power":
        return cast("Fraction | float", node.payload) * _log_primal(
            graph, Expr(graph, node.arguments[0])
        )
    return graph.stable_unary("log", root)


def _tensor_root(graph: Graph, root: Expr, bindings: dict[str, Node]) -> Node:
    """Rebind an AD scalar DAG into the existing scalar TensorIR backend."""
    values: dict[int, Node] = {}
    for identifier in graph.topological_order((root,)):
        node = graph.nodes[identifier]
        operands = [values[child] for child in node.arguments]
        if node.operation == "constant":
            value = constant(node.payload)
        elif node.operation == "variable":
            value = bindings[str(node.payload)]
        elif node.operation == "add":
            value = add(*operands)
        elif node.operation == "multiply":
            inverses = [
                index
                for index, child in enumerate(node.arguments)
                if graph.nodes[child].operation == "reciprocal"
            ]
            if len(node.arguments) == 2 and len(inverses) == 1:
                index = inverses[0]
                denominator = graph.nodes[node.arguments[index]].arguments[0]
                value = divide(operands[1 - index], values[denominator])
            else:
                value = operands[0]
                for operand in operands[1:]:
                    value = multiply(value, operand)
        elif node.operation == "reciprocal":
            value = divide(constant(1), operands[0])
        elif node.operation == "power":
            value = power(operands[0], node.payload)
        else:
            raise ValueError(f"unlowered nonlocal AD operation {node.operation}")
        values[identifier] = value
    return values[root.identifier]


def rvv10_pair_responses(
    first: Node,
    second: Node,
    first_ratio: Node,
    second_ratio: Node,
    radius: Node,
    first_kappa: Node,
    row_inverse: Node,
    phi: Node,
    *,
    preconditioned: bool,
) -> dict[str, Node]:
    """Serial pullbacks of the rVV10 representation and denominator primals.

    Inverse-kappa stretch keeps the preconditioned representation well scaled:
    its Gaussian-scale factor is s**(-3/2), and its first z factor is
    1+(z_i-1)*s. This is the raw kappa primal expressed at s=1, not an
    independently specified kappa slope. AD derives every local partial;
    the final products contract those serial scalar pullbacks in native order.
    """
    graph = Graph()
    left, right = graph.variable("first"), graph.variable("second")
    logarithm = _log_primal(graph, left * right * (left + right))
    first_partial = graph.differentiate(logarithm, left)
    virtual_value = graph.exponential(-logarithm)
    radial = graph.differentiate(
        virtual_value,
        graph.variable("direction"),
        {
            "first": graph.variable("first_ratio"),
            "second": graph.variable("second_ratio"),
        },
    )
    (radial,) = graph.replace_subexpressions(
        (radial,), {virtual_value: graph.variable("phi")}
    )
    bindings = {
        "first": first,
        "second": second,
        "first_ratio": first_ratio,
        "second_ratio": second_ratio,
        "radius": radius,
        "kappa": first_kappa,
        "row_inverse": row_inverse,
        "phi": phi,
    }
    factor = _tensor_root(graph, first_partial, bindings)
    radial_output = _tensor_root(graph, radial, bindings)

    omega, kappa, distance, energy, inverse = (
        graph.variable(name)
        for name in ("omega", "kappa", "radius", "phi", "row_inverse")
    )
    weighted_coordinate = -energy * omega * distance
    weighted_coordinate = (
        weighted_coordinate * inverse if preconditioned else weighted_coordinate / kappa
    )
    omega_weight = graph.differentiate(weighted_coordinate, omega)
    kappa_coordinate = energy * graph.stable_unary("log", kappa)
    kappa_weight = graph.differentiate(kappa_coordinate, kappa)
    if preconditioned:
        (kappa_weight,) = graph.replace_subexpressions(
            (kappa_weight,), {graph.reciprocal(kappa): inverse}
        )
    omega_weight = _tensor_root(graph, omega_weight, bindings)
    kappa_weight = _tensor_root(graph, kappa_weight, bindings)

    stretch = graph.variable("stretch")
    stretched_first = 1 + (left - 1) * stretch
    denominator = (
        stretch.pow(Fraction(-3, 2))
        * stretched_first
        * right
        * (stretched_first + right)
    )
    kappa_partial = graph.differentiate(_log_primal(graph, denominator), stretch)
    (kappa_partial,) = graph.replace_subexpressions(
        (kappa_partial,), {stretch: graph.constant(1)}
    )
    graph, (kappa_partial,) = graph.apply_algebra_form(
        (kappa_partial,), AlgebraForm.FACTORED_NARY
    )
    return {
        "dphi_domega": multiply(omega_weight, factor),
        "dphi_dkappa": multiply(
            kappa_weight, _tensor_root(graph, kappa_partial, bindings)
        ),
        "dphi_dr2": radial_output,
    }


def vv10_pair_responses(
    first: Node,
    second: Node,
    first_omega: Node,
    second_omega: Node,
    radius: Node,
    phi: Node,
    inverse_denominator: Node,
    *,
    rational: bool,
    shared_reciprocal: bool,
) -> dict[str, Node]:
    """Differentiate the ordered VV10 denominator, with qualified lowerings.

    The logarithmic form avoids squared denominator underflow in the raw
    fallback. Rational forms are admitted separately by the runtime; their
    cotangent is derived from the reciprocal primal and reuses either its
    evaluated reciprocal or its energy. Native admission is not broadened.
    """
    graph = Graph()
    left, right = graph.variable("first"), graph.variable("second")
    combined = left + right
    denominator = left * right * combined
    bindings = {
        "first": first,
        "second": second,
        "phi": phi,
        "first_omega": first_omega,
        "second_omega": second_omega,
        "radius": radius,
        "inverse_denominator": inverse_denominator,
    }
    if rational:
        argument = graph.variable("denominator")
        numerator = Fraction(-3, 2)
        value = graph.variable("phi")
        residual = value * argument - numerator
        derivative = -graph.differentiate(residual, argument) / graph.differentiate(
            residual, value
        )
        replacement = (
            graph.variable("inverse_denominator")
            if shared_reciprocal
            else graph.variable("phi") / numerator
        )
        (derivative,) = graph.replace_subexpressions(
            (derivative,), {graph.reciprocal(argument): replacement}
        )
        graph, (derivative,) = graph.apply_algebra_form(
            (derivative,), AlgebraForm.FACTORED_NARY
        )
        cotangent = _tensor_root(graph, derivative, bindings)
        derivative_graph = Graph()
        left, right = (
            derivative_graph.variable("first"),
            derivative_graph.variable("second"),
        )
        combined = left + right
        denominator = left * right * combined
        partials = tuple(
            derivative_graph.differentiate(denominator, item) for item in (left, right)
        )
        # Retain the already evaluated primal sum before associative factoring.
        # Expanding it creates an unnecessary FP64 multiply-by-two in each
        # partial and discards CSE with the energy denominator. Replacement is
        # after AD, so every derivative of that sum is still generated normally.
        partials = derivative_graph.replace_subexpressions(
            partials, {combined: derivative_graph.variable("combined")}
        )
        bindings["combined"] = add(first, second)
        derivative_graph, partials = derivative_graph.apply_algebra_form(
            partials, AlgebraForm.FACTORED_NARY
        )
        first_partial, second_partial = (
            multiply(cotangent, _tensor_root(derivative_graph, item, bindings))
            for item in partials
        )
        return {
            "dphi_domega": multiply(first_partial, radius),
            "dphi_dkappa": first_partial,
            "dphi_domega_j": multiply(second_partial, radius),
            "dphi_dkappa_j": second_partial,
            "dphi_dr2": add(
                multiply(first_omega, first_partial),
                multiply(second_omega, second_partial),
            ),
        }
    logarithmic = _log_primal(graph, denominator)
    virtual_primal = graph.exponential(-logarithmic)
    partials = tuple(
        graph.differentiate(virtual_primal, item) for item in (left, right)
    )
    radial = graph.differentiate(
        virtual_primal,
        graph.variable("direction"),
        {
            "first": graph.variable("first_omega"),
            "second": graph.variable("second_omega"),
        },
    )
    partials = graph.replace_subexpressions(
        (*partials, radial),
        {virtual_primal: graph.variable("phi")},
    )
    first_partial, second_partial, radial = (
        _tensor_root(graph, item, bindings) for item in partials
    )
    return {
        "dphi_domega": multiply(first_partial, radius),
        "dphi_dkappa": first_partial,
        "dphi_domega_j": multiply(second_partial, radius),
        "dphi_dkappa_j": second_partial,
        "dphi_dr2": radial,
    }
