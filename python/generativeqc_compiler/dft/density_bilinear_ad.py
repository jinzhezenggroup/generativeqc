"""AD-derived sparse contractions of real symmetric-density AO bilinears.

The primal is a product of two independent AO Taylor polynomials. Polarizing
that product and contracting with any real symmetric D is valid even when D
is indefinite: no orbital factorization or positive-density assumption enters.
Only linear contractions and symmetry-based reuse are left to runtime owners.
"""

from collections.abc import Callable
from fractions import Fraction
from functools import cache
from math import factorial
from typing import Any

from generativeqc_compiler.integral.expr import AlgebraForm, Expr, Graph
from generativeqc_compiler.integral.scalar_python import (
    polynomial_terms,
    scalar_callable,
)

MultiIndex = tuple[int, int, int]
Field = tuple[MultiIndex, bool]
Term = tuple[Fraction, Field, Field]


@cache
def density_bilinear_terms(directions: tuple[tuple[str, int], ...]) -> tuple[Term, ...]:
    """Return sparse AO-pair terms through two point/center derivatives.

    A field's boolean requests that leg's atom-owned AO columns. Point motion
    shifts both legs positively; nuclear-center motion shifts only owned legs
    negatively. Both are coordinates of the primal Taylor argument, not
    separately supplied spatial or nuclear derivative rules. Symmetric D
    permits exchanging the complete (jet, ownership) legs after AD.
    """
    if len(directions) > 2 or any(
        kind not in ("point", "center") or type(axis) is not int or not 0 <= axis < 3
        for kind, axis in directions
    ):
        raise ValueError(
            "density bilinear supports through-order-two Cartesian directions"
        )
    graph = Graph()
    points = tuple(graph.variable(f"point{axis}") for axis in range(3))
    centers = tuple(graph.variable(f"center{axis}") for axis in range(3))
    indices = tuple(
        (first, degree - first - third, third)
        for degree in range(len(directions) + 1)
        for first in range(degree, -1, -1)
        for third in range(degree - first + 1)
    )
    fields: dict[str, tuple[str, MultiIndex]] = {}
    polynomials = []
    for leg in ("left", "right"):
        ownership = graph.variable(f"owned_{leg}")
        displacement = tuple(
            point - ownership * center
            for point, center in zip(points, centers, strict=True)
        )
        terms = []
        for index in indices:
            name = leg + "_" + "".join(map(str, index))
            fields[name] = leg, index
            term = graph.variable(name)
            for coordinate, power in zip(displacement, index, strict=True):
                for _ in range(power):
                    term *= coordinate
                term *= Fraction(1, factorial(power))
            terms.append(term)
        polynomials.append(graph.sum(terms))
    response = polynomials[0] * polynomials[1]
    for kind, axis in directions:
        response = graph.differentiate(
            response, points[axis] if kind == "point" else centers[axis]
        )
    (response,) = graph.replace_subexpressions(
        (response,),
        {coordinate: graph.constant(0) for coordinate in (*points, *centers)},
    )

    contractions: dict[tuple[Field, Field], Fraction] = {}
    for coefficient, monomial in polynomial_terms(graph, response):
        leg_fields = {}
        for name in monomial:
            if name in fields:
                leg, index = fields[name]
                if leg in leg_fields:
                    raise ValueError("density primal is no longer bilinear")
                leg_fields[leg] = index, f"owned_{leg}" in monomial
            elif name not in ("owned_left", "owned_right"):
                raise ValueError("density response retained a Taylor coordinate")
        if set(leg_fields) != {"left", "right"}:
            raise ValueError("density response lost an AO leg")
        first, second = sorted(leg_fields.values())
        pair = first, second
        contractions[pair] = contractions.get(pair, Fraction(0)) + coefficient
    return tuple(
        (coefficient, *pair)
        for pair, coefficient in sorted(contractions.items())
        if coefficient
    )


@cache
def density_axis_schedule(axis: int) -> tuple[tuple[Fraction, int], ...]:
    """Orient first-order AD terms around the retained value D*AO product."""
    schedule = []
    for coefficient, first, second in density_bilinear_terms((("point", axis),)):
        jets = (first[0], second[0])
        if sorted(map(sum, jets)) != [0, 1] or first[1] or second[1]:
            raise ValueError(
                "first density response no longer fits retained value work"
            )
        derivative = next(index for index in jets if sum(index))
        component = derivative.index(1)
        schedule.append((coefficient, component))
    return tuple(schedule)


def density_axis_roots(graph: Graph, axis: int) -> Expr:
    """Bind a generated first spatial derivative to native retained D*AO work."""
    terms = [
        coefficient
        * graph.variable(f"derivative[{component}]")
        * graph.variable("work[0]")
        for coefficient, component in density_axis_schedule(axis)
    ]
    return graph.sum(terms)


@cache
def density_gradient_callable() -> Callable:
    """Lower all three first responses into one caller-owned CPU output panel."""
    import numpy as np

    lines = ["def evaluate(jets, work, output):"]
    for axis in range(3):
        terms = [
            f"{float(coefficient)!r} * _sum(jets[{component + 1}] * work, axis=1)"
            for coefficient, component in density_axis_schedule(axis)
        ]
        lines.append(f"    output[:, {axis}] = " + " + ".join(terms))
    namespace: dict[str, Any] = {"_sum": np.sum, "__builtins__": {}}
    exec(compile("\n".join(lines), "<generated-density-gradient>", "exec"), namespace)  # noqa: S102 -- validated integer AD schedule
    return namespace["evaluate"]


@cache
def sigma_bilinear_callable(*, mixed: bool) -> Callable:
    """Differentiate the Cartesian summand of the sigma dot-product primal."""
    import numpy as np

    graph = Graph()
    first, second = graph.variable("first"), graph.variable("second")
    primal = first * second
    left = graph.differentiate(
        primal,
        graph.variable("left"),
        {
            "first": graph.variable("left_first"),
            "second": graph.variable("left_second"),
        },
    )
    arguments = ("first", "second", "left_first", "left_second")
    if mixed:
        root = graph.differentiate(
            left,
            graph.variable("right"),
            {
                "first": graph.variable("right_first"),
                "second": graph.variable("right_second"),
                "left_first": graph.variable("mixed_first"),
                "left_second": graph.variable("mixed_second"),
            },
        )
        arguments += ("right_first", "right_second", "mixed_first", "mixed_second")
    else:
        root = left
    return scalar_callable(graph, (root,), arguments, np)


@cache
def sigma_pullback_callable() -> Callable:
    """Generate the unpolarized sigma pullback on a full Cartesian panel."""
    import numpy as np

    graph = Graph()
    cotangent, gradient = graph.variable("cotangent"), graph.variable("gradient")
    root = graph.differentiate(cotangent * gradient * gradient, gradient)
    graph, roots = graph.apply_algebra_form((root,), AlgebraForm.FACTORED_NARY)
    return scalar_callable(graph, roots, ("cotangent", "gradient"), np)


@cache
def density_response_callable(
    directions: tuple[tuple[str, int], ...], products: tuple[Field, ...]
) -> Callable:
    """Compile the sparse CPU schedule once, without per-tile term dispatch.

    Runtime retains BLAS products and storage. Generated code contains only
    validated integer jet indices, ownership booleans, exact AD coefficients
    and fixed NumPy reductions. Equal coefficients stay outside their sum,
    preserving cancellation before scaling at the finite overflow boundary.
    """
    import numpy as np

    if len(products) > 20:
        raise ValueError("unbounded density product fields")
    for index, owned in products:
        if (
            len(index) != 3
            or any(type(value) is not int or value < 0 for value in index)
            or sum(index) > 2
            or type(owned) is not bool
        ):
            raise ValueError("invalid density product field")
    groups: dict[Fraction, list[str]] = {}
    for coefficient, first_field, second_field in density_bilinear_terms(directions):
        first, second = first_field, second_field
        if first in products and (not first[1] or second[1]):
            first, second = second, first
        if second not in products:
            raise ValueError("density response lacks its retained D*AO work")
        value, work = f"jets[{first[0]!r}]", f"products[{second!r}]"
        if first[1]:
            value, work = f"{value}[:, owned]", f"{work}[:, owned]"
        groups.setdefault(coefficient, []).append(f"_sum({value} * {work}, axis=1)")
    if not groups:
        raise ValueError("empty density response")
    expressions = [
        f"{float(coefficient)!r} * (" + " + ".join(terms) + ")"
        for coefficient, terms in groups.items()
    ]
    source = "def evaluate(jets, products, owned):\n    return " + " + ".join(
        expressions
    )
    namespace: dict[str, Any] = {"_sum": np.sum, "__builtins__": {}}
    exec(compile(source, "<generated-density-contraction>", "exec"), namespace)  # noqa: S102 -- validated numeric AD schedule, never external source
    return namespace["evaluate"]
