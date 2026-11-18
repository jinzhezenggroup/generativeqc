"""Build-only mixed geometry jets for native moving-grid Hessian sources.

Local scalar derivatives reuse the canonical Becke/norm/ratio AD programs.
Product jets come from AD of multiplication, not a handwritten second product
rule. AO motion differentiates a Taylor polynomial in the incumbent ordinary
spatial-jet basis; its coefficients are supplied by the existing Gaussian AD.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache
from math import factorial
from typing import TYPE_CHECKING

from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.integral.expr import Expr, Graph
from generativeqc_compiler.integral.graph_composition import import_roots
from generativeqc_compiler.xc.coefficients import coefficient_program
from generativeqc_compiler.xc.grid_response_ir import grid_mixed_response_program
from generativeqc_compiler.xc.program import output_set
from generativeqc_compiler.xc.semilocal_codegen import build_roots

if TYPE_CHECKING:
    from generativeqc_compiler.xc.spec import FunctionalSpec


def _mixed(graph: Graph, root: Expr, names: tuple[str, ...]) -> tuple[Expr, ...]:
    """Compose independent left/right tangents and upstream mixed leaves."""
    left = graph.differentiate(
        root,
        graph.variable("left_direction"),
        {name: graph.variable(f"l{name}") for name in names},
    )
    right = graph.differentiate(
        root,
        graph.variable("right_direction"),
        {name: graph.variable(f"r{name}") for name in names},
    )
    mixed = graph.differentiate(
        left,
        graph.variable("right_direction"),
        {
            **{name: graph.variable(f"r{name}") for name in names},
            **{f"l{name}": graph.variable(f"lr{name}") for name in names},
        },
    )
    return root, left, right, mixed


@cache
def local_program(kind: str) -> tuple[Graph, tuple[Expr, ...], tuple[str, ...]]:
    """Scalar jet arithmetic; binary rescaling remains native scheduling only."""
    if kind == "treutler_adjust":
        program = grid_mixed_response_program(kind)
        return program.graph, program.roots, ("mu", "adjustment")
    if kind in ("norm", "ratio") or kind.startswith("becke_"):
        primitive = kind.split("_")[0]
        iterations = int(kind.split("_")[1]) if primitive == "becke" else 3
        program = grid_mixed_response_program(primitive, iterations)
        names = (
            ("x", "y", "z")
            if primitive == "norm"
            else (("a", "b") if primitive == "ratio" else ("mu",))
        )
        return program.graph, program.roots, names
    graph = Graph()
    names = ("a",) if kind == "complement" else ("a", "b")
    first = graph.variable("a")
    if kind == "multiply":
        root = first * graph.variable("b")
    elif kind == "add":
        root = first + graph.variable("b")
    elif kind == "subtract":
        root = first - graph.variable("b")
    elif kind == "complement":
        root = 1 - first
    else:
        raise ValueError("unknown mixed geometry jet primitive")
    return graph, _mixed(graph, root, names), names


@cache
def ao_motion_program() -> tuple[Graph, tuple[Expr, ...]]:
    """Return left/right/mixed order-one AO jets from ordinary order-three jets.

    Spatial Taylor variables are evaluated at zero by the emitter. Expanding
    monomials as multiplication avoids singular power-rule intermediates at
    zero. The xy coefficient occurs once, with the correct mixed multiplicity
    supplied by AD rather than manually attached to the stored spatial jet.
    """
    graph = Graph()
    coordinates = [graph.variable(f"x{axis}") for axis in range(3)]
    lookup = {index: offset for offset, index in enumerate(jet_indices(3))}
    roots = []
    for base in jet_indices(1):
        terms = []
        for moving in jet_indices(2):
            shifted = tuple(
                first + second for first, second in zip(base, moving, strict=True)
            )
            term = graph.variable(f"jet{lookup[shifted]}")
            denominator = 1
            for axis, degree in enumerate(moving):
                denominator *= factorial(degree)
                for _ in range(degree):
                    term *= coordinates[axis]
            terms.append(term * Fraction(1, denominator))
        polynomial = graph.sum(terms)
        first = graph.differentiate(
            polynomial,
            graph.variable("left"),
            {f"x{axis}": graph.variable(f"left{axis}") for axis in range(3)},
        )
        second = graph.differentiate(
            polynomial,
            graph.variable("right"),
            {f"x{axis}": graph.variable(f"right{axis}") for axis in range(3)},
        )
        mixed = graph.differentiate(
            first,
            graph.variable("right"),
            {f"x{axis}": graph.variable(f"right{axis}") for axis in range(3)},
        )
        roots.extend((first, second, mixed))
    return graph, tuple(roots)


@cache
def ao_first_motion_program() -> tuple[Graph, tuple[Expr, ...]]:
    """Specialize canonical Taylor-AD first motion at its exact expansion origin.

    Native readers validate finite spatial jets before this polynomial-only
    specialization. It removes zero higher-order terms and therefore needs
    ordinary spatial jets through order two, without changing mixed AO roots.
    """
    graph, roots = ao_motion_program()
    selected = graph.replace_subexpressions(
        roots[::3], {graph.variable(f"x{axis}"): graph.constant(0) for axis in range(3)}
    )
    return graph.canonicalize_associative(selected)


@cache
def gga_energy_mixed_program(
    functional: FunctionalSpec,
) -> tuple[Graph, tuple[Expr, ...]]:
    """Weighted frozen-density energy and independent geometry mixed partials.

    Production energy/ingredient gradients/Hessians remain the scalar authority.
    Only the Cartesian ingredient and quadrature product chain rules are added;
    density relaxation is a separate stationary-Hessian assembly contribution.
    """
    if functional.ingredients != ("rho", "sigma"):
        raise ValueError("mixed GGA energy requires exactly rho/sigma ingredients")
    graph, roots, _ = build_roots(
        functional, output_set(functional, 2), production=True
    )
    spins = 2 if functional.spin == "polarized" else 1
    names = tuple(f"f{index}" for index in range(4 * spins))
    fields = [graph.variable(name) for name in names]
    pairs = ((0, 0), (0, 1), (1, 1)) if spins == 2 else ((0, 0),)
    ingredients = [fields[4 * spin] for spin in range(spins)]
    ingredients.extend(
        graph.sum(
            fields[4 * first + axis + 1] * fields[4 * second + axis + 1]
            for axis in range(3)
        )
        for first, second in pairs
    )
    ingredients.extend(graph.constant(0) for _ in range(spins))
    physical = graph.replace_subexpressions(
        roots,
        {
            graph.variable(name): value
            for name, value in zip(functional.features, ingredients, strict=True)
        },
    )
    derivatives = dict(zip(output_set(functional, 2), physical, strict=True))
    moving = [_mixed(graph, ingredient, names) for ingredient in ingredients]
    first = graph.sum(
        derivatives[(index,)] * values[1] for index, values in enumerate(moving)
    )
    second = graph.sum(
        derivatives[(index,)] * values[2] for index, values in enumerate(moving)
    )
    mixed = graph.sum(
        derivatives[(index,)] * values[3] for index, values in enumerate(moving)
    ) + graph.sum(
        derivatives[tuple(sorted((left, right)))] * moving[left][1] * moving[right][2]
        for left in range(len(ingredients))
        for right in range(len(ingredients))
    )
    product, product_roots, _ = local_program("multiply")
    weighted = import_roots(
        graph,
        product,
        product_roots,
        {
            "a": derivatives[()],
            "la": first,
            "ra": second,
            "lra": mixed,
            "b": graph.variable("weight"),
            "lb": graph.variable("lweight"),
            "rb": graph.variable("rweight"),
            "lrb": graph.variable("lrweight"),
        },
    )
    return graph, weighted


@cache
def gga_field_response_program(
    functional: FunctionalSpec,
) -> tuple[Graph, tuple[Expr, ...]]:
    """Compose production XC partials and canonical Cartesian coefficient JVP.

    Roots are weighted primal and directional rho/gradient cotangents. Physical
    XC gradients/Hessians come unchanged from build_roots, including lazy vacuum
    and endpoint policies; no second functional or independent XC AD is built.
    """
    if functional.ingredients != ("rho", "sigma"):
        raise ValueError("moving GGA fields require exactly rho/sigma ingredients")
    graph, roots, _ = build_roots(
        functional, output_set(functional, 2), production=True
    )
    spins = 2 if functional.spin == "polarized" else 1
    fields = [graph.variable(f"f{index}") for index in range(4 * spins)]
    moving = {f"f{index}": graph.variable(f"d{index}") for index in range(4 * spins)}
    pairs = ((0, 0), (0, 1), (1, 1)) if spins == 2 else ((0, 0),)
    ingredients = [fields[4 * spin] for spin in range(spins)]
    ingredients.extend(
        graph.sum(
            fields[4 * first + axis + 1] * fields[4 * second + axis + 1]
            for axis in range(3)
        )
        for first, second in pairs
    )
    ingredients.extend(graph.constant(0) for _ in range(spins))
    replacements = {
        graph.variable(name): value
        for name, value in zip(functional.features, ingredients, strict=True)
    }
    physical = graph.replace_subexpressions(roots, replacements)
    derivatives = dict(zip(output_set(functional, 2), physical, strict=True))
    ingredient_motion = [
        graph.differentiate(value, graph.variable("direction"), moving)
        for value in ingredients
    ]
    potential = [derivatives[(index,)] for index in range(len(ingredients))]
    potential_motion = [
        graph.sum(
            derivatives[tuple(sorted((index, other)))] * value
            for other, value in enumerate(ingredient_motion)
        )
        for index in range(len(ingredients))
    ]
    bindings = {f"v{index}": value for index, value in enumerate(potential)}
    bindings.update(
        {f"dv{index}": value for index, value in enumerate(potential_motion)}
    )
    for spin in range(spins):
        for axis in range(3):
            bindings[f"g{spin}_{axis}"] = fields[4 * spin + axis + 1]
            bindings[f"dg{spin}_{axis}"] = moving[f"f{4 * spin + axis + 1}"]
    primal = coefficient_program(functional.spin, "gga", kinetic=False)
    tangent = coefficient_program(functional.spin, "gga", kinetic=False, response=True)
    coefficients = import_roots(graph, primal.graph, tuple(primal.roots), bindings)
    coefficient_motion = import_roots(
        graph, tangent.graph, tuple(tangent.roots), bindings
    )
    weight, delta_weight = graph.variable("weight"), graph.variable("dweight")
    return graph, tuple(weight * value for value in coefficients) + tuple(
        weight * delta + delta_weight * value
        for value, delta in zip(coefficients, coefficient_motion, strict=True)
    )
