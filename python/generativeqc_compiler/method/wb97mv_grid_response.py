"""Build-time WB97M-V fixed-grid response, using the existing AD authorities.

The native owner supplies traversal, immutable quadrature and resource gates.
This module composes the audited XC work policy, scalar coefficient JVP and
AO-feature VJP; it does not introduce another set of derivative equations.
"""

from __future__ import annotations

from functools import cache

from generativeqc_compiler.integral.expr import Expr, Graph, Node
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_grid_second import feature_programs
from generativeqc_compiler.tensor import Program, transpose_program
from generativeqc_compiler.xc.coefficients import coefficient_program
from generativeqc_compiler.xc.program import build_program
from generativeqc_compiler.xc.wb97mv_maple import (
    DENSITY_THRESHOLD,
    SIGMA_THRESHOLD,
    TAU_THRESHOLD,
)


def _import_roots(
    target: Graph, source: Graph, roots: tuple, bindings: dict[str, Expr]
) -> tuple:
    """Rehome coefficient DAG nodes verbatim, with simultaneous leaf binding.

    Interner reuse preserves all existing operations and exact constants. This
    is graph composition, not a second implementation of the scalar algebra.
    """
    imported = {}
    for identifier in source.topological_order(roots):
        node = source.nodes[identifier]
        imported[identifier] = (
            bindings[str(node.payload)]
            if node.operation == "variable"
            else target._intern(
                Node(
                    node.operation,
                    tuple(imported[child].identifier for child in node.arguments),
                    node.payload,
                )
            )
        )
    return tuple(imported[root.identifier] for root in roots)


@cache
def _point_terms(spins: int) -> tuple:
    """Return weighted Cartesian feature cotangents and their density JVP.

    Fields are [spin,rho/gx/gy/gz/tau,point]. RKS stores total density; UKS
    stores physical alpha/beta densities. Floors and the cross-sigma bound are
    the existing stationary-grid/Libxc work policy, outside differentiation.
    In particular, only the feature-Hessian seed is zeroed below a floor;
    coefficient pullbacks still see the raw spatial gradients. Equality uses
    the same fixed-branch conventions as the incumbent molecular consumer.
    """
    if type(spins) is not int or spins not in (1, 2):
        raise ValueError("WB97M-V grid requires one or two physical spin blocks")
    spin = "polarized" if spins == 2 else "unpolarized"
    spec = resolve_method("WB97M-V", spin=spin).primitives[0].functional
    program = build_program(spec, order=2)
    graph = program.graph
    fields = [graph.variable(f"f{index}") for index in range(5 * spins)]
    moving = [graph.variable(f"d{index}") for index in range(5 * spins)]
    densities = [fields[5 * index] for index in range(spins)]
    tau = [fields[5 * index + 4] for index in range(spins)]
    gradients = [fields[5 * index + 1 : 5 * index + 4] for index in range(spins)]
    dg = [moving[5 * index + 1 : 5 * index + 4] for index in range(spins)]
    pairs = ((0, 0), (0, 1), (1, 1)) if spins == 2 else ((0, 0),)
    sigma = [
        graph.sum(
            first * second
            for first, second in zip(gradients[a], gradients[b], strict=True)
        )
        for a, b in pairs
    ]
    dsigma = [
        graph.sum(
            gradients[a][axis] * dg[b][axis] + dg[a][axis] * gradients[b][axis]
            for axis in range(3)
        )
        for a, b in pairs
    ]
    raw = [*densities, *sigma, *tau]
    draw = [
        *(moving[5 * index] for index in range(spins)),
        *dsigma,
        *(moving[5 * index + 4] for index in range(spins)),
    ]
    floors = (
        (
            DENSITY_THRESHOLD,
            DENSITY_THRESHOLD,
            SIGMA_THRESHOLD**2,
            None,
            SIGMA_THRESHOLD**2,
            TAU_THRESHOLD,
            TAU_THRESHOLD,
        )
        if spins == 2
        else (2 * DENSITY_THRESHOLD, 4 * SIGMA_THRESHOLD**2, 2 * TAU_THRESHOLD)
    )
    work, dwork = [], []
    for value, delta, floor in zip(raw, draw, floors, strict=True):
        work.append(
            value if floor is None else graph.select_le(value, floor, floor, value)
        )
        dwork.append(
            delta if floor is None else graph.select_le(value, floor, 0, delta)
        )
    if spins == 2:
        bound = (work[2] + work[4]) / 2
        clipped = graph.select_le(work[3], -bound, -bound, work[3])
        work[3] = graph.select_le(clipped, bound, clipped, bound)

    roots = dict(zip(program.outputs, program.roots, strict=True))
    seed = graph.variable("density_direction")
    leaves = dict(zip(spec.features, dwork, strict=True))
    substitutions = dict(
        zip((graph.variable(name) for name in spec.features), work, strict=True)
    )
    potential = tuple(roots[(index,)] for index in range(len(spec.features)))
    dpotential = tuple(graph.differentiate(value, seed, leaves) for value in potential)
    potential = graph.replace_subexpressions(potential, substitutions)
    dpotential = graph.replace_subexpressions(dpotential, substitutions)
    bindings = {f"v{index}": value for index, value in enumerate(potential)}
    bindings.update({f"dv{index}": value for index, value in enumerate(dpotential)})
    for spin_index in range(spins):
        for axis in range(3):
            bindings[f"g{spin_index}_{axis}"] = gradients[spin_index][axis]
            bindings[f"dg{spin_index}_{axis}"] = dg[spin_index][axis]
    programs = [
        coefficient_program(spin, "mgga", response=flag) for flag in (False, True)
    ]
    weight = graph.variable("weight")
    cartesian = tuple(
        value * (2 if index % 5 == 4 else 1)
        for coefficients in programs
        for index, value in enumerate(
            _import_roots(
                graph, coefficients.graph, tuple(coefficients.roots), bindings
            )
        )
    )
    outputs = tuple(
        value * weight * (2 if index % 5 == 4 else 1)
        for coefficients in programs
        for index, value in enumerate(
            _import_roots(
                graph, coefficients.graph, tuple(coefficients.roots), bindings
            )
        )
    )
    threshold = graph.constant(DENSITY_THRESHOLD)
    total = graph.sum(densities)
    masked = tuple(graph.select_le(threshold, total, value, 0) for value in outputs)
    energy = graph.replace_subexpressions((roots[()],), substitutions)[0]
    energy_direction = graph.sum(
        value * delta for value, delta in zip(potential, dwork, strict=True)
    )
    return graph, masked, cartesian, energy, energy_direction, total


@cache
def point_program(spins: int) -> tuple:
    """Fixed-grid compatibility roots from the shared physical work policy."""
    graph, roots, *_ = _point_terms(spins)
    return graph, roots


@cache
def point_field_partials(spins: int) -> tuple:
    """Energy and weighted field partials/JVP including the moving measure.

    Outputs are energy, [rho/gx/gy/gz/tau for each spin,weight] partials,
    then their direction. The incumbent fixed-branch floor policy is retained:
    only functional-Hessian seeds and energy-density tangents use floored
    feature directions; Cartesian coefficient pullbacks retain raw gradients.
    Position partials vanish for this semilocal component.
    """
    graph, _, cartesian, energy, energy_direction, total = _point_terms(spins)
    count = 5 * spins
    weight, delta_weight = graph.variable("weight"), graph.variable("dweight")
    outputs = (
        weight * energy,
        *(weight * value for value in cartesian[:count]),
        energy,
        *(
            weight * delta + delta_weight * value
            for value, delta in zip(cartesian[:count], cartesian[count:], strict=True)
        ),
        energy_direction,
    )
    threshold = graph.constant(DENSITY_THRESHOLD)
    return graph, tuple(
        graph.select_le(threshold, total, value, 0) for value in outputs
    )


def grid_maps(points: int = 3, nao: int = 2) -> dict[str, Program]:
    """Reuse the AO-feature primal and derive its density adjoint with TensorIR.

    Feature extraction is linear in D for fixed AOs. Thus the same primal map
    evaluates a signed/indefinite density direction without a PSD factorization.
    The reverse map assembles physical feature cotangents, including tau's
    one-half, and needs neither a D input nor an explicit point-by-AO-pair panel.
    """
    primal, _ = feature_programs(points, nao)
    reverse = transpose_program(primal, tuple(primal.outputs), inputs=("density",))
    return {
        "features": primal,
        "potential": Program({"potential": reverse.program.outputs["bar_density"]}),
    }
