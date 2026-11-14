"""Bounded host D3(BJ) columns from the incumbent dispersion scalar Graphs.

Native code schedules coordination, normalized references, pair energy and the
reverse coordination scatter. All first and right-mixed arithmetic stays here;
the pair contraction includes both atoms' full normalized-reference response.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.expr import Expr, Graph
from generativeqc_compiler.integral.graph_composition import import_roots

from .dispersion_scalar import dispersion_scalar_program

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

D3_COLUMN_ARRAYS = (
    "d3_coordination_column",
    "d3_reference_column",
    "d3_pair_column",
    "d3_scatter_column",
)


@cache
def d3_column_program(name: str) -> tuple:
    """Return canonical local roots; the native owner admits only uncut BJ/ATM-off.

    Coordination layout is value/right/gradient(3)/mixed(3). References are
    seven lanes each of weight/right/dweight_dCN/right_dweight_dCN, followed by
    the unscaled normalization for the incumbent underflow-domain gate. Pair layout
    is energy/right/gradient(3)/mixed(3)/CN adjoints(2)/right CN adjoints(2).
    Scatter returns three gradients and their three right tangents.
    """
    if name not in D3_COLUMN_ARRAYS:
        raise ValueError("unknown D3 column leaf")
    graph = Graph()
    direction = graph.variable("direction")

    def scalar(kind: str, **bindings: Expr) -> Expr:
        source, _, roots = dispersion_scalar_program(kind)
        return import_roots(graph, source, roots[:1], bindings)[0]

    def right(roots: Sequence[Expr], tangents: Mapping[str, Expr]) -> tuple[Expr, ...]:
        return tuple(graph.differentiate(root, direction, tangents) for root in roots)

    displacement = tuple(graph.variable(f"r{axis}") for axis in range(3))
    tangents = {f"r{axis}": graph.variable(f"dr{axis}") for axis in range(3)}
    squared = graph.sum(value * value for value in displacement)
    if name == "d3_coordination_column":
        value = scalar(
            "d3_coordination",
            distance=squared.pow(0.5),
            radius=graph.variable("radius"),
        )
        gradient = tuple(graph.differentiate(value, leaf) for leaf in displacement)
        roots = (
            value,
            *right((value,), tangents),
            *gradient,
            *right(gradient, tangents),
        )
    elif name == "d3_reference_column":
        coordination = graph.variable("cn")
        source, _, primal = dispersion_scalar_program("gaussian")
        exponent = Expr(source, source.node(primal[0]).arguments[0])
        exponents = tuple(
            import_roots(
                graph,
                source,
                (exponent,),
                {
                    "coordination": coordination,
                    "reference": graph.variable(f"ref{slot}"),
                    "factor": graph.constant(4),
                },
            )[0]
            for slot in range(7)
        )
        masks = tuple(graph.variable(f"mask{slot}") for slot in range(7))
        shift = exponents[0]
        for slot in range(1, 7):
            shift = graph.select_le(
                masks[slot],
                0,
                shift,
                graph.select_le(shift, exponents[slot], exponents[slot], shift),
            )
        gaussian = tuple(
            graph.select_le(mask, 0, 0, graph.exponential(exponent - shift))
            for mask, exponent in zip(masks, exponents, strict=True)
        )
        norm = graph.sum(gaussian)
        weights = tuple(
            scalar("normalize", value=value, norm=norm) for value in gaussian
        )
        partials = tuple(
            graph.differentiate(weight, coordination) for weight in weights
        )
        motion = {"cn": graph.variable("dcn")}
        unscaled_norm = graph.sum(
            mask * graph.exponential(exponent)
            for mask, exponent in zip(masks, exponents, strict=True)
        )
        roots = (
            *weights,
            *right(weights, motion),
            *partials,
            *right(partials, motion),
            unscaled_norm,
        )
    elif name == "d3_pair_column":
        weights = {
            (atom, slot): graph.variable(f"w{atom}{slot}")
            for atom in range(2)
            for slot in range(7)
        }
        coefficient = graph.sum(
            graph.variable(f"c{first}{second}") * weights[0, first] * weights[1, second]
            for first in range(7)
            for second in range(7)
        )
        energy = -coefficient * scalar(
            "bj_damping",
            distance_squared=squared,
            rr=graph.variable("rr"),
            radius=graph.variable("radius"),
            s6=graph.variable("s6"),
            s8=graph.variable("s8"),
        )
        gradient = tuple(graph.differentiate(energy, leaf) for leaf in displacement)
        adjoints = tuple(
            graph.differentiate(
                energy,
                graph.variable(f"cn{atom}"),
                {
                    f"w{atom}{slot}": graph.variable(f"p{atom}{slot}")
                    for slot in range(7)
                },
            )
            for atom in range(2)
        )
        tangents.update(
            {
                f"{kind}{atom}{slot}": graph.variable(f"d{kind}{atom}{slot}")
                for kind in ("w", "p")
                for atom in range(2)
                for slot in range(7)
            }
        )
        roots = (
            energy,
            *right((energy,), tangents),
            *gradient,
            *right(gradient, tangents),
            *adjoints,
            *right(adjoints, tangents),
        )
    else:
        gradient = tuple(
            graph.variable("bar") * graph.variable(f"g{axis}") for axis in range(3)
        )
        motion = {
            "bar": graph.variable("dbar"),
            **{f"g{axis}": graph.variable(f"dg{axis}") for axis in range(3)},
        }
        roots = (*gradient, *right(gradient, motion))
    return graph, roots
