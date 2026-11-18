"""Backend-neutral density-incidence binding for ordered quartet objectives."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generativeqc_compiler.tensor import Node, Program


def density_incidence(program: Program, *, exchange: bool) -> Node:
    """Bind two density leaves to Coulomb/exchange incidences without changing AD."""
    prefix = "exchange" if exchange else "coulomb"
    mapped = {}
    for node in program.nodes:
        attrs = dict(node.attrs)
        if node.op == "input" and attrs["name"] in ("density_left", "density_right"):
            attrs["name"] = attrs["name"].replace("density", prefix)
        mapped[node] = replace(
            node,
            inputs=tuple(mapped[value] for value in node.inputs),
            attributes=tuple(attrs.items()),
        )
    return mapped[program.outputs["energy"]]
