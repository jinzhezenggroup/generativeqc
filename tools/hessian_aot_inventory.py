"""Build-time packaging of compiler-owned AD leaves for a native Hessian owner.

These programs preserve the existing graph roots and raw input domain. They do
not perform molecular-grid screening, partition product reductions, CPKS, DF
metric solves or stationary assembly. Those operations still need a native
owner before this package can advertise an end-to-end Hessian capability.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from generativeqc_compiler.integral.array_graph_cuda import emit_array_graph
from generativeqc_compiler.integral.ecp_grid import harmonic_roots, radial_map_roots
from generativeqc_compiler.integral.ecp_projector import radial_roots
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    _nuclear_pair_program,
    _pair_program,
)
from generativeqc_compiler.xc.coefficients import coefficient_program
from generativeqc_compiler.xc.grid_response_ir import (
    grid_mixed_response_program,
    grid_response_graph_identity,
)
from generativeqc_compiler.xc.program import build_program

SCALAR_PROGRAMS = (
    "coefficients_polarized",
    "coefficients_unpolarized",
    "coefficient_response_polarized",
    "coefficient_response_unpolarized",
    "ecp_harmonics",
    "ecp_radial_0",
    "ecp_radial_1",
    "ecp_radial_2",
    "ecp_radial_3",
    "ecp_radial_4",
    "ecp_radial_map",
    "grid_mixed_becke",
    "grid_mixed_log",
    "grid_mixed_norm",
    "grid_mixed_ratio",
    "nuclear_pair_hvp",
    "vv10_pair",
    "xc_polarized",
    "xc_unpolarized",
)


@cache
def scalar_graph(name: str) -> tuple[Any, tuple]:
    """Reuse authoritative energy/AD roots without a parallel scientific algebra."""
    if name not in SCALAR_PROGRAMS:
        raise ValueError("unknown Hessian AOT scalar program")
    if name.startswith("xc_"):
        method = resolve_method("WB97M-V", spin=name.removeprefix("xc_"))
        program = build_program(method.primitives[0].functional, order=2)
        return program.graph, program.roots
    if name.startswith(("coefficients_", "coefficient_response_")):
        spin = "unpolarized" if name.endswith("_unpolarized") else "polarized"
        program = coefficient_program(
            spin, "mgga", response=name.startswith("coefficient_response_")
        )
        return program.graph, tuple(program.roots)
    if name == "vv10_pair":
        return _pair_program(resolve_method("WB97M-V").primitives[-1].spec)
    if name == "nuclear_pair_hvp":
        return _nuclear_pair_program()
    if name.startswith("grid_mixed_"):
        program = grid_mixed_response_program(name.removeprefix("grid_mixed_"), 3)
        return program.graph, program.roots
    if name == "ecp_harmonics":
        return harmonic_roots()
    if name == "ecp_radial_map":
        return radial_map_roots()
    return radial_roots(int(name.removeprefix("ecp_radial_")))


def scalar_inputs(graph: Any, roots: tuple) -> tuple[str, ...]:
    """Stable input-major ABI derived from reachable variable leaves only."""
    return tuple(
        sorted(
            {
                graph.nodes[index].payload
                for index in graph.topological_order(roots)
                if graph.nodes[index].operation == "variable"
            }
        )
    )


def emit_scalar(name: str) -> tuple[str, str, tuple[str, ...], int]:
    """Return CUDA source, scientific identity, input ABI and root count."""
    graph, roots = scalar_graph(name)
    names = scalar_inputs(graph, roots)
    identity = grid_response_graph_identity(
        graph, roots, name, schema="generativeqc.hessian-scalar-aot/v1"
    )
    return emit_array_graph(graph, roots, names), identity, names, len(roots)
