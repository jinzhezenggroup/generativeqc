"""Bounded mixed-geometry lowering of the incumbent S/T/V component DAGs.

Expose and differentiate exact Hermite/Coulomb subexpressions once, then
compose their four-field jets in the existing component expression. This is
factor scheduling, not another recurrence or a displaced production evaluator.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from .expr import Graph
from .graph_composition import import_roots
from .one_electron_values import (
    build_one_electron_component_kernel,
    build_one_electron_value_ir,
)
from .scalar_c import ScalarCEmitter
from .second_derivatives import _eri_differentiate
from .shell_class import build_coulomb_derivative_algebra
from .shell_spec import cartesian_components

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .expr import Expr


def _mixed(
    graph: Graph, root: Expr, names: Sequence[str], *, affine: Sequence[str] = ()
) -> tuple[Expr, ...]:
    left = graph.differentiate(
        root,
        graph.variable("left_coordinate"),
        {name: graph.variable("left_" + name) for name in names},
    )
    right_bindings = {name: graph.variable("right_" + name) for name in names}
    right = graph.differentiate(
        root, graph.variable("right_coordinate"), right_bindings
    )
    mixed = graph.differentiate(
        left,
        graph.variable("right_coordinate"),
        {
            **right_bindings,
            **{
                "left_" + name: graph.constant(0)
                if name in affine
                else graph.variable("mixed_" + name)
                for name in names
            },
        },
    )
    return root, left, right, mixed


def _emit(
    graph: Graph, roots: Sequence[Expr], variables: dict[str, str]
) -> tuple[list[str], list[str]]:
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit(roots)
    return emitter.lines, [emitter.reference(root) for root in roots]


def _hermite_source(
    source: Graph, root: Expr, axis: int, key: tuple[int, int, int]
) -> str:
    graph = Graph()
    label = "xyz"[axis]
    (value,) = import_roots(
        graph,
        source,
        (root,),
        {
            "alpha": graph.variable("alpha"),
            "beta": graph.variable("beta"),
            "a_" + label: graph.variable("coordinate"),
            "b_" + label: graph.constant(0),
        },
    )
    roots = _mixed(graph, value, ("coordinate",), affine=("coordinate",))
    lines, references = _emit(
        graph,
        roots,
        {
            "alpha": "alpha",
            "beta": "beta",
            "coordinate": "coordinate",
            "left_coordinate": "left",
            "right_coordinate": "right",
            "mixed_coordinate": "0.0",
        },
    )
    name = "hermite_" + "_".join(map(str, key))
    return "\n".join(
        [
            f"static Jet {name}([[maybe_unused]] double alpha, [[maybe_unused]] double beta,",
            "    [[maybe_unused]] double coordinate, [[maybe_unused]] double left, [[maybe_unused]] double right) {",
            *lines,
            "return {" + ", ".join(references) + "};",
            "}",
        ]
    )


def _coulomb_sources() -> str:
    algebra = build_coulomb_derivative_algebra(6)
    graph = algebra.graph
    differentiate = _eri_differentiate(graph, 8)
    variables = {
        "rho": "geometry.rho",
        "first_product_scale": "1.0",
        **{
            f"difference_{axis}": f"geometry.difference[{index}]"
            for index, axis in enumerate("xyz")
        },
        **{f"boys_{index}": f"geometry.boys[{index}]" for index in range(9)},
        **{
            f"left_{axis}": f"geometry.left[{index}]"
            for index, axis in enumerate("xyz")
        },
        **{
            f"right_{axis}": f"geometry.right[{index}]"
            for index, axis in enumerate("xyz")
        },
    }
    functions = []
    for orders, value in algebra.roots:
        gradients = [differentiate(value, 0, axis) for axis in "xyz"]
        left = graph.sum(
            gradient * graph.variable("left_" + axis)
            for axis, gradient in zip("xyz", gradients, strict=True)
        )
        right = graph.sum(
            gradient * graph.variable("right_" + axis)
            for axis, gradient in zip("xyz", gradients, strict=True)
        )
        mixed = graph.sum(
            graph.variable("left_" + first)
            * differentiate(gradient, 0, second)
            * graph.variable("right_" + second)
            for first, gradient in zip("xyz", gradients, strict=True)
            for second in "xyz"
        )
        lines, references = _emit(graph, (value, left, right, mixed), variables)
        name = "coulomb_state_" + "_".join(map(str, orders))
        functions.append(
            "\n".join(
                [
                    f"static Jet {name}(const Geometry& geometry) {{",
                    *lines,
                    "return {" + ", ".join(references) + "};",
                    "}",
                ]
            )
        )
    return "\n".join(functions)


def _geometry_source() -> str:
    kernel = build_one_electron_component_kernel(
        build_one_electron_value_ir("nuclear_attraction", (0, 0)), ("", "")
    )
    graph = kernel.graph
    geometry = dict(kernel.coulomb_geometry)
    roots = [
        geometry["rho"],
        *(geometry["difference_" + axis] for axis in "xyz"),
        geometry["argument"],
    ]
    targets = [
        "geometry.rho",
        *(f"geometry.difference[{index}]" for index in range(3)),
        "argument",
    ]
    moving = tuple(f"{center}_{axis}" for center in "abc" for axis in "xyz")
    for index, axis in enumerate("xyz"):
        jets = _mixed(graph, geometry["difference_" + axis], moving)
        roots.extend(jets[1:3])
        targets.extend((f"geometry.left[{index}]", f"geometry.right[{index}]"))
    variables = {"alpha": "alpha", "beta": "beta"}
    for slot, name in enumerate(moving):
        variables[name] = f"coordinates[{slot}]"
        variables["left_" + name] = f"left[{slot}]"
        variables["right_" + name] = f"right[{slot}]"
    lines, references = _emit(graph, roots, variables)
    return "\n".join(
        [
            "static bool prepare(double alpha, double beta, const double* coordinates,",
            "    const double* left, const double* right, Geometry& geometry) {",
            "double argument;",
            *lines,
            *(
                f"{target} = {reference};"
                for target, reference in zip(targets, references, strict=True)
            ),
            "return range_moments(8, argument, 1, CoulombRange::Full, 0, geometry.boys);",
            "}",
        ]
    )


def emit_one_electron_mixed_cpu() -> str:
    """Emit finite unnormalized Cartesian s/p/d/f S/T and unit-charge V jets.

    Public coefficients/normalization and method weights remain the native
    consumer's responsibility. The common-origin input frame makes rigid
    translation exact. Only four mixed scalar fields per operator are retained;
    no coordinate Hessian, runtime compilation or runtime reference is used.
    """
    components = tuple(
        (order, label) for order in range(4) for label in cartesian_components(order)
    )
    helpers, functions, dispatch = {}, [], {"overlap_kinetic": [], "attraction": []}
    for kind, pair in product(dispatch, product(range(20), repeat=2)):
        graph = Graph()
        first, second = (components[index] for index in pair)
        families = (
            ("overlap", "kinetic")
            if kind == "overlap_kinetic"
            else ("nuclear_attraction",)
        )
        kernels = [
            build_one_electron_component_kernel(
                build_one_electron_value_ir(family, (first[0], second[0])),
                (first[1], second[1]),
                graph=graph,
            )
            for family in families
        ]
        factors = {}
        for kernel in kernels:
            for (axis, left_power, right_power, order), value in kernel.hermite_roots:
                if graph.node(value).operation != "constant":
                    factors.setdefault(
                        value.identifier,
                        (
                            f"hermite_{axis}_{left_power}_{right_power}_{order}",
                            value,
                            (axis, left_power, right_power, order),
                        ),
                    )
            for orders, value in kernel.coulomb_roots:
                factors.setdefault(
                    value.identifier,
                    ("coulomb_" + "_".join(map(str, orders)), value, None),
                )
        substituted = graph.replace_subexpressions(
            tuple(kernel.value for kernel in kernels),
            {value: graph.variable(name) for name, value, _ in factors.values()},
        )
        reachable = {
            str(graph.nodes[index].payload)
            for index in graph.topological_order(substituted)
            if graph.nodes[index].operation == "variable"
        }
        variables = {"alpha": "alpha", "beta": "beta"}
        names, prelude = [], []
        for slot, center in enumerate("ab"):
            for axis, label in enumerate("xyz"):
                name = center + "_" + label
                names.append(name)
                variables[name] = f"coordinates[{3 * slot + axis}]"
                variables["left_" + name] = f"left[{3 * slot + axis}]"
                variables["right_" + name] = f"right[{3 * slot + axis}]"
                variables["mixed_" + name] = "0.0"
        for name, value, state in factors.values():
            if name not in reachable:
                continue
            names.append(name)
            for index, prefix in enumerate(("", "left_", "right_", "mixed_")):
                variables[prefix + name] = f"{name}[{index}]"
            if state is None:
                prelude.append(
                    f"const Jet {name} = {name.replace('coulomb_', 'coulomb_state_')}(geometry);"
                )
            else:
                axis, left_power, right_power, order = state
                key = left_power, right_power, order
                if key not in helpers:
                    helpers[key] = _hermite_source(graph, value, axis, key)
                function = "hermite_" + "_".join(map(str, key))
                prelude.append(
                    f"const Jet {name} = {function}(alpha, beta, coordinates[{axis}]-coordinates[{3 + axis}], left[{axis}]-left[{3 + axis}], right[{axis}]-right[{3 + axis}]);"
                )
        if not reachable <= set(variables):
            raise ValueError(
                f"unbound one-electron factor variables: {reachable - set(variables)}"
            )
        affine = tuple(f"{center}_{axis}" for center in "ab" for axis in "xyz")
        roots = tuple(
            root
            for value in substituted
            for root in _mixed(graph, value, names, affine=affine)
        )
        lines, references = _emit(graph, roots, variables)
        name = f"{kind}_{pair[0]}_{pair[1]}"
        parameters = "double alpha, double beta, const double* coordinates, const double* left, const double* right, "
        if kind == "attraction":
            parameters += "const Geometry& geometry, "
        functions.append(
            "\n".join(
                [
                    f"static void {name}({parameters}double* output) {{",
                    *prelude,
                    *lines,
                    *(
                        f"output[{index}] = {reference};"
                        for index, reference in enumerate(references)
                    ),
                    "}",
                ]
            )
        )
        arguments = "alpha, beta, coordinates, moving_left, moving_right, "
        if kind == "attraction":
            arguments += "geometry, "
        dispatch[kind].append(
            f"case {20 * pair[0] + pair[1]}: {name}({arguments}candidate.data()); break;"
        )
    power_table = ",".join(
        "{" + ",".join(str(label.count(axis)) for axis in "xyz") + "}"
        for _, label in components
    )
    source = [
        '#include "integrals/host_mixed_primitives.hpp"',
        '#include "integrals/range_moments.hpp"',
        "#include <algorithm>",
        "#include <array>",
        "#include <cmath>",
        "namespace generativeqc::integrals::host_mixed { namespace {",
        "using Jet = std::array<double, 4>;",
        "struct Geometry { double rho{}, difference[3]{}, left[3]{}, right[3]{}, boys[9]{}; };",
        *helpers.values(),
        _coulomb_sources(),
        _geometry_source(),
        *functions,
        "static unsigned component(const unsigned* powers) {",
        f"constexpr unsigned table[20][3] = {{{power_table}}};",
        "for (unsigned index=0; index<20; ++index)",
        " if (std::equal(powers,powers+3,table[index])) return index;",
        "return 20; }",
        "}",
    ]
    for kind, cases in dispatch.items():
        count, width = (3, 4) if kind == "attraction" else (2, 8)
        source.extend(
            [
                f"bool {kind}(const double* exponents, const double* centers, const unsigned* powers,",
                " const double* left, const double* right, double* output) noexcept {",
                "if (!exponents || !centers || !powers || !left || !right || !output) return false;",
                "const double alpha=exponents[0], beta=exponents[1];",
                "if (!(alpha>0) || !(beta>0) || !std::isfinite(alpha+beta)) return false;",
                "const auto first=component(powers), second=component(powers+3);",
                "if (first>=20 || second>=20) return false;",
                f"double coordinates[{3 * count}], moving_left[{3 * count}], moving_right[{3 * count}];",
                f"for (unsigned field=0; field<{3 * count}; ++field) {{",
                f" const auto origin={3 * (count - 1)}+field%3;",
                " coordinates[field]=centers[field]-centers[origin];",
                " moving_left[field]=left[field]-left[origin]; moving_right[field]=right[field]-right[origin];",
                " if (!std::isfinite(centers[field]) || !std::isfinite(left[field]) || !std::isfinite(right[field]) ||",
                "     !std::isfinite(coordinates[field]) || !std::isfinite(moving_left[field]) || !std::isfinite(moving_right[field])) return false; }",
            ]
        )
        if kind == "attraction":
            source.extend(
                [
                    "Geometry geometry;",
                    "if (!prepare(alpha,beta,coordinates,moving_left,moving_right,geometry)) return false;",
                ]
            )
        source.extend(
            [
                f"std::array<double,{width}> candidate{{}};",
                "switch (20*first+second) {",
                *cases,
                "default: return false; }",
                "for (double value : candidate) if (!std::isfinite(value)) return false;",
                "std::copy(candidate.begin(),candidate.end(),output); return true; }",
            ]
        )
    return "\n".join([*source, "}", ""])
