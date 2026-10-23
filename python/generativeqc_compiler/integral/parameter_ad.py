"""Generate Gaussian primitive pullbacks from the canonical value expressions.

Centers and exponents are genuine inputs to the same S/T/V/ERI expression.
Boys moments retain their analytic chain rule, including exponent dependence;
contraction normalization is differentiated before consuming primitive weights.
There is no molecular force callback or displaced-integral evaluation here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from .expr import Graph
from .ir import build_integral_ir
from .one_electron_cpu import _hostify
from .one_electron_cuda import _emit_boys_support
from .one_electron_values import (
    build_one_electron_component_kernel,
    build_one_electron_value_ir,
)
from .scalar_c import ScalarCEmitter
from .shell_class import build_shell_class_component_kernel
from .shell_spec import AXES, ShellClassSpec, cartesian_components

if TYPE_CHECKING:
    from .expr import Expr


@dataclass(frozen=True)
class ParameterKernel:
    """Unnormalized value and ordered partials: exponents, then center xyz."""

    graph: Graph
    value: Expr
    parameters: tuple[str, ...]
    partials: tuple[Expr, ...]
    boys_argument: Expr | None
    boys_count: int


def parameter_kernel(family: str, components: tuple[str, ...]) -> ParameterKernel:
    """Differentiate a canonical primitive, without freezing its radial inputs."""
    angular = tuple(len(component) for component in components)
    if family == "eri":
        if len(angular) != 4 or any(l > 4 for l in angular):
            raise ValueError("ERI parameter AD requires four through-g components")
        spec = ShellClassSpec(
            "".join("spdfg"[l] for l in angular),
            (angular[0], angular[1], angular[2], angular[3]),
        )
        integral = build_integral_ir(spec, consumers=("fock",))
        kernel = build_shell_class_component_kernel(spec, components, integral=integral)
        names = ("alpha", "beta", "gamma", "delta")
        centers = ("first", "second", "third", "fourth")
        count = integral.maximum_coulomb_order + 1
    else:
        kernel = build_one_electron_component_kernel(
            build_one_electron_value_ir(family, angular), components
        )
        names = ("alpha", "beta")
        centers = ("a", "b", "c") if family == "nuclear_attraction" else ("a", "b")
        count = kernel.boys_count
    graph = kernel.graph
    parameters = (*names, *(f"{center}_{axis}" for center in centers for axis in AXES))
    partials = []
    for name in parameters:
        variable = graph.variable(name)
        leaves = {}
        if kernel.boys_argument is not None:
            tangent = graph.differentiate(kernel.boys_argument, variable)
            leaves = {
                f"boys_{order}": -graph.variable(f"boys_{order + 1}") * tangent
                for order in range(count)
            }
        partials.append(graph.differentiate(kernel.value, variable, leaves))
    return ParameterKernel(
        graph,
        kernel.value,
        parameters,
        tuple(partials),
        kernel.boys_argument,
        count + int(kernel.boys_argument is not None),
    )


def _emit_kernel(name: str, kernel: ParameterKernel) -> str:
    variables = {name: f"x[{i}]" for i, name in enumerate(kernel.parameters)}
    variables["kPi"] = format(math.pi, ".17g")
    variables.update({f"boys_{i}": f"boys[{i}]" for i in range(kernel.boys_count)})
    emitter = ScalarCEmitter(kernel.graph, variables)
    lines = [f"static void {name}(const double* x, double* out, bool derivative) {{"]
    if kernel.boys_argument is not None:
        emitter.emit((kernel.boys_argument,))
        lines.extend(emitter.lines)
        emitter.lines.clear()
        lines.append(f"double boys[{kernel.boys_count}];")
        lines.append(
            f"boys_values<{kernel.boys_count - 1}>("
            f"{emitter.reference(kernel.boys_argument)}, boys);"
        )
    emitter.emit((kernel.value,))
    lines.extend(emitter.lines)
    emitter.lines.clear()
    lines.extend(
        (f"out[0] = {emitter.reference(kernel.value)};", "if (!derivative) return;")
    )
    emitter.emit(kernel.partials)
    lines.extend(emitter.lines)
    lines.extend(
        f"out[{i + 1}] = {emitter.reference(root)};"
        for i, root in enumerate(kernel.partials)
    )
    return "\n".join((*lines, "}"))


def normalization_program(angular: int, count: int) -> tuple[Graph, tuple[Expr, ...]]:
    """Normalized radial coefficients and their seeded pullback to raw alpha/c.

    The component double-factorial is coordinate independent and applied by
    the contraction schedule. A common shell norm is valid for every Cartesian
    component. Coefficients may be zero or signed; no division by a coefficient
    occurs in the pullback.
    """
    graph = Graph()
    alpha = tuple(graph.variable(f"x[{i}]") for i in range(count))
    coefficient = tuple(graph.variable(f"x[{count + i}]") for i in range(count))
    norm2 = graph.sum(
        coefficient[i]
        * coefficient[j]
        * (2 * (alpha[i] * alpha[j]).pow(0.5) / (alpha[i] + alpha[j])).pow(
            angular + 1.5
        )
        for i, j in product(range(count), repeat=2)
    )
    normalized = tuple(
        coefficient[i]
        * norm2.pow(-0.5)
        * (2 * alpha[i] / math.pi).pow(0.75)
        * (4 * alpha[i]).pow(angular / 2)
        for i in range(count)
    )
    objective = graph.sum(
        graph.variable(f"seed[{i}]") * value for i, value in enumerate(normalized)
    )
    derivatives = tuple(
        graph.differentiate(objective, variable) for variable in (*alpha, *coefficient)
    )
    return graph, (*normalized, *derivatives)


def emit_normalization(angular: int, count: int, name: str) -> str:
    """Emit native normalization and VJP from one shared symbolic expression."""
    graph, roots = normalization_program(angular, count)
    variables = {f"x[{i}]": f"x[{i}]" for i in range(2 * count)}
    variables.update({f"seed[{i}]": f"seed[{i}]" for i in range(count)})
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit(roots[:count])
    lines = [
        f"static void {name}(const double* x, const double* seed, double* out) {{",
        *emitter.lines,
    ]
    emitter.lines.clear()
    lines.extend(
        f"out[{i}] = {emitter.reference(root)};" for i, root in enumerate(roots[:count])
    )
    lines.append("if (!seed) return;")
    emitter.emit(roots[count:])
    lines.extend(emitter.lines)
    lines.extend(
        f"out[{count + i}] = {emitter.reference(root)};"
        for i, root in enumerate(roots[count:])
    )
    return "\n".join((*lines, "}"))


def emit_primitive_family(components: tuple[str, ...]) -> str:
    """Emit only the requested Cartesian components, with finite compiler work.

    Dispatch indices are structural AO-component controls, never differentiable
    data. Scientific expressions stay in the existing integral owners.
    """
    # Public through-g ERIs need F_17 for exponent/center pullbacks. The shared
    # positive series/downward recurrence itself is independent of that bound.
    support = _hostify(_emit_boys_support(17))
    lines = ["#include <cmath>", support]
    for family, arity in (
        ("overlap", 2),
        ("kinetic", 2),
        ("nuclear_attraction", 2),
        ("eri", 4),
    ):
        for index, selected in enumerate(product(components, repeat=arity)):
            lines.append(
                _emit_kernel(f"{family}_{index}", parameter_kernel(family, selected))
            )
        lines.append(
            f"static void {family}(int index, const double* x, double* out, bool derivative) {{"
        )
        lines.append("switch (index) {")
        for index in range(len(components) ** arity):
            lines.append(f"case {index}: {family}_{index}(x, out, derivative); return;")
        lines.extend(("default: out[0] = NAN; return;", "}", "}"))
    return "\n".join(lines)


def emit_integral_program(
    numbers: tuple[int, ...], shells: tuple[tuple[int, int, int], ...]
) -> str:
    """Specialize a bounded contraction schedule by immutable basis topology.

    Each shell is (physical atom, angular momentum, primitive count). Geometry,
    primitive exponents and raw coefficients remain runtime parameters and do
    not enter generated source or its cache key.
    """
    components = tuple(
        dict.fromkeys(
            component
            for _, angular, _ in shells
            for component in cartesian_components(angular)
        )
    )
    ao_shell, ao_component, ao_scale = [], [], []
    begin = [0]
    for index, (_, angular, count) in enumerate(shells):
        begin.append(begin[-1] + count)
        for component in cartesian_components(angular):
            ao_shell.append(index)
            ao_component.append(components.index(component))
            denominator = math.prod(
                math.prod(range(1, 2 * component.count(axis), 2)) for axis in AXES
            )
            ao_scale.append(1 / math.sqrt(denominator))
    lines = [
        "#include <cmath>",
        "namespace integral_ad {",
        emit_primitive_family(components),
    ]
    lines.append(
        f"constexpr int nat={len(numbers)}, nshell={len(shells)}, nprim={begin[-1]}, "
        f"nao={len(ao_shell)}, ncomponent={len(components)};"
    )
    for name, values, dtype in (
        ("charges", numbers, "int"),
        ("shell_atom", tuple(s[0] for s in shells), "int"),
        ("shell_begin", begin, "int"),
        ("ao_shell", ao_shell, "int"),
        ("ao_component", ao_component, "int"),
        ("ao_scale", ao_scale, "double"),
    ):
        lines.append(
            f"constexpr {dtype} {name}[] = {{"
            + ",".join(f"{value}" for value in values)
            + "};"
        )
    for index, (_, angular, count) in enumerate(shells):
        lines.append(emit_normalization(angular, count, f"normalize_{index}"))
    lines.append(
        "static void normalize(int shell, const double* x, const double* seed, double* out) { switch (shell) {"
    )
    lines.extend(
        f"case {i}: normalize_{i}(x, seed, out); return;" for i in range(len(shells))
    )
    lines.extend(("default: out[0]=NAN; return;", "}}"))
    graph = Graph()
    variables = tuple(graph.variable(f"x{i}") for i in range(6))
    value = graph.sum((variables[i] - variables[i + 3]).pow(2) for i in range(3)).pow(
        -0.5
    )
    kernel = ParameterKernel(
        graph,
        value,
        tuple(f"x{i}" for i in range(6)),
        tuple(graph.differentiate(value, variable) for variable in variables),
        None,
        0,
    )
    lines.append(_emit_kernel("nuclear_pair", kernel))
    lines.extend(("}", '#include "parameter_ad_runtime.hpp"'))
    return "\n".join(lines)
