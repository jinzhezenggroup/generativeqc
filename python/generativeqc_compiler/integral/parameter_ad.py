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


def parameter_kernel(
    family: str, components: tuple[str, ...], *, with_partials: bool = True
) -> ParameterKernel:
    """Build a canonical primitive without freezing its radial inputs.

    Tape consumers need only the primal graph and its analytic Boys leaves;
    avoid constructing expanded partial graphs that they will never execute.
    """
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
    for name in parameters if with_partials else ():
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


def _emit_kernel(
    name: str, kernel: ParameterKernel, *, second_order: bool = False
) -> str:
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
    source = "\n".join((*lines, "}"))
    if second_order:
        source += "\n" + _emit_directional_kernel(name, kernel)
    return source


def _emit_directional_kernel(name: str, kernel: ParameterKernel) -> str:
    """Differentiate value and VJP together, retaining Boys argument motion.

    The directional leaves are independent inputs. In particular this does not
    differentiate the numerical Boys recurrence or freeze its radial argument.
    """
    graph = kernel.graph
    direction = graph.variable("parameter_direction")
    leaves = {
        name: graph.variable(f"dx{i}") for i, name in enumerate(kernel.parameters)
    }
    variables = {name: f"x[{i}].value" for i, name in enumerate(kernel.parameters)}
    variables.update(
        {f"dx{i}": f"x[{i}].tangent" for i in range(len(kernel.parameters))}
    )
    variables["kPi"] = format(math.pi, ".17g")
    count = kernel.boys_count + int(kernel.boys_argument is not None)
    variables.update({f"boys_{i}": f"boys[{i}]" for i in range(count)})
    if kernel.boys_argument is not None:
        argument_direction = graph.differentiate(
            kernel.boys_argument, direction, leaves
        )
        leaves.update(
            {
                f"boys_{i}": -graph.variable(f"boys_{i + 1}") * argument_direction
                for i in range(kernel.boys_count)
            }
        )
    roots = (kernel.value, *kernel.partials)
    tangents = tuple(graph.differentiate(root, direction, leaves) for root in roots)
    emitter = ScalarCEmitter(graph, variables)
    lines = [
        f"static void {name}(const Directional* x, Directional* out, bool derivative) {{"
    ]
    if kernel.boys_argument is not None:
        emitter.emit((kernel.boys_argument,))
        lines.extend(emitter.lines)
        emitter.lines.clear()
        lines.append(f"double boys[{count}];")
        lines.append(
            f"boys_values<{count - 1}>({emitter.reference(kernel.boys_argument)}, boys);"
        )
    for index, (root, tangent) in enumerate(zip(roots, tangents, strict=True)):
        if index == 1:
            lines.append("if (!derivative) return;")
        emitter.emit((root, tangent))
        lines.extend(emitter.lines)
        emitter.lines.clear()
        lines.append(
            f"out[{index}] = {{{emitter.reference(root)}, {emitter.reference(tangent)}}};"
        )
    return "\n".join((*lines, "}"))


def _directional_arithmetic() -> str:
    """Generate the two scalar rules used by the contraction schedule."""
    graph = Graph()
    left, right, direction = (
        graph.variable(name) for name in ("left", "right", "direction")
    )
    variables = {
        "left": "a.value",
        "right": "b.value",
        "dleft": "a.tangent",
        "dright": "b.tangent",
    }
    lines = [
        "struct Directional { double value{}, tangent{}; Directional() = default; Directional(double v, double d=0): value(v), tangent(d) {} };"
    ]
    for op, root in (("+", left + right), ("*", left * right)):
        tangent = graph.differentiate(
            root,
            direction,
            {"left": graph.variable("dleft"), "right": graph.variable("dright")},
        )
        emitter = ScalarCEmitter(graph, variables)
        emitter.emit((root, tangent))
        lines.extend(
            (
                f"static Directional operator{op}(Directional a, Directional b) {{",
                *emitter.lines,
                f"return {{{emitter.reference(root)}, {emitter.reference(tangent)}}};",
                "}",
                f"static Directional& operator{op}=(Directional& a, Directional b) {{ a=a{op}b; return a; }}",
            )
        )
    lines.extend(
        (
            "static bool finite(Directional x) { return std::isfinite(x.value) && std::isfinite(x.tangent); }",
            "static bool zero(Directional x) { return x.value == 0; }",
        )
    )
    return "\n".join(lines)


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


def emit_normalization(
    angular: int, count: int, name: str, *, second_order: bool = False
) -> str:
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
    source = "\n".join((*lines, "}"))
    if not second_order:
        return source
    # The seed itself moves when the electronic response or other primitives
    # change. Its tangent is part of the VJP JVP, including zero coefficients.
    direction = graph.variable("normalization_direction")
    leaves = {name: graph.variable("d" + name) for name in variables}
    directional_variables = {
        name: value + ".value" for name, value in variables.items()
    }
    directional_variables.update(
        {"d" + name: value + ".tangent" for name, value in variables.items()}
    )
    emitter = ScalarCEmitter(graph, directional_variables)
    lines = [
        f"static void {name}(const Directional* x, const Directional* seed, Directional* out) {{"
    ]
    for i, root in enumerate(roots):
        if i == count:
            lines.append("if (!seed) return;")
        tangent = graph.differentiate(root, direction, leaves)
        emitter.emit((root, tangent))
        lines.extend(emitter.lines)
        emitter.lines.clear()
        lines.append(
            f"out[{i}] = {{{emitter.reference(root)}, {emitter.reference(tangent)}}};"
        )
    return source + "\n" + "\n".join((*lines, "}"))


def emit_primitive_family(
    components: tuple[str, ...],
    *,
    second_order: bool = False,
    compact: bool = False,
    metadata: dict[str, int] | None = None,
) -> str:
    """Emit canonical primitive values with bounded derivative graph storage.

    Compact and second-order consumers use the original scalar graph as a tape. Generated
    local AD rules replace expanded per-parameter Hessian expressions, keeping
    native compilation proportional to primal expression size.
    """
    from .parameter_tape import emit_tape_kernel, emit_tape_support

    compact = compact or second_order
    support = _hostify(_emit_boys_support(18 if second_order else 17))
    lines = [
        "#include <cmath>",
        "#ifndef GQC_AD_FN\n#define GQC_AD_FN static\n#endif",
        support,
    ]
    families = (("overlap", 2), ("kinetic", 2), ("nuclear_attraction", 2), ("eri", 4))
    kernels = {
        (family, index): parameter_kernel(family, selected, with_partials=not compact)
        for family, arity in families
        for index, selected in enumerate(product(components, repeat=arity))
    }
    maximum = 0
    codes = {}
    if compact:
        helpers, codes, maximum = emit_tape_support(
            tuple(kernels.values()), second_order=second_order
        )
        lines.extend((helpers, '#include "parameter_ad_tape.hpp"'))
    else:
        lines.append("constexpr int parameter_tape_nodes=0;")
    if metadata is not None:
        metadata["tape_nodes"] = maximum
    for family, arity in families:
        for index in range(len(components) ** arity):
            name = f"{family}_{index}"
            kernel = kernels[(family, index)]
            lines.append(
                emit_tape_kernel(name, kernel, codes, second_order=second_order)
                if compact
                else _emit_kernel(name, kernel)
            )
        lines.append(
            f"template<class Scalar> static void {family}(int index, const Scalar* x, Scalar* out, bool derivative, Scalar* scratch=nullptr) {{"
        )
        lines.append("switch (index) {")
        for index in range(len(components) ** arity):
            extra = ",scratch" if compact else ""
            lines.append(
                f"case {index}: {family}_{index}(x, out, derivative{extra}); return;"
            )
        lines.extend(("default: out[0] = NAN; return;", "}", "}"))
    return "\n".join(lines)


def emit_integral_program(
    numbers: tuple[int, ...],
    shells: tuple[tuple[int, int, int], ...],
    *,
    second_order: bool = False,
    backend: str = "cpu",
    metadata: dict[str, int] | None = None,
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
        _directional_arithmetic() if second_order else "",
        emit_primitive_family(
            components,
            second_order=second_order,
            compact=backend == "cuda",
            metadata=metadata,
        ),
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
        lines.append(
            emit_normalization(
                angular, count, f"normalize_{index}", second_order=second_order
            )
        )
    lines.append(
        "template<class Scalar> static void normalize(int shell, const Scalar* x, const Scalar* seed, Scalar* out) { switch (shell) {"
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
    lines.append(_emit_kernel("nuclear_pair", kernel, second_order=second_order))
    if second_order:
        lines.append("#define GQC_INTEGRAL_AD_SECOND 1")
    lines.extend(("}", '#include "parameter_ad_runtime.hpp"'))
    source = "\n".join(lines)
    if backend == "cpu":
        return source
    if backend != "cuda":
        raise ValueError("integral AD backend must be cpu or cuda")
    # Arithmetic is unchanged. Qualifiers and the runtime task schedule are
    # backend choices; device topology has static lifetime and no host aliases.
    source = source.replace("static ", "GQC_AD_FN ")
    source = source.replace("inline ", "__device__ inline ")
    source = source.replace(
        "constexpr int charges", "__device__ __constant__ int charges"
    )
    for name in ("shell_atom", "shell_begin", "ao_shell", "ao_component"):
        source = source.replace(
            "constexpr int " + name, "__device__ __constant__ int " + name
        )
    source = source.replace(
        "constexpr double ao_scale", "__device__ __constant__ double ao_scale"
    )
    source = source.replace(
        "Directional() = default;", "__host__ __device__ Directional() = default;"
    )
    source = source.replace(
        "Directional(double v", "__host__ __device__ Directional(double v"
    )
    return (
        "#include <cuda_runtime.h>\n#define GQC_AD_FN __device__\n#define GQC_INTEGRAL_AD_CUDA 1\n"
        + source
        + '\n#include "parameter_ad_cuda_runtime.cuh"'
    )
