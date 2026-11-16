"""CUDA lowering of pruned, shared one-electron S/T/V scalar DAGs.

The native contraction layer owns normalized coefficients and output layouts.
Standalone generation defaults to s/p/d/f; full-DF CUDA builds request g.
Pair geometry remains hoisted out of the nuclear loop and S/T share scalar
subexpressions. G-bearing scalars share joint-axis and Gaussian-center orbits,
then reuse exact ancestors without evaluating inactive components.
"""

import re
import typing
from functools import cache
from itertools import permutations, product

from .cuda import CudaEmitter
from .expr import Expr, Graph, Node
from .ir_serialization import integral_to_payload
from .one_electron_values import (
    build_one_electron_component_kernel,
    build_one_electron_value_ir,
)
from .shell_spec import cartesian_components

_MAXIMUM_GENERATED_ANGULAR_MOMENTUM = 4


def _validate_maximum_angular_momentum(value: typing.Any) -> int:
    """Validate the bounded generated one-electron shell family."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAXIMUM_GENERATED_ANGULAR_MOMENTUM
    ):
        raise ValueError(
            "generated one-electron shells require maximum angular momentum in [0,4]"
        )
    return value


def _component_layout(
    maximum_angular_momentum: typing.Any,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """Return per-shell counts, offsets, upper bounds and flattened AO count."""

    maximum = _validate_maximum_angular_momentum(maximum_angular_momentum)
    counts = tuple(len(cartesian_components(l)) for l in range(maximum + 1))
    offsets = []
    total = 0
    limits = []
    for count in counts:
        offsets.append(total)
        total += count
        limits.append(total)
    return counts, tuple(offsets), tuple(limits), total


def _shell_index_expression(component: str, limits: tuple[int, ...]) -> str:
    """Emit a branch-only shell lookup for one flattened Cartesian component."""

    expression = str(len(limits) - 1)
    for angular in range(len(limits) - 2, -1, -1):
        expression = f"{component} < {limits[angular]} ? {angular} : {expression}"
    return expression


def one_electron_program_inventory(
    maximum_angular_momentum: int = 3,
) -> typing.Any:
    """Record scientific signatures separately from runtime/promotion evidence."""
    maximum = _validate_maximum_angular_momentum(maximum_angular_momentum)
    return {
        "schema": "generativeqc.one_electron_values",
        "version": 1,
        "precision": "fp64",
        "schedules": ["thread", "shell_warp", "nucleus_8", "nucleus_32"],
        "programs": [
            integral_to_payload(build_one_electron_value_ir(family, angular))
            for family in ("overlap", "kinetic", "nuclear_attraction")
            for angular in product(range(maximum + 1), repeat=2)
        ],
    }


def _geometry_boundary(
    kernel: typing.Any, roots: typing.Any, *, target: Graph | None = None
) -> typing.Any:
    """Cut hoisted geometry roots out of a copy of the executable DAG.

    Substitution before topological scheduling also removes their ancestors;
    renaming emitted temporaries alone would leave redundant exponentials and
    divisions inside every nuclear-center invocation.
    """
    target = Graph() if target is None else target
    replacements = {expr.identifier: name for name, expr in kernel.pair_geometry}

    @cache
    def visit(identifier: typing.Any) -> typing.Any:
        if identifier in replacements:
            return target.variable("pair." + replacements[identifier])
        node = kernel.graph.nodes[identifier]
        if node.operation == "variable":
            name = str(node.payload)
            if name.startswith("boys_"):
                return target.variable(f"boys[{name[5:]}]")
            return target.variable(name if name.startswith("c_") else "pair." + name)
        if node.operation == "constant":
            return target.clone_constant(node)
        # Preserve the already simplified algebra, interning shared ancestors
        # in the target graph just as the scientific builder does.
        return target._intern(
            Node(
                node.operation,
                tuple(visit(i).identifier for i in node.arguments),
                node.payload,
            )
        )

    return target, tuple(visit(root.identifier) for root in roots)


def _pair_geometry_inventory() -> typing.Any:
    """Return the kernel and complete field order shared by value/response emitters."""
    kernel = build_one_electron_component_kernel(
        build_one_electron_value_ir("overlap", (0, 0)), ("", "")
    )
    fields = ["alpha", "beta"] + [
        f"{center}_{axis}" for center in "ab" for axis in "xyz"
    ]
    fields += [name for name, _ in kernel.pair_geometry]
    return kernel, fields


def _emit_pair_geometry() -> typing.Any:
    kernel, fields = _pair_geometry_inventory()
    lines = ["struct PairGeometry {", "  double " + ", ".join(fields) + ";", "};"]
    lines += [
        "__device__ __forceinline__ PairGeometry make_pair(",
        "    double alpha, double beta, double a_x, double a_y, double a_z,",
        "    double b_x, double b_y, double b_z) {",
        "  PairGeometry pair{};",
    ]
    for field in fields[:8]:
        lines.append(f"  pair.{field} = {field};")
    emitter = CudaEmitter(kernel.graph, {})
    for name, expr in kernel.pair_geometry:
        emitter.emit_assignment(expr, f"pair.{name}")
    return "\n".join(lines + emitter.lines + ["  return pair;", "}"])


@cache
def _canonical_value_component_map(
    angular: tuple[int, int],
) -> tuple[tuple[int, int, bool], ...]:
    """Reach one scalar representative with simultaneous Cartesian permutations.

    Scalars permit all six axis permutations, unlike a fixed-axis derivative.
    Equal shells additionally permit A/B exchange. Prefer the identity and no
    exchange on ties so representative entries require no coordinate changes.
    """
    components = tuple(cartesian_components(value) for value in angular)
    mapping = []
    for first, second in product(*components):
        alternatives = []
        for exchanged in (False, True) if angular[0] == angular[1] else (False,):
            pair = (second, first) if exchanged else (first, second)
            for permutation, axes in enumerate(permutations("xyz")):
                transformed = tuple(
                    "".join(
                        axis * component.count(source)
                        for axis, source in zip("xyz", axes, strict=True)
                    )
                    for component in pair
                )
                index = components[0].index(transformed[0]) * len(components[1])
                index += components[1].index(transformed[1])
                alternatives.append((index, permutation, exchanged))
        canonical = min(alternatives, key=lambda entry: (entry[0], entry[2], entry[1]))
        mapping.append(canonical)
    return tuple(mapping)


def _emit_value_symmetries() -> str:
    """Emit g-only scalar orbit maps and rounded Cartesian field transforms.

    Radial quantities are already rounded by ``make_pair``. Copy them unchanged
    and permute or exchange only the coordinate/exponent ownership, just as the
    gradient emitter's axis covariance does. This is a mathematical symmetry,
    not a promise of bitwise floating-point equivalence.
    """
    _, fields = _pair_geometry_inventory()
    axis_fields = sorted(
        {field[:-2] for field in fields if field.endswith(("_x", "_y", "_z"))}
    )
    lines = [
        "struct CanonicalValueComponent { unsigned index, permutation; bool exchanged; };",
        "__device__ __forceinline__ PairGeometry permute_value_pair(const PairGeometry& pair, unsigned permutation) {",
        "  PairGeometry result = pair;",
        "  switch (permutation) {",
    ]
    for permutation, axes in enumerate(permutations("xyz")):
        if not permutation:
            continue
        lines.append(f"    case {permutation}U:")
        for field in axis_fields:
            for axis, source in zip("xyz", axes, strict=True):
                if axis != source:
                    lines.append(
                        f"      result.{field}_{axis} = pair.{field}_{source};"
                    )
        lines.append("      break;")
    lines += ["  }", "  return result;", "}"]
    lines += [
        "__device__ __forceinline__ PairGeometry exchange_value_pair(const PairGeometry& pair) {",
        "  PairGeometry result = pair;",
        "  result.alpha = pair.beta; result.beta = pair.alpha;",
    ]
    for first, second in (("a", "b"), ("pa", "pb")):
        for axis in "xyz":
            lines += [
                f"  result.{first}_{axis} = pair.{second}_{axis};",
                f"  result.{second}_{axis} = pair.{first}_{axis};",
            ]
    lines += ["  return result;", "}"]
    lines += [
        "struct ValueCenter { double x, y, z; };",
        "__device__ __forceinline__ ValueCenter permute_value_center(double c_x, double c_y, double c_z, unsigned permutation) {",
        "  switch (permutation) {",
    ]
    for permutation, axes in enumerate(permutations("xyz")):
        if permutation:
            values = ", ".join("c_" + axis for axis in axes)
            lines.append(f"    case {permutation}U: return {{{values}}};")
    lines += ["  }", "  return {c_x, c_y, c_z};", "}"]
    for angular in ((0, 4), (1, 4), (2, 4), (3, 4), (4, 4)):
        lines += [
            f"__device__ __forceinline__ CanonicalValueComponent canonical_value_{angular[0]}{angular[1]}(unsigned component) {{",
            "  switch (component) {",
        ]
        for index, (canonical, permutation, exchanged) in enumerate(
            _canonical_value_component_map(angular)
        ):
            if index != canonical:
                lines.append(
                    f"    case {index}U: return {{{canonical}U, {permutation}U, {str(exchanged).lower()}}};"
                )
        lines += ["  }", "  return {component, 0U, false};", "}"]
    return "\n".join(lines)


def _emit_shared_shell_helper(attraction: bool, angular: tuple[int, int]) -> str:
    """Lower scalar g orbits with non-speculative sharing between representatives.

    Joint coordinate/power permutations select one canonical representative;
    reversing unequal shells additionally transposes the component pair. Keep
    representatives in a separate noinline body rather than recursive dispatch.
    Existing Graph value numbering hoists only ancestors needed by every
    representative. Invalid indices refuse before any transform or arithmetic,
    and the complete <=f artifact stays on its established scalar lowering.
    """
    name = "attraction" if attraction else "overlap_kinetic"
    result_type, invalid = ("double", "NAN") if attraction else ("ST", "ST{NAN, NAN}")
    arguments = "const PairGeometry& pair, unsigned component"
    if attraction:
        arguments += ", double c_x, double c_y, double c_z"
    count_first, count_second = (len(cartesian_components(value)) for value in angular)
    if angular[0] > angular[1]:
        external = ", c_x, c_y, c_z" if attraction else ""
        return "\n".join(
            [
                f"static __device__ __forceinline__ {result_type} {name}_{angular[0]}{angular[1]}({arguments}) {{",
                f"  if (component >= {count_first * count_second}U) return {invalid};",
                f"  const unsigned transposed = (component % {count_second}U) * {count_first}U + component / {count_second}U;",
                f"  return {name}_{angular[1]}{angular[0]}(exchange_value_pair(pair), transposed{external});",
                "}",
            ]
        )
    mapping = _canonical_value_component_map(angular)
    target = Graph()
    roots_by_case = {}
    boys_count = sum(angular) + 1
    for index, components in enumerate(
        product(*(cartesian_components(value) for value in angular))
    ):
        if mapping[index][0] != index:
            continue
        # Keep each scientific DAG's original operand ordering. Global source
        # IDs can reorder n-ary sums/products even before scalar emission.
        graph = Graph()
        if attraction:
            kernel = build_one_electron_component_kernel(
                build_one_electron_value_ir("nuclear_attraction", angular),
                components,
                graph=graph,
            )
            _, roots = _geometry_boundary(
                kernel, (kernel.boys_argument, kernel.value), target=target
            )
        else:
            overlap = build_one_electron_component_kernel(
                build_one_electron_value_ir("overlap", angular), components, graph=graph
            )
            kernel = build_one_electron_component_kernel(
                build_one_electron_value_ir("kinetic", angular), components, graph=graph
            )
            _, roots = _geometry_boundary(
                kernel, (overlap.value, kernel.value), target=target
            )
        roots_by_case[index] = roots
    lines = [
        f"static __device__ __noinline__ {result_type} {name}_{angular[0]}{angular[1]}_canonical({arguments}) {{",
        f"  if (component >= {count_first * count_second}U) return {invalid};",
    ]
    common = set.intersection(
        *(set(target.topological_order(roots)) for roots in roots_by_case.values())
    )
    shared = CudaEmitter(target, {})
    if attraction:
        # This order/argument is identical for all Cartesian components. Boys
        # inputs must be initialized before any hoisted node that reads them.
        argument = roots_by_case[0][0]
        shared.emit((argument,))
        lines.extend(shared.lines)
        shared.lines.clear()
        lines += [
            f"  double boys[{boys_count}];",
            f"  boys_values<{boys_count - 1}>({shared.reference(argument)}, boys);",
        ]
    shared.emit(tuple(Expr(target, identifier) for identifier in sorted(common)))
    lines.extend(shared.lines)

    lines.append("  switch (component) {")
    for index, roots in roots_by_case.items():
        emitter = shared.fork()
        emitter.emit(roots)
        result = (
            emitter.reference(roots[1])
            if attraction
            else "{" + ", ".join(emitter.reference(root) for root in roots) + "}"
        )
        lines.append(f"    case {index}U: {{")
        lines.extend(emitter.lines)
        lines.extend([f"      return {result};", "    }"])
    lines.append("  }")
    lines.extend([f"  return {invalid};", "}"])
    lines += [
        f"static __device__ __forceinline__ {result_type} {name}_{angular[0]}{angular[1]}({arguments}) {{",
        f"  if (component >= {count_first * count_second}U) return {invalid};",
        f"  const auto canonical = canonical_value_{angular[0]}{angular[1]}(component);",
        "  auto rotated = permute_value_pair(pair, canonical.permutation);",
        "  if (canonical.exchanged) rotated = exchange_value_pair(rotated);",
    ]
    external = ""
    if attraction:
        lines.append(
            "  const auto center = permute_value_center(c_x, c_y, c_z, canonical.permutation);"
        )
        external = ", center.x, center.y, center.z"
    lines += [
        f"  return {name}_{angular[0]}{angular[1]}_canonical(rotated, canonical.index{external});",
        "}",
    ]
    return "\n".join(lines)


def _emit_operator_helpers(
    attraction: typing.Any, maximum_angular_momentum: int = 3
) -> typing.Any:
    maximum = _validate_maximum_angular_momentum(maximum_angular_momentum)
    counts, offsets, limits, total_components = _component_layout(maximum)
    shell_count = maximum + 1
    name = "attraction" if attraction else "overlap_kinetic"
    return_type = "double" if attraction else "ST"
    arguments = "const PairGeometry& pair, unsigned component"
    if attraction:
        arguments += ", double c_x, double c_y, double c_z"
    lines = []
    for angular in product(range(maximum + 1), repeat=2):
        if 4 in angular:
            lines.append(
                _emit_shared_shell_helper(bool(attraction), (angular[0], angular[1]))
            )
            continue
        suffix = f"{angular[0]}{angular[1]}"
        lines += [
            f"static __device__ __noinline__ {return_type} {name}_{suffix}({arguments}) {{",
            "  switch (component) {",
        ]
        for index, components in enumerate(
            product(*(cartesian_components(l) for l in angular))
        ):
            graph = Graph()
            if attraction:
                kernel = build_one_electron_component_kernel(
                    build_one_electron_value_ir("nuclear_attraction", angular),
                    components,
                    graph=graph,
                )
                target, roots = _geometry_boundary(
                    kernel, (kernel.boys_argument, kernel.value)
                )
            else:
                s = build_one_electron_component_kernel(
                    build_one_electron_value_ir("overlap", angular),
                    components,
                    graph=graph,
                )
                kernel = build_one_electron_component_kernel(
                    build_one_electron_value_ir("kinetic", angular),
                    components,
                    graph=graph,
                )
                target, roots = _geometry_boundary(kernel, (s.value, kernel.value))
            emitter = CudaEmitter(target, {})
            lines.append(f"    case {index}U: {{")
            if attraction:
                emitter.emit((roots[0],))
                lines += emitter.lines
                emitter.lines.clear()
                lines += [
                    f"      double boys[{kernel.boys_count}];",
                    f"      boys_values<{kernel.boys_count - 1}>({emitter.reference(roots[0])}, boys);",
                ]
                emitter.emit((roots[1],))
                result = emitter.reference(roots[1])
            else:
                emitter.emit(roots)
                result = (
                    "{" + ", ".join(emitter.reference(root) for root in roots) + "}"
                )
            lines += emitter.lines + [f"      return {result};", "    }"]
        invalid = "NAN" if attraction else "ST{NAN, NAN}"
        lines += ["  }", f"  return {invalid};", "}"]
    lines += [
        f"__device__ __forceinline__ {return_type} {name}(",
        "    const PairGeometry& pair, unsigned first, unsigned second"
        + (", double c_x, double c_y, double c_z) {" if attraction else ") {"),
        "  const unsigned a = " + _shell_index_expression("first", limits) + ";",
        "  const unsigned b = " + _shell_index_expression("second", limits) + ";",
        "  const unsigned offsets[] = {" + ", ".join(map(str, offsets)) + "};",
        "  const unsigned counts[] = {" + ", ".join(map(str, counts)) + "};",
        "  const unsigned component = (first - offsets[a]) * counts[b] + second - offsets[b];",
        f"  if (first >= {total_components} || second >= {total_components}) return "
        + invalid
        + ";",
        f"  switch (a * {shell_count}U + b) {{",
    ]
    for a, b in product(range(maximum + 1), repeat=2):
        args = "pair, component" + (", c_x, c_y, c_z" if attraction else "")
        lines.append(f"    case {a * shell_count + b}U: return {name}_{a}{b}({args});")
    lines += ["  }", f"  return {invalid};", "}"]
    return "\n".join(lines)


def _emit_boys_support(maximum_boys_order: int = 6) -> typing.Any:
    """Emit the shared Boys evaluator body without backend/header wrappers."""
    if (
        isinstance(maximum_boys_order, bool)
        or not isinstance(maximum_boys_order, int)
        or not 0 <= maximum_boys_order <= 18
    ):
        raise ValueError("generated Boys support requires maximum order in [0,18]")
    source = r"""template<unsigned Order>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  static_assert(Order <= 6);
  const double decay = exp(-argument);
  if (argument < 20.0) {
    double term = 1.0 / (2 * Order + 1);
    double sum = term;
    for (unsigned k = 1; k < 160; ++k) {
      term *= 2.0 * argument / (2 * Order + 2 * k + 1);
      sum += term;
      if (term < 1.0e-17 * sum) break;
    }
    values[Order] = decay * sum;
    for (unsigned n = Order; n > 0; --n)
      values[n - 1] = (2.0 * argument * values[n] + decay) / (2 * n - 1);
  } else {
    values[0] = 0.88622692545275801365 * erf(sqrt(argument)) / sqrt(argument);
    for (unsigned n = 1; n <= Order; ++n)
      values[n] = ((2 * n - 1) * values[n - 1] - decay) / (2.0 * argument);
  }
}
"""
    if maximum_boys_order != 6:
        source = source.replace(
            "static_assert(Order <= 6);",
            f"static_assert(Order <= {maximum_boys_order});",
        )
    return source


def _emission_symbols(
    owner: str | None, *, derivatives: bool = False
) -> tuple[str, str]:
    """Isolate consumer-specific specializations to avoid cross-TU ODR collisions.

    Ownership changes C++ symbols and include guards, not mathematical IR. The
    default preserves the full-DF artifact's existing public generated names.
    """
    if owner is not None and (
        not isinstance(owner, str)
        or re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", owner) is None
    ):
        raise ValueError(
            "one-electron emission owner must be a lower snake-case identifier"
        )
    qualifier = f"_{owner}" if owner is not None else ""
    suffix = "_derivatives" if derivatives else ""
    namespace = f"generativeqc::scf::generated{qualifier}_one_electron{suffix}"
    kind = "DERIVATIVES" if derivatives else "VALUES"
    guard = f"GENERATIVEQC_GENERATED{qualifier.upper()}_ONE_ELECTRON_{kind}_CUH"
    return namespace, guard


def _emit_support_cuda(
    maximum_boys_order: int = 6,
    *,
    owner: str | None = None,
    derivatives: bool = False,
) -> typing.Any:
    """Wrap the shared Boys evaluator in a consumer-owned primitive preamble."""
    namespace, guard = _emission_symbols(owner, derivatives=derivatives)
    return "\n".join(
        (
            "// Generated by tools/generate_one_electron_kernels.py; do not edit.",
            f"#ifndef {guard}",
            f"#define {guard}",
            "#include <cuda_runtime.h>",
            "#include <cmath>",
            f"namespace {namespace} {{",
            "struct ST { double overlap, kinetic; };",
            "",
            "/** Positive-term series plus downward recurrence avoids cancellation at small T.",
            " * F_m(T) = exp(-T) sum_k (2T)^k / [(2m+1)(2m+3)...(2m+2k+1)].",
            " * The generated bound is explicit in the shared evaluator. */",
            _emit_boys_support(maximum_boys_order),
        )
    )


def _emit_component_index(maximum_angular_momentum: int = 3) -> typing.Any:
    """Map normalized basis expansion powers to the common public AO order."""
    maximum = _validate_maximum_angular_momentum(maximum_angular_momentum)
    _, _, _, total_components = _component_layout(maximum)
    radix = maximum + 1
    index = [
        "/** Public Cartesian order, shared by all native basis expansion terms. */",
        "__device__ __forceinline__ unsigned component_index(unsigned x, unsigned y, unsigned z) {",
        f"  if (x > {maximum}U || y > {maximum}U || z > {maximum}U ||",
        f"      x + y + z > {maximum}U) return {total_components}U;",
        f"  switch (x * {radix * radix}U + y * {radix}U + z) {{",
    ]
    for i, component in enumerate(
        c for l in range(maximum + 1) for c in cartesian_components(l)
    ):
        x, y, z = (component.count(axis) for axis in "xyz")
        key = x * radix * radix + y * radix + z
        index.append(f"    case {key}U: return {i}U;")
    index += ["  }", f"  return {total_components}U;", "}"]
    return "\n".join(index)


def emit_one_electron_values_cuda(
    maximum_angular_momentum: int = 3,
    *,
    owner: str | None = None,
) -> typing.Any:
    """Emit S/T and unit-charge V, optionally in a consumer-owned namespace.

    The caller chooses its supported shell bound and applies nuclear charges.
    Distinct owners can link different bounds without sharing inline definitions.
    """
    maximum = _validate_maximum_angular_momentum(maximum_angular_momentum)
    namespace, _ = _emission_symbols(owner)
    return "\n".join(
        [
            _emit_support_cuda(2 * maximum, owner=owner),
            _emit_pair_geometry(),
            _emit_component_index(maximum),
            *([_emit_value_symmetries()] if maximum == 4 else []),
            _emit_operator_helpers(False, maximum),
            _emit_operator_helpers(True, maximum),
            f"}}  // namespace {namespace}",
            "#endif",
            "",
        ]
    )
