"""CUDA scalar lowering of shared S/T/V first-derivative DAG roots."""

import typing
from functools import cache
from itertools import product

from .cuda import CudaEmitter
from .expr import Graph
from .ir_serialization import integral_to_payload
from .one_electron_cuda import (
    _component_layout,
    _emission_symbols,
    _emit_component_index,
    _emit_pair_geometry,
    _emit_support_cuda,
    _geometry_boundary,
    _pair_geometry_inventory,
    _shell_index_expression,
    _validate_maximum_angular_momentum,
)
from .one_electron_derivatives import (
    build_one_electron_derivative_ir,
    build_one_electron_derivative_kernel,
)
from .shell_spec import cartesian_components


def one_electron_derivative_inventory(
    maximum_angular_momentum: int = 3,
) -> typing.Any:
    """Record mathematical center/sign/layout contracts before runtime scheduling."""
    maximum = _validate_maximum_angular_momentum(maximum_angular_momentum)
    return {
        "schema": "generativeqc.one_electron_derivatives",
        "version": 1,
        "precision": "fp64",
        "programs": [
            integral_to_payload(
                build_one_electron_derivative_ir(family, angular, weighted=weighted)
            )
            for family in ("overlap", "kinetic", "nuclear_attraction")
            for angular in product(range(maximum + 1), repeat=2)
            for weighted in (False, True)
        ],
    }


@cache
def _canonical_yz_component_map(angular: tuple[int, int]) -> tuple[int, ...]:
    """Choose one representative of each y/z orbit, leaving the x axis fixed."""
    components = tuple(cartesian_components(value) for value in angular)
    mapping = []
    for index, pair in enumerate(product(*components)):
        rotated = tuple(
            "x" * component.count("x")
            + "y" * component.count("z")
            + "z" * component.count("y")
            for component in pair
        )
        rotated_first = components[0].index(rotated[0])
        rotated_second = components[1].index(rotated[1])
        swapped = rotated_first * len(components[1]) + rotated_second
        mapping.append(min(index, swapped))
    return tuple(mapping)


@cache
def _canonical_exchange_component_map(
    angular: tuple[int, int],
) -> tuple[tuple[int, bool, bool], ...]:
    """Compose y/z covariance with A/B exchange for equal shell families.

    The transformations commute and preserve the differentiated x axis. Return
    the representative and the y/z and center-exchange flags needed to reach it.
    Prefer no center exchange on ties, keeping canonical entries idempotent and
    avoiding unnecessary sign/field changes.
    """
    if angular[0] != angular[1]:
        raise ValueError("component center-exchange orbits require equal shells")
    count = len(cartesian_components(angular[0]))
    yz = _canonical_yz_component_map(angular)
    mapping = []
    for index, representative in enumerate(yz):
        transposed = (index % count) * count + index // count
        exchanged = yz[transposed]
        if exchanged < representative:
            mapping.append((exchanged, exchanged != transposed, True))
        else:
            mapping.append((representative, representative != index, False))
    return tuple(mapping)


def _emit_axis_permutations(maximum_angular_momentum: int = 3) -> typing.Any:
    """Exploit Cartesian-axis covariance without widening the public AO family.

    Every primitive operator is a scalar under an x/y or x/z coordinate swap.
    Swapping both Gaussian powers and all A/B/C positions turns its requested
    derivative into the emitted x derivative. Radial/component normalization is
    invariant under these permutations. This bounds live CSE roots to two.
    """
    maximum = _validate_maximum_angular_momentum(maximum_angular_momentum)
    _, _, _, total_components = _component_layout(maximum)
    components = tuple(c for l in range(maximum + 1) for c in cartesian_components(l))
    lines = [
        "__device__ __forceinline__ unsigned permute_component(unsigned index, unsigned axis) {",
        "  if (axis == 0) return index;",
        "  switch (index) {",
    ]
    for index, component in enumerate(components):
        choices = []
        for axis in (1, 2):
            powers = [component.count(a) for a in "xyz"]
            powers[0], powers[axis] = powers[axis], powers[0]
            transformed = "".join(a * n for a, n in zip("xyz", powers, strict=True))
            choices.append(components.index(transformed))
        lines.append(
            f"    case {index}U: return axis == 1 ? {choices[0]}U : {choices[1]}U;"
        )
    lines += [
        "  }",
        f"  return {total_components}U;",
        "}",
        "__device__ __forceinline__ PairGeometry permute_pair(const PairGeometry& pair, unsigned axis) {",
        "  PairGeometry result = pair;",
    ]
    _, fields = _pair_geometry_inventory()
    axis_fields = {field[:-2] for field in fields if field.endswith(("_x", "_y", "_z"))}
    for prefix in axis_fields:
        if not all(f"{prefix}_{axis}" in fields for axis in "xyz"):
            raise ValueError(f"incomplete Cartesian geometry field: {prefix}")
    for axis, name in ((1, "y"), (2, "z")):
        lines.append(f"  if (axis == {axis}) {{")
        for center in sorted(axis_fields):
            lines += [
                f"    result.{center}_x = pair.{center}_{name};",
                f"    result.{center}_{name} = pair.{center}_x;",
            ]
        lines.append("  }")
    lines += ["  return result;", "}"]
    if maximum == 4:
        lines += [
            "__device__ __forceinline__ PairGeometry permute_pair_yz(const PairGeometry& pair){",
            "  PairGeometry result=pair;",
        ]
        for prefix in sorted(axis_fields):
            lines += [
                f"  result.{prefix}_y=pair.{prefix}_z;",
                f"  result.{prefix}_z=pair.{prefix}_y;",
            ]
        lines += ["  return result;", "}"]
        # Preserve the already rounded radial geometry. Only exponents, centers
        # and P-A/P-B offsets change ownership when the Gaussian pair is reversed.
        lines += [
            "__device__ __forceinline__ PairGeometry exchange_pair(const PairGeometry& pair){",
            "  PairGeometry result=pair;",
            "  result.alpha=pair.beta; result.beta=pair.alpha;",
        ]
        for first, second in (("a", "b"), ("pa", "pb")):
            for axis in "xyz":
                lines += [
                    f"  result.{first}_{axis}=pair.{second}_{axis};",
                    f"  result.{second}_{axis}=pair.{first}_{axis};",
                ]
        lines += ["  return result;", "}"]
        for first_angular, second_angular in product(range(maximum + 1), repeat=2):
            angular = (first_angular, second_angular)
            if 4 not in angular or angular[0] > angular[1]:
                continue
            if angular == (4, 4):
                lines += [
                    "struct CanonicalGradientComponent { unsigned index; bool yz, exchanged; };",
                    "__device__ __forceinline__ CanonicalGradientComponent canonical_exchange_44(unsigned component){",
                    "  switch(component){",
                ]
                for index, (canonical, yz, exchanged) in enumerate(
                    _canonical_exchange_component_map(angular)
                ):
                    if index != canonical:
                        lines.append(
                            f"    case {index}U:return {{{canonical}U,{str(yz).lower()},{str(exchanged).lower()}}};"
                        )
                lines += ["    default:return {component,false,false};", "  }", "}"]
                continue
            lines += [
                f"__device__ __forceinline__ unsigned canonical_yz_{angular[0]}{angular[1]}(unsigned component){{",
                "  switch(component){",
            ]
            for index, canonical in enumerate(_canonical_yz_component_map(angular)):
                if index != canonical:
                    lines.append(f"    case {index}U:return {canonical}U;")
            lines += ["    default:return component;", "  }", "}"]
    return "\n".join(lines)


def _emit_gradient_shell_helper(attraction: bool, angular: tuple[int, int]) -> str:
    """Emit x roots, sharing g-shell y/z and Gaussian-center exchange orbits.

    A y/z swap fixes the differentiated x axis. Swap both Gaussian powers,
    every Cartesian pair field and the external center, but retain the radial
    geometry/Boys values just as the outer x/y and x/z lowering already does.
    Reversing A/B transposes Cartesian components. S/T first-center derivatives
    change sign by translational invariance; attraction derivatives exchange
    their A/B output fields instead. These are mathematical identities, not
    bitwise floating-point equivalence. Only g-bearing pairs use the additional
    dispatch; <=f lowering and precomputed Boys values are intact.
    """
    name = "attraction_gradient" if attraction else "overlap_kinetic_gradient"
    name += f"_x_{angular[0]}{angular[1]}"
    arguments = "const PairGeometry& pair, unsigned component"
    if attraction:
        arguments += ", double c_x, double c_y, double c_z, const double* boys"
    if 4 in angular and angular[0] > angular[1]:
        count_first, count_second = (
            len(cartesian_components(value)) for value in angular
        )
        canonical_name = name.removesuffix(f"{angular[0]}{angular[1]}")
        canonical_name += f"{angular[1]}{angular[0]}"
        external = ",c_x,c_y,c_z,boys" if attraction else ""
        result = (
            "{value.second,value.first}"
            if attraction
            else "{-value.first,-value.second}"
        )
        return "\n".join(
            [
                f"static __device__ __forceinline__ GradientAxis {name}({arguments}) {{",
                f"  if(component >= {count_first * count_second}U) return {{NAN,NAN}};",
                f"  const unsigned transposed=(component%{count_second}U)*{count_first}U+component/{count_second}U;",
                f"  const auto value={canonical_name}(exchange_pair(pair),transposed{external});",
                f"  return {result};",
                "}",
            ]
        )
    components = tuple(product(*(cartesian_components(value) for value in angular)))
    mapping = _canonical_yz_component_map(angular) if 4 in angular else None
    exchange_mapping = (
        _canonical_exchange_component_map(angular) if angular == (4, 4) else None
    )
    kernel_name = name + "_canonical" if mapping is not None else name
    lines = [
        f"static __device__ __noinline__ GradientAxis {kernel_name}({arguments}) {{",
        "  switch (component) {",
    ]
    for index, component in enumerate(components):
        if exchange_mapping is not None and exchange_mapping[index][0] != index:
            continue
        if exchange_mapping is None and mapping is not None and mapping[index] != index:
            continue
        graph = Graph()
        if attraction:
            kernel = build_one_electron_derivative_kernel(
                build_one_electron_derivative_ir("nuclear_attraction", angular),
                component,
                graph=graph,
            )
            roots = (kernel.gradients[0][0], kernel.gradients[1][0])
        else:
            overlap = build_one_electron_derivative_kernel(
                build_one_electron_derivative_ir("overlap", angular),
                component,
                graph=graph,
            )
            kernel = build_one_electron_derivative_kernel(
                build_one_electron_derivative_ir("kinetic", angular),
                component,
                graph=graph,
            )
            roots = (overlap.gradients[0][0], kernel.gradients[0][0])
        target, roots = _geometry_boundary(kernel, roots)
        emitter = CudaEmitter(target, {})
        emitter.emit(roots)
        lines += [
            f"    case {index}U: {{",
            *emitter.lines,
            "      return {"
            + ", ".join(emitter.reference(root) for root in roots)
            + "};",
            "    }",
        ]
    lines += ["  }", "  return {NAN,NAN};", "}"]
    if exchange_mapping is not None:
        external = (
            ",c_x,canonical.yz?c_z:c_y,canonical.yz?c_y:c_z,boys" if attraction else ""
        )
        result = (
            "canonical.exchanged ? GradientAxis{value.second,value.first} : value"
            if attraction
            else "canonical.exchanged ? GradientAxis{-value.first,-value.second} : value"
        )
        lines += [
            f"static __device__ __forceinline__ GradientAxis {name}({arguments}) {{",
            f"  if(component >= {len(exchange_mapping)}U) return {{NAN,NAN}};",
            "  const auto canonical=canonical_exchange_44(component);",
            "  auto transformed=canonical.yz ? permute_pair_yz(pair) : pair;",
            "  if(canonical.exchanged) transformed=exchange_pair(transformed);",
            f"  const auto value={name}_canonical(transformed,canonical.index{external});",
            f"  return {result};",
            "}",
        ]
        return "\n".join(lines)
    if mapping is not None:
        external = ",c_x,c_z,c_y,boys" if attraction else ""
        lines += [
            f"static __device__ __forceinline__ GradientAxis {name}({arguments}) {{",
            f"  if(component >= {len(mapping)}U) return {{NAN,NAN}};",
            f"  const unsigned canonical=canonical_yz_{angular[0]}{angular[1]}(component);",
            "  if(canonical!=component){",
            "    const auto rotated=permute_pair_yz(pair);",
            f"    return {name}_canonical(rotated,canonical{external});",
            "  }",
        ]
        external = ",c_x,c_y,c_z,boys" if attraction else ""
        lines += [f"  return {name}_canonical(pair,canonical{external});", "}"]
    return "\n".join(lines)


def _emit_gradient_helpers(
    attraction: typing.Any, maximum_angular_momentum: int = 3
) -> typing.Any:
    """Emit two x-axis roots; reuse them for y/z through exact permutations.

    Attraction evaluates Boys values once before all three axis calls. Keeping
    each helper to S/T or A/B on one axis avoids retaining six roots and their
    cross-axis CSE ancestors simultaneously. This trades modest recomputation
    for a smaller optimization unit and bounded register pressure.
    """
    maximum = _validate_maximum_angular_momentum(maximum_angular_momentum)
    counts, offsets, limits, total_components = _component_layout(maximum)
    shell_count = maximum + 1
    maximum_boys_order = 2 * maximum + 1
    name = "attraction_gradient" if attraction else "overlap_kinetic_gradient"
    lines = [
        _emit_gradient_shell_helper(bool(attraction), (first, second))
        for first, second in product(range(maximum + 1), repeat=2)
    ]
    external = (
        ", double c_x, double c_y, double c_z, const double* boys" if attraction else ""
    )
    lines += [
        f"__device__ __forceinline__ GradientAxis {name}_x(",
        "    const PairGeometry& pair, unsigned first, unsigned second"
        + external
        + ") {",
        f"  if (first >= {total_components} || second >= {total_components}) return {{NAN,NAN}};",
        "  const unsigned a = " + _shell_index_expression("first", limits) + ";",
        "  const unsigned b = " + _shell_index_expression("second", limits) + ";",
        "  const unsigned offsets[] = {" + ",".join(map(str, offsets)) + "};",
        "  const unsigned counts[] = {" + ",".join(map(str, counts)) + "};",
        "  const unsigned component = (first-offsets[a])*counts[b]+second-offsets[b];",
        f"  switch (a*{shell_count}U+b) {{",
    ]
    for a, b in product(range(maximum + 1), repeat=2):
        args = "pair, component" + (", c_x, c_y, c_z, boys" if attraction else "")
        lines.append(
            f"    case {a * shell_count + b}U: return {name}_x_{a}{b}({args});"
        )
    lines += [
        "  }",
        "  return {NAN,NAN};",
        "}",
        f"__device__ __forceinline__ GradientPair {name}(",
        "    const PairGeometry& pair, unsigned first, unsigned second"
        + (", double c_x, double c_y, double c_z" if attraction else "")
        + ") {",
        "  GradientPair result{};",
    ]
    if attraction:
        # The Boys argument is rotationally invariant and independent of the
        # shell powers. Use its validated graph root once for all axis calls.
        kernel = build_one_electron_derivative_kernel(
            build_one_electron_derivative_ir("nuclear_attraction", (0, 0)), ("", "")
        )
        target, roots = _geometry_boundary(kernel, (kernel.boys_argument,))
        emitter = CudaEmitter(target, {})
        emitter.emit(roots)
        lines += [
            *emitter.lines,
            f"  double boys[{maximum_boys_order + 1}];",
            f"  boys_values<{maximum_boys_order}>({emitter.reference(roots[0])},boys);",
        ]
    lines += [
        "  for (unsigned axis=0; axis<3; ++axis) {",
        "    const auto rotated = permute_pair(pair,axis);",
        "    const auto a = permute_component(first,axis), b = permute_component(second,axis);",
    ]
    args = "rotated, a, b"
    if attraction:
        args += ", axis==0 ? c_x : axis==1 ? c_y : c_z, axis==1 ? c_x : c_y, axis==2 ? c_x : c_z, boys"
    lines += [
        f"    const auto value = {name}_x({args});",
        "    result.first[axis]=value.first; result.second[axis]=value.second;",
        "  }",
        "  return result;",
        "}",
    ]
    return "\n".join(lines)


def emit_one_electron_derivatives_cuda(
    maximum_angular_momentum: int = 3,
    *,
    owner: str | None = None,
) -> typing.Any:
    """Emit gradients in an optional consumer namespace; the caller owns AO weights.

    Use distinct owners when linking specializations with different shell bounds.
    The bound and owner are emission choices, not changes to mathematical IR.
    """
    maximum = _validate_maximum_angular_momentum(maximum_angular_momentum)
    namespace, _ = _emission_symbols(owner, derivatives=True)
    prefix = _emit_support_cuda(2 * maximum + 1, owner=owner, derivatives=True)
    return "\n".join(
        [
            prefix,
            "struct GradientPair { double first[3], second[3]; };",
            "struct GradientAxis { double first, second; };",
            "__device__ __forceinline__ GradientPair invalid_gradient() { const double n = NAN; return {{n,n,n},{n,n,n}}; }",
            _emit_pair_geometry(),
            _emit_component_index(maximum),
            _emit_axis_permutations(maximum),
            _emit_gradient_helpers(False, maximum),
            _emit_gradient_helpers(True, maximum),
            f"}}  // namespace {namespace}",
            "#endif",
            "",
        ]
    )
