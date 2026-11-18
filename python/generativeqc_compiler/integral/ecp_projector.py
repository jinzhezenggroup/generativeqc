"""Bounded scalar ECP quadrature lowering shared by host tests and CUDA.

The radial measure cancels the r**-2 in EcpRadialTerm. Projected AO jets
carry derivatives of the two basis centers at fixed ECP-centered nodes;
translation invariance recovers the third center after each reduction.
The native adapter owns scheduling, storage and physical-atom scatter.
"""

import typing

from .ecp import emit_ecp_ao_cuda
from .ecp_grid import emit_ecp_grid_cpp
from .ecp_policy import emit_ecp_policy_cpp
from .ecp_schedule import emit_ecp_schedule_cpp
from .expr import Graph
from .ir import ECP_MAX_PROJECTOR_ANGULAR, EcpRadialTerm
from .scalar_c import ScalarCEmitter


def radial_roots(power: typing.Any) -> typing.Any:
    """Weighted residual radial integrand, including the volume measure."""
    EcpRadialTerm(-1, power, 1.0, 1.0)  # Reuse the operator domain contract.
    graph = Graph()
    r, alpha, coefficient, weight = (
        graph.variable(name) for name in ("r", "alpha", "coefficient", "weight")
    )
    polynomial = graph.constant(1)
    for _ in range(power):
        polynomial = polynomial * r
    return graph, (
        weight * coefficient * polynomial * graph.exponential(-alpha * r * r),
    )


def pair_roots() -> typing.Any:
    """Value and A/B center jets from the same bilinear expression."""
    graph = Graph()
    a, b, weight = (graph.variable(name) for name in ("a0", "b0", "weight"))
    value = weight * a * b
    da, db = graph.differentiate(value, a), graph.differentiate(value, b)
    return graph, (
        value,
        *(da * graph.variable(f"a{d}") for d in range(1, 4)),
        *(db * graph.variable(f"b{d}") for d in range(1, 4)),
    )


def direction_roots() -> typing.Any:
    """Project independent relative A/C and B/C gradients after integral AD."""
    graph = Graph()
    return graph, (
        graph.sum(
            graph.variable(f"gradient{slot * 3 + axis}")
            * (graph.variable(f"u{slot}_{axis}") - graph.variable(f"center{axis}"))
            for slot in range(2)
            for axis in range(3)
        ),
    )


def second_pair_roots() -> typing.Any:
    """Canonical value/relative gradient/ordered Hessian from bilinear AO jets."""
    graph = Graph()
    value = graph.variable("weight") * graph.variable("a0") * graph.variable("b0")
    coordinates = tuple((slot, axis) for slot in "ab" for axis in range(3))
    hessian = {
        (row, column): index + 4
        for index, (row, column) in enumerate(
            (row, column) for row in range(3) for column in range(row, 3)
        )
    }

    def derivative(root: typing.Any, slot: str, axis: int) -> typing.Any:
        bindings = {f"{slot}0": graph.variable(f"{slot}{1 + axis}")}
        bindings.update(
            {
                f"{slot}{1 + first}": graph.variable(
                    f"{slot}{hessian[min(first, axis), max(first, axis)]}"
                )
                for first in range(3)
            }
        )
        return graph.differentiate(root, graph.variable("coordinate"), bindings)

    gradients = tuple(derivative(value, slot, axis) for slot, axis in coordinates)
    return graph, (
        value,
        *gradients,
        *(
            derivative(root, slot, axis)
            for root in gradients
            for slot, axis in coordinates
        ),
    )


def column_roots() -> typing.Any:
    """Consume one relative jet with a fixed-weight left and supplied right JVP."""
    graph = Graph()
    weight = graph.variable("weight")
    motion = tuple(
        graph.variable(f"right{slot}_{axis}") - graph.variable(f"center{axis}")
        for slot in range(2)
        for axis in range(3)
    )
    gradient = tuple(graph.variable(f"jet{1 + axis}") for axis in range(6))
    value = weight * graph.variable("jet0")
    partials = tuple(weight * root for root in gradient)
    bindings = {
        "weight": graph.variable("weight_right"),
        "jet0": graph.sum(
            first * second for first, second in zip(gradient, motion, strict=True)
        ),
        **{
            f"jet{1 + axis}": graph.sum(
                graph.variable(f"jet{7 + 6 * axis + other}") * direction
                for other, direction in enumerate(motion)
            )
            for axis in range(6)
        },
    }
    coordinate = graph.variable("column_right")
    return graph, (
        value,
        graph.differentiate(value, coordinate, bindings),
        *partials,
        *(graph.differentiate(root, coordinate, bindings) for root in partials),
    )


def physical_second_roots() -> typing.Any:
    """Apply independently supplied physical incidence rows, including collisions."""
    graph = Graph()
    first = tuple(graph.variable(f"first{axis}") for axis in range(6))
    second = tuple(graph.variable(f"second{axis}") for axis in range(6))
    return graph, (
        graph.sum(
            graph.variable(f"jet{1 + axis}") * value for axis, value in enumerate(first)
        ),
        graph.sum(
            graph.variable(f"jet{7 + 6 * axis + other}") * left * right
            for axis, left in enumerate(first)
            for other, right in enumerate(second)
        ),
    )


def weighted_mixed_roots() -> typing.Any:
    """Contract canonical relative-center integral jets, not displaced operators.

    The 43 fields are value, six relative A/C and B/C gradients, then their
    ordered six-by-six Hessian. Physical centers may repeat; only their motion
    differences enter, preserving exact rigid-translation zeros.
    """
    graph = Graph()
    weight = graph.variable("weight")
    motions = {
        leg: tuple(
            graph.variable(f"{leg}{slot}_{axis}")
            - graph.variable(f"{leg}_center{axis}")
            for slot in range(2)
            for axis in range(3)
        )
        for leg in ("left", "right")
    }
    return graph, (
        weight * graph.variable("jet0"),
        *(
            weight
            * graph.sum(
                graph.variable(f"jet{1 + axis}") * motion
                for axis, motion in enumerate(motions[leg])
            )
            for leg in ("left", "right")
        ),
        weight
        * graph.sum(
            first * graph.variable(f"jet{7 + 6 * axis + other}") * second
            for axis, first in enumerate(motions["left"])
            for other, second in enumerate(motions["right"])
        ),
    )


def _emit(
    graph: typing.Any,
    roots: typing.Any,
    variables: typing.Any = None,
    *,
    accumulate: typing.Any = False,
) -> typing.Any:
    emitter = ScalarCEmitter(graph, variables or {})
    emitter.emit(roots)
    op = "+=" if accumulate else "="
    return emitter.lines + [
        f"  out[{i}] {op} {emitter.reference(root)};" for i, root in enumerate(roots)
    ]


def _scalar_function(
    name: typing.Any, arguments: typing.Any, expression: typing.Any
) -> typing.Any:
    """Lower small consumer expressions through the shared scalar algebra."""
    graph = Graph()
    root = expression(*(graph.variable(arg) for arg in arguments))
    emitter = ScalarCEmitter(graph, {})
    emitter.emit((root,))
    return [
        f"GENERATIVEQC_ECP_INLINE double {name}("
        + ", ".join(f"double {arg}" for arg in arguments)
        + ") {",
        *emitter.lines,
        f"  return {emitter.reference(root)};",
        "}",
    ]


def _emit_ao_consumer(*, second: bool = False) -> typing.Any:
    """Contract normalized Cartesian components without changing loop order."""
    lines = _scalar_function(
        "ecp_node_displacement",
        ("center", "radius", "direction", "basis_center"),
        lambda c, r, u, a: c + r * u - a,
    )
    lines += _scalar_function(
        "ecp_ao_term",
        ("component", "primitive", "value"),
        lambda c, p, v: c * p * v,
    )
    if second:
        lines = []
    name, kernel, width = (
        ("ecp_evaluate_second_ao", "ecp_ao_second", 10)
        if second
        else ("ecp_evaluate_ao", "ecp_ao", 4)
    )
    lines += [
        "template<class Jet, class AO, class Primitive, class Point>",
        f"GENERATIVEQC_ECP_INLINE Jet {name}(const AO& ao,",
        "    const Primitive* primitives, const Point& point, double radius,",
        "    double cx, double cy, double cz, bool derivatives) {",
        "  const double x = ecp_node_displacement(cx, radius, point.x, ao.x);",
        "  const double y = ecp_node_displacement(cy, radius, point.y, ao.y);",
        "  const double z = ecp_node_displacement(cz, radius, point.z, ao.z);",
        "  Jet out{};",
        "  for (int t=0; t<ao.term_count; ++t) {",
        "    const auto term = ao.components[t];",
        "    for (int k=0; k<ao.primitive_count; ++k) {",
        "      const auto p = primitives[ao.primitive_offset+k];",
        f"      double roots[{width}];",
        f"      {kernel}(term.x, term.y, term.z, x, y, z, p.exponent, roots);",
        f"      for (int d=0; d<(derivatives ? {width} : 1); ++d)",
        "        out.v[d] += ecp_ao_term(term.coefficient, p.coefficient, roots[d]);",
        "    }",
        "  }",
        "  return out;",
        "}",
    ]
    return lines


def _emit_weighted_consumer() -> typing.Any:
    """Full AO contraction: arbitrary real weights, no occupancy multiplier."""
    lines = _scalar_function(
        "ecp_add_operator",
        ("hcore", "local", "nonlocal_value"),
        lambda h, l, n: h + (l + n),
    )
    lines += _scalar_function(
        "ecp_weighted_term",
        ("weight", "local", "nonlocal_value"),
        lambda w, l, n: w * (l + n),
    )
    lines += _scalar_function(
        "ecp_energy_derivative_to_force", ("derivative",), lambda d: -d
    )
    lines += [
        "GENERATIVEQC_ECP_INLINE double ecp_force_component(const double* local,",
        "    const double* nonlocal_value, const double* weights, int size) {",
        "  double sum = 0;",
        "  for (int j=0; j<size; ++j)",
        "    sum += ecp_weighted_term(weights[j], local[j], nonlocal_value[j]);",
        "  return ecp_energy_derivative_to_force(sum);",
        "}",
    ]
    return lines


def _emit_column_consumer() -> list[str]:
    """Lower one integral jet into column fields and raw physical acceptance."""
    graph, roots = column_roots()
    variables = {
        "weight": "weight",
        "weight_right": "weight_right",
        **{f"jet{index}": f"jet[{index}]" for index in range(43)},
        **{
            f"right{slot}_{axis}": f"right_{name}[{axis}]"
            for slot, name in enumerate(("first", "second"))
            for axis in range(3)
        },
        **{f"center{axis}": f"center[{axis}]" for axis in range(3)},
    }
    lines = [
        "GENERATIVEQC_ECP_INLINE void ecp_weighted_column(double weight, double weight_right, const double* jet,",
        "    const double* right_first, const double* right_second, const double* center, double* out) {",
        *_emit(graph, roots, variables),
        "}",
    ]
    graph, roots = physical_second_roots()
    variables = {
        **{f"jet{index}": f"jet[{index}]" for index in range(43)},
        **{
            f"{slot}{index}": f"{slot}[{index}]"
            for slot in ("first", "second")
            for index in range(6)
        },
    }
    for name, root, parameters in (
        ("gradient", roots[0], "const double* first"),
        ("hessian", roots[1], "const double* first, const double* second"),
    ):
        emitter = ScalarCEmitter(graph, variables)
        emitter.emit((root,))
        lines += [
            f"GENERATIVEQC_ECP_INLINE double ecp_physical_{name}(const double* jet, {parameters}) {{",
            *emitter.lines,
            f"return {emitter.reference(root)};",
            "}",
        ]
    return lines


def emit_ecp_quadrature_cpp() -> typing.Any:
    """Emit the validated s/p/d/f projector and powers 0..4 domain.

    Type parameters expose only scalar term/node/jet records. They do not
    select a mathematical backend. The same emitted arithmetic is compiled
    as ordinary C++ for independent native tests and as CUDA for production.
    """
    lines = [
        "#pragma once",
        "#include <cmath>",
        "#include <array>",
        "#include <stdexcept>",
        "#include <vector>",
        "#if defined(__CUDACC__)",
        "#define GENERATIVEQC_ECP_INLINE __host__ __device__ inline",
        "#else",
        "#define GENERATIVEQC_ECP_INLINE inline",
        "#endif",
        emit_ecp_ao_cuda()
        .replace("#pragma once\n", "")
        .replace("__device__ inline", "GENERATIVEQC_ECP_INLINE"),
        "namespace generativeqc::generated {",
        *emit_ecp_schedule_cpp(),
        *emit_ecp_policy_cpp(),
        f"inline constexpr int ecp_max_projector_angular = {ECP_MAX_PROJECTOR_ANGULAR};",
        f"inline constexpr int ecp_projector_count = {(ECP_MAX_PROJECTOR_ANGULAR + 1) ** 2};",
        *emit_ecp_grid_cpp(),
        *_emit_ao_consumer(),
        *_emit_ao_consumer(second=True),
        *_emit_weighted_consumer(),
        *_emit_column_consumer(),
        "// Scalar ECP operator contract: local=-1, projectors=0..3, powers=0..4.",
        "GENERATIVEQC_ECP_INLINE double ecp_radial(unsigned power, double r,",
        "    double alpha, double coefficient, double weight) {",
        "  double out[1];",
        "  switch (power) {",
    ]
    for power in range(5):
        graph, roots = radial_roots(power)
        lines += [f"  case {power}: {{", *_emit(graph, roots), "return out[0]; }"]
    lines += [
        "  default: return NAN; // The public resolver rejects unsupported powers.",
        "  }",
        "}",
        "GENERATIVEQC_ECP_INLINE void ecp_pair(const double* a, const double* b,",
        "    double weight, bool derivatives, double* out) {",
    ]
    graph, roots = pair_roots()
    variables = {f"{ab}{d}": f"{ab}[{d}]" for ab in "ab" for d in range(4)}
    lines += _emit(graph, roots[:1], variables, accumulate=True)
    # A separate lexical scope avoids temporary-name collisions and does not
    # read derivative slots on a value-only call.
    lines += ["  if (derivatives) {"]
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit(roots[1:])
    lines += emitter.lines
    lines += [
        f"  out[{i}] += {emitter.reference(root)};"
        for i, root in enumerate(roots[1:], 1)
    ]
    lines += ["  }", "}"]
    graph, roots = second_pair_roots()
    lines += [
        "GENERATIVEQC_ECP_INLINE void ecp_second_pair(const double* a, const double* b,",
        "    double weight, double* out) {",
        *_emit(
            graph,
            roots,
            {
                f"{slot}{index}": f"{slot}[{index}]"
                for slot in "ab"
                for index in range(10)
            },
            accumulate=True,
        ),
        "}",
    ]
    graph = Graph()
    weight, harmonic, value = (
        graph.variable(name) for name in ("weight", "harmonic", "value")
    )
    emitter = ScalarCEmitter(graph, {})
    root = weight * harmonic * value
    emitter.emit((root,))
    lines += [
        "GENERATIVEQC_ECP_INLINE double ecp_projection_term(double weight,",
        "    double harmonic, double value) {",
        *emitter.lines,
        f"  return {emitter.reference(root)};",
        "}",
        "template<class Jet, class Point>",
        "GENERATIVEQC_ECP_INLINE Jet ecp_project(const Jet* values, const Point* sphere,",
        "    int nq, int m, bool derivatives, int fields=4) {",
        "  Jet out{};",
        "  for (int q=0; q<nq; ++q) {",
        "    const auto value = values[q];",
        "    for (int d=0; d<(derivatives ? fields : 1); ++d)",
        "      out.v[d] += ecp_projection_term(sphere[q].weight,",
        "          sphere[q].harmonics[m], value.v[d]);",
        "  }",
        "  return out;",
        "}",
        "template<class Term, class Radial>",
        "GENERATIVEQC_ECP_INLINE void ecp_potentials(const Term* terms, int nt,",
        "    const Radial& radial, int center, double* potential) {",
        f"  for (int l=0; l<{ECP_MAX_PROJECTOR_ANGULAR + 2}; ++l) potential[l]=0;",
        "  for (int t=0; t<nt; ++t) {",
        "    const auto term=terms[t];",
        "    if (term.atom_index == static_cast<unsigned>(center))",
        "      potential[term.channel+1] += ecp_radial(term.power, radial.r,",
        "          term.exponent, term.coefficient, radial.weight);",
        "  }",
        "}",
        "template<class Point, class Jet, int Fields>",
        "GENERATIVEQC_ECP_INLINE void ecp_contract_potentials(const double* potential,",
        "    const Point* sphere, int nq, const Jet* va, const Jet* vb,",
        "    const Jet* pa, const Jet* pb, bool derivatives, double (&parts)[2][Fields]) {",
        "  for (int part=0; part<2; ++part)",
        "    for (int d=0; d<Fields; ++d) parts[part][d]=0;",
        "  for (int q=0; q<nq; ++q) {",
        "    if constexpr (Fields == 43)",
        "      ecp_second_pair(va[q].v, vb[q].v, potential[0]*sphere[q].weight, parts[0]);",
        "    else ecp_pair(va[q].v, vb[q].v, potential[0]*sphere[q].weight, derivatives, parts[0]);",
        "  }",
    ]
    # Emit the projector multiplicities from the operator channel convention;
    # each l contains 2*l+1 orthonormal real harmonics, beginning at l*l.
    for channel in range(ECP_MAX_PROJECTOR_ANGULAR + 1):
        EcpRadialTerm(channel, 0, 1.0, 1.0)
        lines += [
            f"  for (int m={channel**2}; m<{(channel + 1) ** 2}; ++m) {{",
            "    if constexpr (Fields == 43)",
            f"      ecp_second_pair(pa[m].v, pb[m].v, potential[{channel + 1}], parts[1]);",
            f"    else ecp_pair(pa[m].v, pb[m].v, potential[{channel + 1}], derivatives, parts[1]);",
            "  }",
        ]
    graph = Graph()
    a, b = graph.variable("a"), graph.variable("b")
    emitter = ScalarCEmitter(graph, {"a": "parts[part][d+1]", "b": "parts[part][d+4]"})
    root = -(a + b)
    emitter.emit((root,))
    lines += [
        "  if constexpr (Fields == 10) if (derivatives)",
        "    for (int part=0; part<2; ++part)",
        "      for (int d=0; d<3; ++d) {",
        *emitter.lines,
        f"        parts[part][d+7] = {emitter.reference(root)};",
        "      }",
        "}",
        "template<class Term, class Radial, class Point, class Jet>",
        "GENERATIVEQC_ECP_INLINE void ecp_contract(const Term* terms, int nt,",
        "    const Radial& radial, const Point* sphere, int nq, int center,",
        "    const Jet* va, const Jet* vb, const Jet* pa, const Jet* pb,",
        "    bool derivatives, double (&parts)[2][10]) {",
        f"  double potential[{ECP_MAX_PROJECTOR_ANGULAR + 2}];",
        "  ecp_potentials(terms, nt, radial, center, potential);",
        "  ecp_contract_potentials(potential, sphere, nq, va, vb, pa, pb, derivatives, parts);",
        "}",
    ]
    graph, roots = direction_roots()
    variables = {
        **{f"gradient{index}": f"gradient[{index}]" for index in range(6)},
        **{
            f"u{slot}_{axis}": f"{name}[{axis}]"
            for slot, name in enumerate(("first", "second"))
            for axis in range(3)
        },
        **{f"center{axis}": f"center[{axis}]" for axis in range(3)},
    }
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit(roots)
    lines += [
        "GENERATIVEQC_ECP_INLINE double ecp_frozen_direction(const double* gradient,",
        "    const double* first, const double* second, const double* center) {",
        *emitter.lines,
        f"  return {emitter.reference(roots[0])};",
        "}",
    ]
    graph, roots = weighted_mixed_roots()
    variables = {
        "weight": "weight",
        **{f"jet{index}": f"jet[{index}]" for index in range(43)},
        **{
            f"{leg}{slot}_{axis}": f"{leg}_{name}[{axis}]"
            for leg in ("left", "right")
            for slot, name in enumerate(("first", "second"))
            for axis in range(3)
        },
        **{
            f"{leg}_center{axis}": f"{leg}_center[{axis}]"
            for leg in ("left", "right")
            for axis in range(3)
        },
    }
    lines += [
        "GENERATIVEQC_ECP_INLINE void ecp_weighted_mixed(double weight, const double* jet,",
        "    const double* left_first, const double* left_second, const double* left_center,",
        "    const double* right_first, const double* right_second, const double* right_center,",
        "    double* out) {",
        *_emit(graph, roots, variables),
        "}",
        "} // namespace generativeqc::generated",
        "#undef GENERATIVEQC_ECP_INLINE",
        "",
    ]
    return "\n".join(lines)
