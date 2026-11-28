"""Bounded factor lowering of the canonical Hermite/Boys second-order algebra.

Expanded quartet-component AD repeats the same one-dimensional pair factors
and Coulomb states in thousands of translation units. This emitter differentiates
those existing factors once, then composes their jets with compiler-generated
product rules. It is a schedule for the existing scalar algebra, not another
integral recurrence. All differentiation is finished before native compilation.
"""

from __future__ import annotations

from collections import defaultdict
from functools import cache
from itertools import product

from .cuda import CudaEmitter
from .expr import Expr, Graph
from .second_derivatives import _eri_differentiate
from .shell_class import _pair_expansion, build_coulomb_derivative_algebra
from .shell_spec import cartesian_components


def _jet_function(
    graph: Graph,
    roots: tuple,
    variables: dict,
    targets: tuple,
    *,
    output: str = "full",
    fields: tuple[str, ...] = (),
) -> str:
    """Share scalar CSE without teaching the native template derivative rules."""
    if fields:
        selected = {
            "full": ("value", "directional", "gradient", "jvp"),
            "first": ("value", "gradient"),
            "directional": ("value", "directional"),
            "value": ("value",),
        }[output]
        pairs = tuple(
            (root, target)
            for root, target, field in zip(roots, targets, fields, strict=True)
            if field in selected
        )
        roots = tuple(root for root, _ in pairs)
        targets = tuple(target for _, target in pairs)
    emitter = CudaEmitter(graph, variables)
    emitter.emit(roots)
    return "\n".join(
        (
            *emitter.lines,
            *(
                f"{target}={emitter.reference(root)};"
                for target, root in zip(targets, roots, strict=True)
            ),
        )
    )


def _product_source(output: str) -> str:
    graph = Graph()
    left, right = graph.variable("left"), graph.variable("right")
    value = left * right
    coordinate = graph.variable("coordinate")
    direction = graph.variable("direction")
    gradient = graph.differentiate(
        value,
        coordinate,
        {
            "left": graph.variable("left_gradient"),
            "right": graph.variable("right_gradient"),
        },
    )
    leaves = {name: graph.variable(name + "_directional") for name in ("left", "right")}
    leaves.update(
        left_gradient=graph.variable("left_jvp"),
        right_gradient=graph.variable("right_jvp"),
    )
    directional = graph.differentiate(value, direction, leaves)
    jvp = graph.differentiate(gradient, direction, leaves)
    variables = {
        "left": "left.value",
        "right": "right.value",
        "left_directional": "left.directional",
        "right_directional": "right.directional",
        "left_gradient": "left.gradient[index]",
        "right_gradient": "right.gradient[index]",
        "left_jvp": "left.jvp[index]",
        "right_jvp": "right.jvp[index]",
    }
    head = _jet_function(
        graph,
        (value, directional),
        variables,
        ("result.value", "result.directional"),
        output=output,
        fields=("value", "directional"),
    )
    body = _jet_function(
        graph,
        (gradient, jvp),
        variables,
        ("result.gradient[index]", "result.jvp[index]"),
        output=output,
        fields=("gradient", "jvp"),
    )
    return f"""
template<unsigned Dimension> GQC_FACTOR_PRODUCT_FUNCTION Jet<Dimension> multiply(
    const Jet<Dimension>& left, const Jet<Dimension>& right) {{
  Jet<Dimension> result{{}};
  {{ {head} }}
  for (unsigned index=0; index<Dimension; ++index) {{ {body} }}
  return result;
}}
"""


def _cuda_sp_product_source() -> str:
    """Lower a separable Hermite term without materializing four 12-lane jets.

    Differentiate the coefficient and Coulomb factors in their native spaces.
    The affine Coulomb incidence is constant within a primitive, so lift its
    three derivative lanes to four centers once, after their Hermite sum.
    """
    graph = Graph()
    axes = ("axis_x", "axis_y", "axis_z")
    leaves = (*axes, "state")
    primal = {leaf: graph.variable(leaf) for leaf in leaves}
    motion = {leaf: graph.variable(leaf + "_motion") for leaf in leaves}
    variables = {leaf: f"{leaf}.value" for leaf in leaves}
    variables.update({leaf + "_motion": f"{leaf}.directional" for leaf in leaves})
    direction = graph.variable("direction")
    coordinate = graph.variable("coordinate")
    coefficient = (primal["axis_x"] * primal["axis_y"]) * primal["axis_z"]
    value = coefficient * primal["state"]
    head = _jet_function(
        graph,
        (
            value,
            graph.differentiate(value, direction, motion),
            coefficient,
            graph.differentiate(coefficient, direction, motion),
        ),
        variables,
        ("value", "directional", "product_value", "product_directional"),
    )
    state_gradient = graph.variable("state_gradient")
    product_value = graph.variable("product_value")
    state_partial = graph.differentiate(
        product_value * primal["state"], coordinate, {"state": state_gradient}
    )
    state_body = _jet_function(
        graph,
        (
            state_partial,
            graph.differentiate(
                state_partial,
                direction,
                {
                    "product_value": graph.variable("product_directional"),
                    "state_gradient": graph.variable("state_jvp"),
                },
            ),
        ),
        {
            "product_value": "product_value",
            "product_directional": "product_directional",
            "state_gradient": "state.gradient[axis]",
            "state_jvp": "state.jvp[axis]",
        },
        ("gradient", "jvp"),
    )
    coefficient_bodies = []
    for axis, leaf in enumerate(axes):
        # The transpose incidence of an axis jet is zero off that axis.
        # Hoist the derivative weight (and its motion) out of the center loop.
        weight = graph.differentiate(value, coordinate, {leaf: graph.constant(1)})
        weight_body = _jet_function(
            graph,
            (weight, graph.differentiate(weight, direction, motion)),
            variables,
            ("weight", "weight_motion"),
        )
        lane = graph.variable("coefficient_gradient")
        partial = graph.variable("weight") * lane
        lane_body = _jet_function(
            graph,
            (
                partial,
                graph.differentiate(
                    partial,
                    direction,
                    {
                        "weight": graph.variable("weight_motion"),
                        "coefficient_gradient": graph.variable("coefficient_jvp"),
                    },
                ),
            ),
            {
                "weight": "weight",
                "weight_motion": "weight_motion",
                "coefficient_gradient": f"{leaf}.gradient[center]",
                "coefficient_jvp": f"{leaf}.jvp[center]",
            },
            ("gradient", "jvp"),
        )
        coefficient_bodies.append(f"""  if (degree[{axis}]) {{
    double weight, weight_motion;
    {weight_body}
    for (unsigned center=0; center<4; ++center) {{
      double gradient, jvp;
      {lane_body}
      result.gradient[3*center+{axis}]+=gradient;
      result.jvp[3*center+{axis}]+=jvp;
    }}
  }}""")
    coefficients = "\n".join(coefficient_bodies)
    return f"""
__device__ __forceinline__ void accumulate_axis_product(Jet<12>& result, Jet<3>& separation,
    const Jet<4>& axis_x, const Jet<4>& axis_y, const Jet<4>& axis_z,
    const Jet<3>& state, const unsigned* degree) {{
  double value, directional, product_value, product_directional;
  {{ {head} }}
  result.value+=value; result.directional+=directional;
  for (unsigned axis=0; axis<3; ++axis) {{
    double gradient, jvp;
    {state_body}
    separation.gradient[axis]+=gradient;
    separation.jvp[axis]+=jvp;
  }}
  // An axis with no Cartesian power has a constant unit Hermite coefficient.
{coefficients}
}}
"""


def _cuda_sp_pair_product_source() -> str:
    """Accumulate s/p pair jets without expanding their disjoint center supports."""
    graph = Graph()
    left, right = graph.variable("left"), graph.variable("right")
    value = left * right
    direction, coordinate = graph.variable("direction"), graph.variable("coordinate")
    motion = {name: graph.variable(name + "_directional") for name in ("left", "right")}
    roots = [value, graph.differentiate(value, direction, motion)]
    targets = ["value", "directional"]
    variables = {name: f"{name}.value" for name in ("left", "right")}
    variables.update({name + "_directional": f"{name}.directional" for name in motion})
    for name in ("left", "right"):
        gradient_name = name + "_gradient"
        tangent = graph.variable(gradient_name)
        gradient = graph.differentiate(value, coordinate, {name: tangent})
        roots.extend(
            (
                gradient,
                graph.differentiate(
                    gradient,
                    direction,
                    {**motion, gradient_name: graph.variable(name + "_jvp")},
                ),
            )
        )
        targets.extend((gradient_name, name + "_jvp"))
        variables[gradient_name] = f"{name}.gradient[center]"
        variables[name + "_jvp"] = f"{name}.jvp[center]"
    head = _jet_function(graph, tuple(roots[:2]), variables, tuple(targets[:2]))
    body = _jet_function(graph, tuple(roots[2:]), variables, tuple(targets[2:]))
    return f"""
__device__ __forceinline__ void accumulate_pair_product(Jet<4>& target,
    const Jet<2>& left, const Jet<2>& right, double sign) {{
  double value, directional;
  {{ {head} }}
  target.value+=sign*value;
  target.directional+=sign*directional;
  for (unsigned center=0; center<2; ++center) {{
    double left_gradient, left_jvp, right_gradient, right_jvp;
    {body}
    target.gradient[center]+=sign*left_gradient;
    target.gradient[center+2]+=sign*right_gradient;
    target.jvp[center]+=sign*left_jvp;
    target.jvp[center+2]+=sign*right_jvp;
  }}
}}
"""


def _cuda_prepared_low_degree_source() -> str:
    """Lower up to three p factors in their sparse affine coordinates.

    Each p factor depends on one pair shift and the common product separation.
    Specialize the zero pattern in a permuted Cartesian frame, then lift the
    derivatives to physical centers once, without runtime Hermite jet arrays.
    """
    algebra = build_coulomb_derivative_algebra(3)
    graph = algebra.graph
    states = dict(algebra.roots)
    differentiate = _eri_differentiate(graph, 5)
    shifts = tuple(graph.variable(f"shift_{slot}") for slot in range(3))
    first_inverse = graph.variable("first_inverse")
    second_inverse = graph.variable("second_inverse")
    variables = {
        "rho": "geometry.rho",
        "first_product_scale": "1.0",
        "first_inverse": "first_inverse",
        "second_inverse": "second_inverse",
        **{f"shift_{slot}": f"shift[{slot}]" for slot in range(3)},
        **{
            f"difference_{axis}": f"difference[{index}]"
            for index, axis in enumerate("xyz")
        },
        **{f"boys_{order}": f"geometry.boys[{order}]" for order in range(6)},
        **{f"motion_{slot}": f"motion[{slot}]" for slot in range(6)},
    }
    motions = tuple(graph.variable(f"motion_{slot}") for slot in range(6))

    def partials(root: Expr, degree: int) -> tuple[Expr, ...]:
        return (
            *(differentiate(root, 0, axis) for axis in "xyz"),
            *(graph.differentiate(root, shift, {}) for shift in shifts[:degree]),
        )

    def directional(root: Expr, degree: int) -> Expr:
        return graph.sum(
            motion * derivative
            for motion, derivative in zip(
                motions[: 3 + degree], partials(root, degree), strict=True
            )
        )

    def signed_expansion(quantums: tuple, inverse: Expr, name: str) -> tuple:
        # The signed inverse incorporates the ket parity for each selected
        # Hermite derivative, but a same-axis covariance has no ket sign.
        if len(quantums) != 2 or quantums[0][0] != quantums[1][0]:
            return _pair_expansion(graph, quantums, inverse)
        unsigned_inverse = graph.variable(f"unsigned_{name}")
        sign = graph.variable(f"sign_{name}")
        variables[f"unsigned_{name}"] = f"fabs({name})"
        variables[f"sign_{name}"] = f"({name} < 0 ? -1.0 : 1.0)"
        return tuple(
            (orders, coefficient * sign if sum(orders) % 2 else coefficient)
            for orders, coefficient in _pair_expansion(
                graph, quantums, unsigned_inverse
            )
        )

    functions = []
    for name, axes, split in (
        ("scalar", (), 0),
        ("single", (0,), 1),
        ("same_pair", (0, 0), 2),
        ("different_pairs", (0, 0), 1),
        ("different_axes", (0, 1), 1),
        ("triple_parallel", (0, 0, 0), 2),
        ("triple_parallel_pair", (0, 0, 1), 2),
        ("triple_parallel_cross", (0, 1, 0), 2),
        ("triple_distinct", (0, 1, 2), 2),
    ):
        degree = len(axes)
        factors = tuple(zip(axes, shifts[:degree], strict=True))
        first = signed_expansion(factors[:split], first_inverse, "first_inverse")
        second = signed_expansion(factors[split:], second_inverse, "second_inverse")
        value = graph.sum(
            left
            * right
            * states[
                tuple(
                    bra_order + ket_order
                    for bra_order, ket_order in zip(
                        left_orders, right_orders, strict=True
                    )
                )
            ]
            for left_orders, left in first
            for right_orders, right in second
        )

        gradient = partials(value, degree)
        body = _jet_function(
            graph,
            (
                value,
                directional(value, degree),
                *gradient,
                *(directional(root, degree) for root in gradient),
            ),
            variables,
            (
                "result.value",
                "result.directional",
                *(f"result.gradient[{slot}]" for slot in range(3 + degree)),
                *(f"result.jvp[{slot}]" for slot in range(3 + degree)),
            ),
        )
        functions.append(f"""
__device__ __forceinline__ Jet<{max(5, 3 + degree)}> prepared_{name}(const Geometry& geometry,
    const double* difference, [[maybe_unused]] const double* shift, const double* motion,
    [[maybe_unused]] double first_inverse, [[maybe_unused]] double second_inverse) {{
  Jet<{max(5, 3 + degree)}> result{{}};
  {body}
  return result;
}}
""")
    return (
        "".join(functions)
        + r"""
template<unsigned degree, unsigned dimension>
__device__ __forceinline__ void lift_prepared_partials(const Geometry& geometry,
    const double* direction, const unsigned* slots, const unsigned* axes,
    unsigned axis0, unsigned next, const Jet<dimension>& partials, double* output) {
  const auto prefactor=gaussian_prefactor(geometry,direction);
  output[0]=prefactor.value*partials.value;
#pragma unroll
  for (unsigned center=0; center<4; ++center) {
    const auto scale=geometry.product_scales[center];
    const auto separation_scale=center<2 ? scale : -scale;
#pragma unroll
    for (unsigned axis=0; axis<3; ++axis) {
      const unsigned index=3*center+axis;
      const unsigned separation_axis=axis==axis0 ? 0 : axis==next ? 1 : 2;
      auto gradient=separation_scale*partials.gradient[separation_axis];
      auto jvp=separation_scale*partials.jvp[separation_axis];
#pragma unroll
      for (unsigned factor=0; factor<degree; ++factor) {
        const auto incidence=axis==axes[factor] && (center&2)==(slots[factor]&2)
            ? scale-(center==slots[factor]) : 0.0;
        gradient+=incidence*partials.gradient[3+factor];
        jvp+=incidence*partials.jvp[3+factor];
      }
      output[1+index]=prefactor.value*gradient+prefactor.gradient[index]*partials.value;
      output[13+index]=prefactor.value*jvp+prefactor.directional*gradient+
          prefactor.gradient[index]*partials.directional+prefactor.jvp[index]*partials.value;
    }
  }
}

__device__ __forceinline__ bool prepared_scalar_jet(const Geometry& geometry,
    const double* direction, double* output) {
  double motion[3];
#pragma unroll
  for (unsigned axis=0; axis<3; ++axis) {
    motion[axis]=(geometry.product_scales[0]*direction[axis]+
                  geometry.product_scales[1]*direction[3+axis])-
                 (geometry.product_scales[2]*direction[6+axis]+
                  geometry.product_scales[3]*direction[9+axis]);
  }
  const auto partials=prepared_scalar(geometry,geometry.difference,nullptr,motion,0,0);
  lift_prepared_partials<0>(geometry,direction,nullptr,nullptr,0,1,partials,output);
  return true;
}

template<unsigned degree>
__device__ __noinline__ bool prepared_low_degree(const Geometry& geometry, const unsigned* angular,
    unsigned component, const double* direction, double* output) {
  static_assert(degree>=1 && degree<=3);
  // Component indices are in xyz order, with the rightmost p shell changing
  // fastest. Fixed scalar slots keep this dispatch independent of the radial count.
  unsigned slots[degree]{}, axes[degree]{}, count=0, extent=1;
  for (unsigned slot=0; slot<4; ++slot) if (angular[slot]) {
    if (count==degree) return false;
    slots[count++]=slot;
    extent*=3;
  }
  if (count!=degree || component>=extent) return false;
#pragma unroll
  for (unsigned factor=degree; factor-- > 0;) {
    axes[factor]=component%3;
    component/=3;
  }
  if constexpr (degree==3) {
    const auto swap_factors=[&](unsigned lhs,unsigned rhs) {
      const auto slot=slots[lhs], axis=axes[lhs];
      slots[lhs]=slots[rhs]; axes[lhs]=axes[rhs];
      slots[rhs]=slot; axes[rhs]=axis;
    };
    // Put the two factors from the same pair first, including when both are
    // in the ket. Of two distinct axes, prefer the one shared by the single p.
    if ((slots[0]&2)!=(slots[1]&2)) {
      swap_factors(0,1);
      swap_factors(1,2);
    }
    if (axes[0]!=axes[1] && axes[1]==axes[2]) swap_factors(0,1);
  }
  const unsigned axis0=axes[0];
  unsigned next=(axis0+1)%3;
  if constexpr (degree>=2) if (axes[1]!=axis0) next=axes[1];
  if constexpr (degree==3) if (axes[1]==axis0 && axes[2]!=axis0) next=axes[2];
  const unsigned last=3-axis0-next;
  const double difference[]={geometry.difference[axis0], geometry.difference[next],
                             geometry.difference[last]};
  const auto signed_inverse=[&](unsigned slot) {
    return slot<2 ? geometry.inverse_two_p : -geometry.inverse_two_q;
  };
  const auto pair_motion=[&](unsigned pair, unsigned axis) {
    return geometry.product_scales[pair]*direction[3*pair+axis]+
           geometry.product_scales[pair+1]*direction[3*(pair+1)+axis];
  };
  const auto separation=[&](unsigned axis) { return pair_motion(0,axis)-pair_motion(2,axis); };
  const auto shift_motion=[&](unsigned slot, unsigned axis) {
    return pair_motion(slot&2,axis)-direction[3*slot+axis];
  };
  double shift[degree], motion[3+degree]={separation(axis0),separation(next),separation(last)};
#pragma unroll
  for (unsigned factor=0; factor<degree; ++factor) {
    shift[factor]=geometry.shifts[slots[factor]][axes[factor]];
    motion[3+factor]=shift_motion(slots[factor],axes[factor]);
  }
  Jet<(degree==3 ? 6 : 5)> partials;
  const auto first_inverse=signed_inverse(slots[0]);
  if constexpr (degree==1) {
    partials=prepared_single(geometry,difference,shift,motion,first_inverse,0);
  } else if constexpr (degree==2) {
    const auto second_inverse=signed_inverse(slots[1]);
    if (axis0!=axes[1])
      partials=prepared_different_axes(geometry,difference,shift,motion,first_inverse,second_inverse);
    else if ((slots[0]&2)==(slots[1]&2))
      partials=prepared_same_pair(geometry,difference,shift,motion,first_inverse,0);
    else
      partials=prepared_different_pairs(geometry,difference,shift,motion,first_inverse,second_inverse);
  } else {
    const auto second_inverse=signed_inverse(slots[2]);
    if (axis0==axes[1] && axis0==axes[2])
      partials=prepared_triple_parallel(geometry,difference,shift,motion,first_inverse,second_inverse);
    else if (axis0==axes[1])
      partials=prepared_triple_parallel_pair(geometry,difference,shift,motion,first_inverse,second_inverse);
    else if (axis0==axes[2])
      partials=prepared_triple_parallel_cross(geometry,difference,shift,motion,first_inverse,second_inverse);
    else
      partials=prepared_triple_distinct(geometry,difference,shift,motion,first_inverse,second_inverse);
  }
  lift_prepared_partials<degree>(geometry,direction,slots,axes,axis0,next,partials,output);
  return true;
}
"""
    )


def _pair_source(output: str) -> str:
    direction_attribute = "[[maybe_unused]] " if output in ("first", "value") else ""
    scale_attribute = "[[maybe_unused]] " if output == "value" else ""
    graph = Graph()
    differentiate = _eri_differentiate(graph, 0)
    variables = {
        "inverse_two_p": "inverse_two",
        "pa_x": "left_shift",
        "pb_x": "right_shift",
        "first_product_scale": "left_scale",
        "second_product_scale": "right_scale",
        "direction_left": "left_direction",
        "direction_right": "right_direction",
    }
    cases = []
    # An auxiliary g shell has a unit-Gaussian partner, never another g shell
    # in the same pair. Physical orbital pairs remain bounded by f/f.
    powers = (*product(range(4), repeat=2), (4, 0))
    for left_power, right_power in powers:
        quantums = (
            *((0, graph.variable("pa_x")) for _ in range(left_power)),
            *((0, graph.variable("pb_x")) for _ in range(right_power)),
        )
        coefficients = defaultdict(list)
        for orders, coefficient in _pair_expansion(
            graph, quantums, graph.variable("inverse_two_p")
        ):
            coefficients[orders[0]].append(coefficient)
        roots, targets = [], []
        for order in range(left_power + right_power + 1):
            value = graph.sum(coefficients[order])
            gradient = tuple(differentiate(value, center, "x") for center in range(2))
            directional = graph.sum(
                graph.variable(f"direction_{label}") * gradient[center]
                for center, label in enumerate(("left", "right"))
            )
            jvp = tuple(
                graph.sum(
                    graph.variable(f"direction_{label}")
                    * differentiate(root, center, "x")
                    for center, label in enumerate(("left", "right"))
                )
                for root in gradient
            )
            roots.extend((value, directional, *gradient, *jvp))
            targets.extend(
                (
                    f"output[{order}].value",
                    f"output[{order}].directional",
                    *(f"output[{order}].gradient[{center}]" for center in range(2)),
                    *(f"output[{order}].jvp[{center}]" for center in range(2)),
                )
            )
        cases.append(
            f"case {5 * left_power + right_power}: "
            f"if constexpr (!sp_only || {max(left_power, right_power)} <= 1) {{\n"
            + _jet_function(
                graph,
                tuple(roots),
                variables,
                tuple(targets),
                output=output,
                fields=("value", "directional", "gradient", "gradient", "jvp", "jvp")
                * (left_power + right_power + 1),
            )
            + "\nreturn true; } else return false;"
        )
    return (
        f"""
template<bool sp_only=false>
GQC_FACTOR_FUNCTION bool pair_coefficients(unsigned left_power, unsigned right_power,
    double inverse_two, double left_shift, double right_shift,
    {scale_attribute}double left_scale, {scale_attribute}double right_scale, {direction_attribute}double left_direction,
    {direction_attribute}double right_direction, Jet<2>* output) {{
  switch (5*left_power+right_power) {{
"""
        + "\n".join(cases)
        + "\n} return false; }\n"
    )


def _coulomb_source(output: str) -> str:
    direction_attribute = "[[maybe_unused]] " if output in ("first", "value") else ""
    algebra = build_coulomb_derivative_algebra(12)
    graph = algebra.graph
    differentiate = _eri_differentiate(graph, 14)
    variables = {
        "rho": "geometry.rho",
        "first_product_scale": "1.0",
        **{
            f"difference_{axis}": f"geometry.difference[{index}]"
            for index, axis in enumerate("xyz")
        },
        **{f"boys_{order}": f"geometry.boys[{order}]" for order in range(15)},
        **{
            f"direction_{axis}": f"direction[{index}]"
            for index, axis in enumerate("xyz")
        },
    }
    cases, functions = [], []
    for orders, value in algebra.roots:
        gradient = tuple(differentiate(value, 0, axis) for axis in "xyz")
        directional = graph.sum(
            root * graph.variable(f"direction_{axis}")
            for axis, root in zip("xyz", gradient, strict=True)
        )
        jvp = tuple(
            graph.sum(
                graph.variable(f"direction_{axis}") * differentiate(root, 0, axis)
                for axis in "xyz"
            )
            for root in gradient
        )
        roots = (value, directional, *gradient, *jvp)
        targets = (
            "result.value",
            "result.directional",
            *(f"result.gradient[{axis}]" for axis in range(3)),
            *(f"result.jvp[{axis}]" for axis in range(3)),
        )
        code = 169 * orders[0] + 13 * orders[1] + orders[2]
        name = "coulomb_" + "_".join(map(str, orders))
        functions.append(
            f"GQC_FACTOR_FUNCTION Jet<3> {name}(const Geometry& geometry, {direction_attribute}const double* direction) {{\nJet<3> result{{}};\n"
            + _jet_function(
                graph,
                roots,
                variables,
                targets,
                output=output,
                fields=("value", "directional", *("gradient",) * 3, *("jvp",) * 3),
            )
            + "\nreturn result; }\n"
        )
        cases.append(
            f"case {code}: if constexpr (maximum_order >= {sum(orders)}) "
            f"return {name}(geometry,direction); else return {{}};"
        )
    return (
        "\n".join(functions)
        + """
template<unsigned maximum_order=12>
GQC_FACTOR_FUNCTION Jet<3> coulomb_state(unsigned first, unsigned second, unsigned third,
    const Geometry& geometry, const double* direction) {
  Jet<3> result{};
  switch (169*first+13*second+third) {
"""
        + "\n".join(cases)
        + "\n} return result; }\n"
    )


def _prefactor_source(output: str) -> str:
    direction_attribute = "[[maybe_unused]] " if output in ("first", "value") else ""
    graph = Graph()
    value = graph.variable("prefactor")
    differentiate = _eri_differentiate(graph, 0)
    gradient = tuple(
        differentiate(value, center, axis) for center in range(4) for axis in "xyz"
    )
    directional = graph.sum(
        root * graph.variable(f"direction_{index}")
        for index, root in enumerate(gradient)
    )
    jvp = tuple(
        graph.sum(
            graph.variable(f"direction_{3 * center + axis_index}")
            * differentiate(root, center, axis)
            for center in range(4)
            for axis_index, axis in enumerate("xyz")
        )
        for root in gradient
    )
    variables = {
        "prefactor": "geometry.prefactor",
        "inverse_two_p": "geometry.inverse_two_p",
        "inverse_two_q": "geometry.inverse_two_q",
        **{
            f"{label}_product_scale": f"geometry.product_scales[{center}]"
            for center, label in enumerate(("first", "second", "third", "fourth"))
        },
        **{
            f"decay_{label}_{axis}": f"geometry.decay[{center}][{axis_index}]"
            for center, label in enumerate(("first", "second", "third", "fourth"))
            for axis_index, axis in enumerate("xyz")
        },
        **{f"direction_{index}": f"direction[{index}]" for index in range(12)},
    }
    roots = (value, directional, *gradient, *jvp)
    targets = (
        "result.value",
        "result.directional",
        *(f"result.gradient[{index}]" for index in range(12)),
        *(f"result.jvp[{index}]" for index in range(12)),
    )
    return (
        f"""
GQC_FACTOR_FUNCTION Jet<12> gaussian_prefactor(const Geometry& geometry, {direction_attribute}const double* direction) {{
  Jet<12> result{{}};
"""
        + _jet_function(
            graph,
            roots,
            variables,
            targets,
            output=output,
            fields=("value", "directional", *("gradient",) * 12, *("jvp",) * 12),
        )
        + "\nreturn result; }\n"
    )


@cache
def emit_factorized_second(*, backend: str = "cpu", output: str = "full") -> str:
    """Emit one finite reusable f/g factor library, with no runtime compilation.

    ``primitive`` retains every physical center and returns value, gradient and
    gradient-JVP. Record scaling and ordered radial contraction remain outside
    this primitive. Family codes are ERI=0, three-center=1 and metric=2; radial
    codes are full=0, long=1 and short=2. CUDA obtains the positive range
    parameter from the authenticated launch metadata.
    CPU callers may borrow a full-range four-center geometry bound from the
    same immutable record. Omitting it retains the original canonical binder.
    ``output`` selects full value/gradient/gradient-JVP, value/gradient only,
    value/scalar-directional, or value-only packing. Selection is independent of motion:
    a zero direction does not change the requested operation or its ABI.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("factorized second-order backend must be cpu or cuda")
    if output not in ("full", "first", "directional", "value"):
        raise ValueError("factorized output must be full, first, directional or value")
    qualifier = "static" if backend == "cpu" else "__device__ __noinline__"
    # Use the positive recurrence for the bounded full-range s/p CUDA leaf.
    # Other CUDA angular classes and every LR/SR path keep their current policy.
    geometry_arguments = (
        ("Geometry,14,0,true", "Geometry,14,8,true", "Geometry,14,10,true")
        if backend == "cpu"
        else (
            "Geometry,full_eri_sp ? sp_degree+2 : 14,0,full_eri_sp",
            "Geometry,full_eri_sp ? sp_degree+2 : 14,8",
            "Geometry,full_eri_sp ? sp_degree+2 : 14,10",
        )
    )
    geometry_target = "geometry" if backend == "cpu" else "local_geometry"
    components = []
    for angular in range(5):
        entries = [
            "{" + ",".join(str(component.count(axis)) for axis in "xyz") + "}"
            for component in cartesian_components(angular)
        ]
        components.append("{" + ",".join(entries) + "}")
    return (
        '#include "integrals/eri_geometry.hpp"\n#include <cmath>\n'
        f"#define GQC_FACTOR_FUNCTION {qualifier}\n"
        + (
            "#define GQC_FACTOR_PRODUCT_FUNCTION __device__ __forceinline__\n"
            if backend == "cuda"
            else "#define GQC_FACTOR_PRODUCT_FUNCTION GQC_FACTOR_FUNCTION\n"
        )
        + "namespace generativeqc::scf::factorized_second {\n"
        f"constexpr bool emit_gradient={'true' if output in ('full', 'first') else 'false'};\n"
        f"constexpr bool emit_jvp={'true' if output == 'full' else 'false'};\n"
        f"constexpr bool emit_directional={'true' if output in ('full', 'directional') else 'false'};\n"
        "template<unsigned Dimension> struct Jet { double value{}, directional{}; double gradient[Dimension]{}, jvp[Dimension]{}; };\n"
        + (
            "struct Geometry { double inverse_two_p, inverse_two_q, rho; double difference[3], shifts[4][3], product_scales[4], decay[4][3]; double prefactor, boys[15]; };\n"
            if backend == "cpu"
            else "using Geometry=generativeqc::dft::hessian_aot::FactorizedPairGeometry;\n"
        )
        + _product_source(output)
        + (
            _cuda_sp_product_source() + _cuda_sp_pair_product_source()
            if backend == "cuda" and output == "full"
            else ""
        )
        + _pair_source(output)
        + _coulomb_source(output)
        + _prefactor_source(output)
        + (
            _cuda_prepared_low_degree_source()
            if backend == "cuda" and output == "full"
            else ""
        )
        + (
            "\ntemplate<unsigned sp_degree=5, unsigned FixedFamily=0>\n"
            if backend == "cpu"
            else "\ntemplate<unsigned sp_degree=5, bool Prepared=false, unsigned FixedFamily=0>\n"
        )
        + "GQC_FACTOR_FUNCTION bool primitive(unsigned input_family, unsigned input_radial, const unsigned* angular, unsigned component, const double* record, const double* direction, double* output"
        + (
            ", const Geometry* prepared_geometry=nullptr"
            if backend == "cpu"
            else ", double omega, const Geometry* prepared_geometry=nullptr"
        )
        + ") {\n"
        + "  constexpr unsigned powers[5][15][3]={"
        + ",".join(components)
        + "};\n"
        + (
            "  constexpr bool prepared_record=false;\n"
            if backend == "cpu"
            else "  constexpr bool prepared_record=Prepared;\n"
        )
        + r"""
  constexpr bool full_eri_sp=sp_degree<=4;
  const unsigned family=full_eri_sp ? FixedFamily : input_family;
  const unsigned radial=full_eri_sp ? 0 : input_radial;
  if (family>2 || radial>2 || (family && radial==2) || !angular || !record || !direction || !output) return false;
  if ((full_eri_sp && (input_family!=FixedFamily || input_radial)) ||
      (prepared_record && !prepared_geometry)) return false;
  const unsigned centers=family==0 ? 4 : family==1 ? 3 : 2;
  unsigned orders[4]{}, incidence[4]{};
  if (family==0) {
    for (unsigned center=0; center<4; ++center) { orders[center]=angular[center]; incidence[center]=center; }
  } else if (family==1) {
    for (unsigned center=0; center<3; ++center) { orders[center]=angular[center]; incidence[center]=center; }
    incidence[3]=2;
  } else {
    orders[0]=angular[0]; orders[2]=angular[1]; incidence[2]=incidence[3]=1;
  }
  for (unsigned center=0; center<centers; ++center)
    if (angular[center]>(full_eri_sp ? 1u : family && (family==2 || center==2) ? 4u : 3u) ||
        (!prepared_record && (!std::isfinite(record[center]) || record[center]<=0))) return false;
  double exponents[4]{}, coordinates[12]{}, motion[12]{};
  for (unsigned center=0; center<4; ++center) {
    const bool unit=(family==1 && center==3) || (family==2 && (center==1 || center==3));
    // The prepared binder already authenticated the four physical inputs.
    // Only the changing motion has to be read here.
    if constexpr (!prepared_record) if (!unit) exponents[center]=record[incidence[center]];
    for (unsigned axis=0; axis<3; ++axis) {
      if constexpr (!prepared_record) {
        coordinates[3*center+axis]=record[centers+3*incidence[center]+axis];
        if (!std::isfinite(coordinates[3*center+axis])) return false;
      }
      if (!unit) motion[3*center+axis]=direction[3*incidence[center]+axis];
      if (!std::isfinite(motion[3*center+axis])) return false;
    }
  }
  unsigned quantums[4][3]{}, sizes[4]{};
  if constexpr (sp_degree==0) {
    if (component) return false;
  } else {
    for (unsigned center=0; center<4; ++center) sizes[center]=(orders[center]+1)*(orders[center]+2)/2;
    unsigned remaining=component;
    for (int center=3; center>=0; --center) {
      const unsigned selected=remaining%sizes[center]; remaining/=sizes[center];
      for (unsigned axis=0; axis<3; ++axis) quantums[center][axis]=powers[orders[center]][selected][axis];
    }
    if (remaining) return false;
  }
  unsigned maximum_order=2;
  for (unsigned center=0; center<4; ++center) maximum_order+=orders[center];
  if constexpr (full_eri_sp) {
    // The authenticated shell total bounds all Cartesian component states.
    if (maximum_order!=sp_degree+2) return false;
    maximum_order=sp_degree+2;
  }
"""
        + (
            "  Geometry geometry{};\n"
            if backend == "cpu"
            else "  Geometry local_geometry{};\n"
            "  static_assert(!Prepared || (full_eri_sp && FixedFamily==0));\n"
            "  const auto& geometry=Prepared ? *prepared_geometry : local_geometry;\n"
        )
        + r"""
  const auto range=radial==0 ? generativeqc::integrals::CoulombRange::Full : radial==1 ? generativeqc::integrals::CoulombRange::Long : generativeqc::integrals::CoulombRange::Short;
  bool valid=false;
"""
        + ("  const double omega=radial==0 ? 0 : 0.3;\n" if backend == "cpu" else "")
        + (
            """  if (prepared_geometry) {
    if (family!=0 || radial!=0) return false;
    geometry=*prepared_geometry;
    valid=true;
  } else {
"""
            if backend == "cpu"
            else "  if constexpr (Prepared) valid=true;\n  else {\n"
        )
        + f"""  if (family==0) valid=generativeqc::integrals::make_eri_geometry<{geometry_arguments[0]}>(exponents,coordinates,maximum_order,range,omega,{geometry_target});
  if (family==1) valid=generativeqc::integrals::make_eri_geometry<{geometry_arguments[1]}>(exponents,coordinates,maximum_order,range,omega,{geometry_target});
  if (family==2) valid=generativeqc::integrals::make_eri_geometry<{geometry_arguments[2]}>(exponents,coordinates,maximum_order,range,omega,{geometry_target});
"""
        + "  }\n"
        + r"""  if (!valid) return false;
  // Four s/p orbitals need at most quartic axis states and quadratic pair
  // states. A separate instantiation bounds per-thread scratch and branches.
  constexpr unsigned max_axis_order=full_eri_sp ? sp_degree : 12;
  // Each orbital contributes at most one Cartesian power, on only one axis.
  // Consequently the three axis degrees sum to sp_degree; store the three
  // constant coefficients plus only that many higher coefficients.
  Jet<4> coefficients[full_eri_sp ? sp_degree+3 : 3*(max_axis_order+1)]{};
  unsigned axis_orders[3]{}, axis_offsets[3]{};
  if constexpr (sp_degree==0) {
    // Admission above proved all s shells. Each Hermite pair polynomial
    // consists only of its constant unit coefficient and has no motion.
    for (unsigned axis=0; axis<3; ++axis) {
      axis_offsets[axis]=axis;
      coefficients[axis].value=1;
    }
  } else for (unsigned axis=0; axis<3; ++axis) {
    Jet<2> first[full_eri_sp ? 3 : 7]{}, second[full_eri_sp ? 3 : 7]{};
    if (!pair_coefficients<full_eri_sp>(quantums[0][axis],quantums[1][axis],geometry.inverse_two_p,geometry.shifts[0][axis],geometry.shifts[1][axis],geometry.product_scales[0],geometry.product_scales[1],motion[axis],motion[3+axis],first) ||
        !pair_coefficients<full_eri_sp>(quantums[2][axis],quantums[3][axis],geometry.inverse_two_q,geometry.shifts[2][axis],geometry.shifts[3][axis],geometry.product_scales[2],geometry.product_scales[3],motion[6+axis],motion[9+axis],second)) return false;
    const unsigned first_order=quantums[0][axis]+quantums[1][axis], second_order=quantums[2][axis]+quantums[3][axis];
    axis_orders[axis]=first_order+second_order;
    if constexpr (full_eri_sp) {
      if (axis) axis_offsets[axis]=axis_offsets[axis-1]+axis_orders[axis-1]+1;
    } else axis_offsets[axis]=axis*(max_axis_order+1);
    for (unsigned left=0; left<=first_order; ++left) {
      for (unsigned right=0; right<=second_order; ++right) {
        auto& target=coefficients[axis_offsets[axis]+left+right];
        const double sign=right%2 ? -1 : 1;
"""
        + (
            r"""        if constexpr (full_eri_sp) {
          accumulate_pair_product(target,first[left],second[right],sign);
          continue;
        }
"""
            if backend == "cuda" and output == "full"
            else ""
        )
        + r"""
        Jet<4> first_jet{}, second_jet{};
        first_jet.value=first[left].value; first_jet.directional=first[left].directional;
        second_jet.value=second[right].value; second_jet.directional=second[right].directional;
        if constexpr (emit_gradient) for (unsigned center=0; center<2; ++center) {
          first_jet.gradient[center]=first[left].gradient[center]; first_jet.jvp[center]=first[left].jvp[center];
          second_jet.gradient[center+2]=second[right].gradient[center]; second_jet.jvp[center+2]=second[right].jvp[center];
        }
        const auto term=multiply(first_jet,second_jet);
        target.value+=sign*term.value; target.directional+=sign*term.directional;
        if constexpr (emit_gradient) for (unsigned center=0; center<4; ++center) { target.gradient[center]+=sign*term.gradient[center]; target.jvp[center]+=sign*term.jvp[center]; }
      }
    }
  }
  double difference_direction[3]{};
  for (unsigned center=0; center<4; ++center)
    for (unsigned axis=0; axis<3; ++axis) difference_direction[axis]+=(center<2 ? 1 : -1)*geometry.product_scales[center]*motion[3*center+axis];
  Jet<12> result{};
"""
        + ("  Jet<3> separation{};\n" if backend == "cuda" and output == "full" else "")
        + r"""  for (unsigned first=0; first<=axis_orders[0] && first<=max_axis_order; ++first)
    for (unsigned second=0; second<=axis_orders[1] && second<=max_axis_order; ++second)
      for (unsigned third=0; third<=axis_orders[2] && third<=max_axis_order; ++third) {
"""
        + (
            r"""        if constexpr (full_eri_sp) {
          const auto state=coulomb_state<sp_degree>(first,second,third,geometry,difference_direction);
          accumulate_axis_product(result,separation,coefficients[first],
              coefficients[axis_offsets[1]+second],coefficients[axis_offsets[2]+third],state,axis_orders);
          continue;
        }
"""
            if backend == "cuda" and output == "full"
            else ""
        )
        + r"""
        const unsigned selected[3]={first,second,third};
        Jet<12> axes[3]{};
        for (unsigned axis=0; axis<3; ++axis) {
          const auto& coefficient=coefficients[axis_offsets[axis]+selected[axis]];
          axes[axis].value=coefficient.value; axes[axis].directional=coefficient.directional;
          if constexpr (emit_gradient) for (unsigned center=0; center<4; ++center) { axes[axis].gradient[3*center+axis]=coefficient.gradient[center]; axes[axis].jvp[3*center+axis]=coefficient.jvp[center]; }
        }
        const auto state=coulomb_state<max_axis_order>(first,second,third,geometry,difference_direction);
        Jet<12> coulomb{}; coulomb.value=state.value; coulomb.directional=state.directional;
        if constexpr (emit_gradient) for (unsigned center=0; center<4; ++center)
          for (unsigned axis=0; axis<3; ++axis) {
            const double scale=(center<2 ? 1 : -1)*geometry.product_scales[center];
            coulomb.gradient[3*center+axis]=scale*state.gradient[axis]; coulomb.jvp[3*center+axis]=scale*state.jvp[axis];
          }
        const auto term=multiply(multiply(multiply(axes[0],axes[1]),axes[2]),coulomb);
        result.value+=term.value; result.directional+=term.directional;
        if constexpr (emit_gradient) for (unsigned index=0; index<12; ++index) { result.gradient[index]+=term.gradient[index]; result.jvp[index]+=term.jvp[index]; }
      }
"""
        + (
            r"""  if constexpr (full_eri_sp) for (unsigned center=0; center<4; ++center) {
    const double scale=(center<2 ? 1 : -1)*geometry.product_scales[center];
    for (unsigned axis=0; axis<3; ++axis) {
      result.gradient[3*center+axis]+=scale*separation.gradient[axis];
      result.jvp[3*center+axis]+=scale*separation.jvp[axis];
    }
  }
"""
            if backend == "cuda" and output == "full"
            else ""
        )
        + r"""  result=multiply(gaussian_prefactor(geometry,motion),result);
  output[0]=result.value;
"""
        + (
            "  constexpr unsigned output_size=1;\n"
            if output == "value"
            else r"""  constexpr unsigned derivative_blocks=emit_jvp ? 2 : 1;
  const unsigned output_size=emit_gradient ? 1+3*centers*derivative_blocks : 2;
  for (unsigned index=1; index<output_size; ++index) output[index]=0;
  if constexpr (!emit_gradient) output[1]=result.directional;
  if constexpr (emit_gradient) for (unsigned center=0; center<4; ++center)
    for (unsigned axis=0; axis<3; ++axis) {
      output[1+3*incidence[center]+axis]+=result.gradient[3*center+axis];
      if constexpr (emit_jvp) output[1+3*centers+3*incidence[center]+axis]+=result.jvp[3*center+axis];
    }
"""
        )
        + r"""  for (unsigned index=0; index<output_size; ++index) if (!std::isfinite(output[index])) return false;
  return true;
}
}  // namespace generativeqc::scf::factorized_second
#undef GQC_FACTOR_FUNCTION
#undef GQC_FACTOR_PRODUCT_FUNCTION
"""
    )


def emit_factorized_resident() -> str:
    """Bind the finite factor library to the same ordered radial contraction.

    Metadata is a bounded by-value kernel argument, not a wildcard capability.
    This launch owns no allocation, synchronization, runtime compiler or oracle.
    """
    return (
        '#include "dft/factorized_hessian_radial_pair.hpp"\n'
        + emit_factorized_second(backend="cuda")
        + r"""
#include "dft/factorized_hessian_integral.hpp"
#include "dft/factorized_hessian_radial_pair.hpp"
#include <cuda_runtime.h>
#include <limits>
namespace generativeqc::dft::hessian_aot {
namespace {
struct FactorizedSelection {
  unsigned family, radial, angular[4], components[16], component_count, centers, stride, cache_stride;
  double omega;
};
template<unsigned sp_degree, bool paired=false, bool Prepared=false, bool Indexed=false,
    unsigned FixedFamily=0>
__global__ void factorized_integral_kernel(FactorizedSelection selection, const double* records,
    size_t count, const long long* atoms, const double* molecular_direction,
    double* output, int* status, const FactorizedIntegralBatch* batches, long long frozen_atom,
    const FactorizedPairGeometry* prepared=nullptr, const double* window_records=nullptr,
    const FactorizedActiveComponent* tasks=nullptr) {
  static_assert(!Indexed || (paired && Prepared));
  unsigned batch_index=blockIdx.y, component=0;
  unsigned angular[4];
  for (unsigned slot=0; slot<4; ++slot) angular[slot]=selection.angular[slot];
  if constexpr (Indexed) {
    const auto task=tasks[blockIdx.x];
    batch_index=task.batch;
    component=task.selector&127;
    for (unsigned slot=0; slot<4; ++slot) angular[slot]=(task.selector>>(7+slot))&1;
  } else {
    if (blockIdx.x>=selection.component_count) return;
    component=selection.components[blockIdx.x];
  }
  const double* coordinates=nullptr;
  if (batches) {
    const auto& batch=batches[batch_index];
    records=batch.records; count=batch.count; atoms=batch.atoms; output=batch.output;
    coordinates=batch.coordinates;
  }
  if constexpr (!Indexed) if (!active_eri_centers(atoms,frozen_atom)) return;
  // Primitive producers write one row; adjacent output lanes then read the
  // same radial row without striding by a whole tile through shared banks.
  constexpr unsigned unindexed_producers=paired ? 128 : FixedFamily ? 32 : 64;
  const unsigned producers=Indexed ? blockDim.x : unindexed_producers;
  __shared__ double fixed_terms[Indexed ? 1 : unindexed_producers][25];
  extern __shared__ double indexed_terms[][25];
  auto* terms=Indexed ? indexed_terms : fixed_terms;
  if constexpr (paired) if (!count || count>producers) {
    if (threadIdx.x==0) atomicExch(status,1);
    return;
  }
  double direction[12]{}, result=0;
  const unsigned outputs=1+6*selection.centers;
  for (unsigned center=0; center<selection.centers; ++center)
    for (unsigned axis=0; axis<3; ++axis) direction[3*center+axis]=molecular_direction[3*atoms[center]+axis];
  for (size_t begin=0; begin<count;) {
    const size_t active=count-begin<producers ? count-begin : producers;
    bool valid=true;
    if (threadIdx.x<active) {
      double expanded[17];
      const double* record=records+(begin+threadIdx.x)*(coordinates ? selection.centers+1 : selection.stride);
      const double scale=record[coordinates ? selection.centers : selection.stride-1];
      if (!Prepared && coordinates) {
        for (unsigned center=0; center<selection.centers; ++center) {
          expanded[center]=record[center];
          for (unsigned axis=0; axis<3; ++axis)
            expanded[selection.centers+3*center+axis]=coordinates[3*atoms[center]+axis];
        }
        expanded[selection.stride-1]=record[selection.centers];
        record=expanded;
      }
      double values[25]{};
      const auto* geometry=Prepared ? prepared+(records-window_records)/5+begin+threadIdx.x : nullptr;
      if constexpr (Indexed && sp_degree==0) {
        valid=component==0 && !(angular[0]|angular[1]|angular[2]|angular[3]) &&
            generativeqc::scf::factorized_second::prepared_scalar_jet(*geometry,direction,values);
      } else if constexpr (Indexed && sp_degree<=3) {
        valid=generativeqc::scf::factorized_second::prepared_low_degree<sp_degree>(
            *geometry,angular,component,direction,values);
      } else {
        valid=generativeqc::scf::factorized_second::primitive<sp_degree,Prepared,FixedFamily>(selection.family,selection.radial,
            angular,component,record,direction,values,selection.omega,geometry);
      }
      for (unsigned index=0; index<outputs; ++index) terms[threadIdx.x][index]=scale*values[index];
    }
    if (__syncthreads_or(!valid)) {
      if (threadIdx.x==0) atomicExch(status,1);
      return;
    }
    if constexpr (paired) {
      // Two independent groups of lanes fold the original chunks. Their
      // rounded sums are merged in the same order as the separate kernel.
      const auto chunk=threadIdx.x/32;
      const auto field=threadIdx.x%32;
      double partial=0;
      if (chunk<2 && field<25)
        for (size_t radial=64*chunk; radial<active && radial<64*(chunk+1); ++radial)
          partial+=terms[radial][field];
      __syncthreads();
      if (chunk<2 && field<25) {
        if (!std::isfinite(partial)) atomicExch(status,2);
        terms[chunk][field]=partial;
      }
      __syncthreads();
      if (threadIdx.x<25) {
        const auto sum=0.0+terms[0][field]+(active>64 ? terms[1][field] : 0.0);
        output[field*selection.cache_stride+component]=sum;
      }
      return;
    }
    // Each derivative lane sums its own radial axis in the original order,
    // including across tile boundaries. There is no reassociating reduction.
    if (threadIdx.x<outputs)
      for (size_t radial=0; radial<active; ++radial) result+=terms[radial][threadIdx.x];
    __syncthreads();
    begin+=active;
  }
  if (threadIdx.x<outputs) {
    if (!std::isfinite(result)) atomicExch(status,2);
    output[blockIdx.x*outputs+threadIdx.x]=result;
  }
}
// DF uses the same bounded Cartesian jet as Direct, with the explicit
// unit-Gaussian embedding supplied by the family. One warp folds each radial
// chunk in the original order; larger contractions simply take more chunks.
template<unsigned Family>
void launch_fitted_sp(FactorizedSelection selection, dim3 grid, unsigned degree,
    const FactorizedIntegralBatch* batches, const double* direction, int* status,
    cudaStream_t stream) {
#define GQC_FITTED_DEGREE(degree) \
  case degree: factorized_integral_kernel<degree,false,false,false,Family><<<grid,32,0,stream>>>( \
      selection,nullptr,0,nullptr,direction,nullptr,status,batches,-1); break
  switch (degree) {
    GQC_FITTED_DEGREE(0);
    GQC_FITTED_DEGREE(1);
    GQC_FITTED_DEGREE(2);
    case 3:
      if constexpr (Family==1)
        factorized_integral_kernel<3,false,false,false,Family><<<grid,32,0,stream>>>(
            selection,nullptr,0,nullptr,direction,nullptr,status,batches,-1);
      break;
  }
#undef GQC_FITTED_DEGREE
}
}  // namespace
int launch_factorized_integral(unsigned family, unsigned radial, double omega, const unsigned* angular,
    const std::uint32_t* components, size_t component_count, const double* records, size_t count,
    const long long* atoms, const double* direction, double* output, int* status,
    cudaStream_t stream, const FactorizedIntegralBatch* batches, size_t batch_count,
    long long frozen_atom) noexcept {
  if (family>2 || radial>2 || (family && radial==2) || !angular || !components ||
      (radial==0 ? omega!=0 : !std::isfinite(omega) || omega<=0) ||
      !component_count || component_count>16 || !batch_count || batch_count>65535 ||
      (!batches && (!records || !count || !atoms || !output || batch_count!=1)) ||
      (frozen_atom>=0 && (!batches || family!=0)) ||
      !direction || !status) return static_cast<int>(cudaErrorInvalidValue);
  FactorizedSelection selection{};
  selection.family=family; selection.radial=radial; selection.omega=omega;
  selection.centers=family==0 ? 4 : family==1 ? 3 : 2;
  selection.stride=4*selection.centers+1;
  selection.component_count=static_cast<unsigned>(component_count);
  size_t size=1;
  bool full_eri_sp=radial==0;
  unsigned degree=0;
  for (unsigned center=0; center<selection.centers; ++center) {
    if (angular[center]>(family && (family==2 || center==2) ? 4u : 3u))
      return static_cast<int>(cudaErrorInvalidValue);
    selection.angular[center]=angular[center];
    full_eri_sp=full_eri_sp && angular[center]<=1;
    degree+=angular[center];
    size*=(angular[center]+1)*(angular[center]+2)/2;
  }
  if (count>std::numeric_limits<size_t>::max()/sizeof(double)/selection.stride)
    return static_cast<int>(cudaErrorInvalidValue);
  for (size_t index=0; index<component_count; ++index) {
    if (components[index]>=size || (index && components[index]<=components[index-1]))
      return static_cast<int>(cudaErrorInvalidValue);
    selection.components[index]=components[index];
  }
  const dim3 grid(selection.component_count,batch_count);
  if (full_eri_sp && family && batches) {
    if (family==1) launch_fitted_sp<1>(selection,grid,degree,batches,direction,status,stream);
    else launch_fitted_sp<2>(selection,grid,degree,batches,direction,status,stream);
    return static_cast<int>(cudaGetLastError());
  }
  switch (full_eri_sp && !family ? degree : 5) {
#define GQC_LAUNCH_DEGREE(degree) \
    case degree: factorized_integral_kernel<degree><<<grid,64,0,stream>>>( \
        selection,records,count,atoms,direction,output,status,batches,frozen_atom); break
    GQC_LAUNCH_DEGREE(0);
    GQC_LAUNCH_DEGREE(1);
    GQC_LAUNCH_DEGREE(2);
    GQC_LAUNCH_DEGREE(3);
    GQC_LAUNCH_DEGREE(4);
    GQC_LAUNCH_DEGREE(5);
#undef GQC_LAUNCH_DEGREE
  }
  return static_cast<int>(cudaGetLastError());
}
int launch_factorized_radial_pair(const unsigned* angular, const std::uint32_t* components,
    size_t component_count, const FactorizedIntegralBatch* batches, size_t batch_count,
    const double* direction, long long frozen_atom, int* status, cudaStream_t stream,
    const FactorizedPairGeometry* prepared, const double* window_records) noexcept {
  if (!angular || !components || !component_count || component_count>16 || !batches ||
      !batch_count || batch_count>65535 || !direction || !status ||
      (bool(prepared)!=bool(window_records)))
    return static_cast<int>(cudaErrorInvalidValue);
  FactorizedSelection selection{};
  selection.centers=4; selection.stride=17; selection.component_count=component_count;
  unsigned degree=0, size=1;
  for (unsigned center=0; center<4; ++center) {
    if (angular[center]>1) return static_cast<int>(cudaErrorInvalidValue);
    selection.angular[center]=angular[center];
    degree+=angular[center];
    size*=angular[center] ? 3 : 1;
  }
  for (size_t index=0; index<component_count; ++index) {
    if (components[index]>=size || (index && components[index]<=components[index-1]))
      return static_cast<int>(cudaErrorInvalidValue);
    selection.components[index]=components[index];
  }
  selection.cache_stride=size;
  const dim3 grid(component_count,batch_count);
  switch (degree) {
#define GQC_LAUNCH_PAIRED(degree) \
    case degree: \
      if (prepared) factorized_integral_kernel<degree,true,true><<<grid,128,0,stream>>>( \
          selection,nullptr,0,nullptr,direction,nullptr,status,batches,frozen_atom,prepared,window_records); \
      else factorized_integral_kernel<degree,true><<<grid,128,0,stream>>>( \
          selection,nullptr,0,nullptr,direction,nullptr,status,batches,frozen_atom); \
      break
    GQC_LAUNCH_PAIRED(0);
    GQC_LAUNCH_PAIRED(1);
    GQC_LAUNCH_PAIRED(2);
    GQC_LAUNCH_PAIRED(3);
    GQC_LAUNCH_PAIRED(4);
#undef GQC_LAUNCH_PAIRED
  }
  return static_cast<int>(cudaGetLastError());
}
int launch_factorized_active_components(unsigned degree, const FactorizedActiveComponent* tasks,
    size_t count, size_t max_radials, const FactorizedIntegralBatch* batches, const double* direction,
    int* status, cudaStream_t stream, const FactorizedPairGeometry* prepared,
    const double* window_records) noexcept {
  if (degree>4 || !tasks || !count || count>65535 || !max_radials || max_radials>128 ||
      !batches || !direction || !status ||
      !prepared || !window_records)
    return static_cast<int>(cudaErrorInvalidValue);
  FactorizedSelection selection{};
  selection.centers=4; selection.stride=17;
  constexpr unsigned strides[]={1,3,9,27,81};
  selection.cache_stride=strides[degree];
  // The first warp publishes the result; a second warp is present if a
  // second 64-record fold is needed. Neither fold changes its add order.
  const unsigned threads=static_cast<unsigned>((max_radials+31)/32)*32;
  const auto shared=threads*25*sizeof(double);
  switch (degree) {
#define GQC_LAUNCH_INDEXED(degree) \
    case degree: factorized_integral_kernel<degree,true,true,true><<<count,threads,shared,stream>>>( \
        selection,nullptr,0,nullptr,direction,nullptr,status,batches,-1,prepared,window_records,tasks); \
      break
    GQC_LAUNCH_INDEXED(0);
    GQC_LAUNCH_INDEXED(1);
    GQC_LAUNCH_INDEXED(2);
    GQC_LAUNCH_INDEXED(3);
    GQC_LAUNCH_INDEXED(4);
#undef GQC_LAUNCH_INDEXED
  }
  return static_cast<int>(cudaGetLastError());
}
}  // namespace generativeqc::dft::hessian_aot
"""
    )
