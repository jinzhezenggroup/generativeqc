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
from .expr import Graph
from .second_derivatives import _eri_differentiate
from .shell_class import _pair_expansion, build_coulomb_derivative_algebra
from .shell_spec import cartesian_components


def _jet_function(graph: Graph, roots: tuple, variables: dict, targets: tuple) -> str:
    """Share scalar CSE without teaching the native template derivative rules."""
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


def _product_source() -> str:
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
        graph, (value, directional), variables, ("result.value", "result.directional")
    )
    body = _jet_function(
        graph,
        (gradient, jvp),
        variables,
        ("result.gradient[index]", "result.jvp[index]"),
    )
    return f"""
template<unsigned Dimension> GQC_FACTOR_FUNCTION Jet<Dimension> multiply(
    const Jet<Dimension>& left, const Jet<Dimension>& right) {{
  Jet<Dimension> result{{}};
  {{ {head} }}
  for (unsigned index=0; index<Dimension; ++index) {{ {body} }}
  return result;
}}
"""


def _pair_source() -> str:
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
            f"case {5 * left_power + right_power}: {{\n"
            + _jet_function(graph, tuple(roots), variables, tuple(targets))
            + "\nreturn true; }"
        )
    return (
        """
GQC_FACTOR_FUNCTION bool pair_coefficients(unsigned left_power, unsigned right_power,
    double inverse_two, double left_shift, double right_shift,
    double left_scale, double right_scale, double left_direction,
    double right_direction, Jet<2>* output) {
  switch (5*left_power+right_power) {
"""
        + "\n".join(cases)
        + "\n} return false; }\n"
    )


def _coulomb_source() -> str:
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
            f"GQC_FACTOR_FUNCTION Jet<3> {name}(const Geometry& geometry, const double* direction) {{\nJet<3> result{{}};\n"
            + _jet_function(graph, roots, variables, targets)
            + "\nreturn result; }\n"
        )
        cases.append(f"case {code}: return {name}(geometry,direction);")
    return (
        "\n".join(functions)
        + """
GQC_FACTOR_FUNCTION Jet<3> coulomb_state(unsigned first, unsigned second, unsigned third,
    const Geometry& geometry, const double* direction) {
  Jet<3> result{};
  switch (169*first+13*second+third) {
"""
        + "\n".join(cases)
        + "\n} return result; }\n"
    )


def _prefactor_source() -> str:
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
        """
GQC_FACTOR_FUNCTION Jet<12> gaussian_prefactor(const Geometry& geometry, const double* direction) {
  Jet<12> result{};
"""
        + _jet_function(graph, roots, variables, targets)
        + "\nreturn result; }\n"
    )


@cache
def emit_factorized_second(*, backend: str = "cpu") -> str:
    """Emit one finite reusable f/g factor library, with no runtime compilation.

    ``primitive`` retains every physical center and returns value, gradient and
    gradient-JVP. Record scaling and ordered radial contraction remain outside
    this primitive. Family codes are ERI=0, three-center=1 and metric=2; radial
    codes are full=0, long=1 and short=2 (fixed WB97M-V omega 0.3).
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("factorized second-order backend must be cpu or cuda")
    qualifier = "static" if backend == "cpu" else "__device__ __noinline__"
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
        "namespace generativeqc::scf::factorized_second {\n"
        "template<unsigned Dimension> struct Jet { double value{}, directional{}; double gradient[Dimension]{}, jvp[Dimension]{}; };\n"
        "struct Geometry { double inverse_two_p, inverse_two_q, rho; double difference[3], shifts[4][3], product_scales[4], decay[4][3]; double prefactor, boys[15]; };\n"
        + _product_source()
        + _pair_source()
        + _coulomb_source()
        + _prefactor_source()
        + "\nGQC_FACTOR_FUNCTION bool primitive(unsigned family, unsigned radial, const unsigned* angular, unsigned component, const double* record, const double* direction, double* output) {\n"
        + "  constexpr unsigned powers[5][15][3]={"
        + ",".join(components)
        + "};\n"
        + r"""
  if (family>2 || radial>2 || (family && radial) || !angular || !record || !direction || !output) return false;
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
    if (angular[center]>(family && (family==2 || center==2) ? 4u : 3u) || !std::isfinite(record[center]) || record[center]<=0) return false;
  double exponents[4]{}, coordinates[12]{}, motion[12]{};
  for (unsigned center=0; center<4; ++center) {
    const bool unit=(family==1 && center==3) || (family==2 && (center==1 || center==3));
    if (!unit) exponents[center]=record[incidence[center]];
    for (unsigned axis=0; axis<3; ++axis) {
      coordinates[3*center+axis]=record[centers+3*incidence[center]+axis];
      if (!unit) motion[3*center+axis]=direction[3*incidence[center]+axis];
      if (!std::isfinite(coordinates[3*center+axis]) || !std::isfinite(motion[3*center+axis])) return false;
    }
  }
  unsigned quantums[4][3]{}, sizes[4]{};
  for (unsigned center=0; center<4; ++center) sizes[center]=(orders[center]+1)*(orders[center]+2)/2;
  unsigned remaining=component;
  for (int center=3; center>=0; --center) {
    const unsigned selected=remaining%sizes[center]; remaining/=sizes[center];
    for (unsigned axis=0; axis<3; ++axis) quantums[center][axis]=powers[orders[center]][selected][axis];
  }
  if (remaining) return false;
  unsigned maximum_order=2;
  for (unsigned center=0; center<4; ++center) maximum_order+=orders[center];
  Geometry geometry{};
  const auto range=radial==0 ? generativeqc::integrals::CoulombRange::Full : radial==1 ? generativeqc::integrals::CoulombRange::Long : generativeqc::integrals::CoulombRange::Short;
  const double omega=radial==0 ? 0 : 0.3;
  bool valid=false;
  if (family==0) valid=generativeqc::integrals::make_eri_geometry<Geometry,14>(exponents,coordinates,maximum_order,range,omega,geometry);
  if (family==1) valid=generativeqc::integrals::make_eri_geometry<Geometry,14,8>(exponents,coordinates,maximum_order,range,omega,geometry);
  if (family==2) valid=generativeqc::integrals::make_eri_geometry<Geometry,14,10>(exponents,coordinates,maximum_order,range,omega,geometry);
  if (!valid) return false;
  Jet<4> coefficients[3][13]{};
  unsigned axis_orders[3]{};
  for (unsigned axis=0; axis<3; ++axis) {
    Jet<2> first[7]{}, second[7]{};
    if (!pair_coefficients(quantums[0][axis],quantums[1][axis],geometry.inverse_two_p,geometry.shifts[0][axis],geometry.shifts[1][axis],geometry.product_scales[0],geometry.product_scales[1],motion[axis],motion[3+axis],first) ||
        !pair_coefficients(quantums[2][axis],quantums[3][axis],geometry.inverse_two_q,geometry.shifts[2][axis],geometry.shifts[3][axis],geometry.product_scales[2],geometry.product_scales[3],motion[6+axis],motion[9+axis],second)) return false;
    const unsigned first_order=quantums[0][axis]+quantums[1][axis], second_order=quantums[2][axis]+quantums[3][axis];
    axis_orders[axis]=first_order+second_order;
    for (unsigned left=0; left<=first_order; ++left) {
      for (unsigned right=0; right<=second_order; ++right) {
        Jet<4> first_jet{}, second_jet{};
        first_jet.value=first[left].value; first_jet.directional=first[left].directional;
        second_jet.value=second[right].value; second_jet.directional=second[right].directional;
        for (unsigned center=0; center<2; ++center) {
          first_jet.gradient[center]=first[left].gradient[center]; first_jet.jvp[center]=first[left].jvp[center];
          second_jet.gradient[center+2]=second[right].gradient[center]; second_jet.jvp[center+2]=second[right].jvp[center];
        }
        const auto term=multiply(first_jet,second_jet);
        auto& target=coefficients[axis][left+right];
        const double sign=right%2 ? -1 : 1;
        target.value+=sign*term.value; target.directional+=sign*term.directional;
        for (unsigned center=0; center<4; ++center) { target.gradient[center]+=sign*term.gradient[center]; target.jvp[center]+=sign*term.jvp[center]; }
      }
    }
  }
  double difference_direction[3]{};
  for (unsigned center=0; center<4; ++center)
    for (unsigned axis=0; axis<3; ++axis) difference_direction[axis]+=(center<2 ? 1 : -1)*geometry.product_scales[center]*motion[3*center+axis];
  Jet<12> result{};
  for (unsigned first=0; first<=axis_orders[0]; ++first)
    for (unsigned second=0; second<=axis_orders[1]; ++second)
      for (unsigned third=0; third<=axis_orders[2]; ++third) {
        const unsigned selected[3]={first,second,third};
        Jet<12> axes[3]{};
        for (unsigned axis=0; axis<3; ++axis) {
          const auto& coefficient=coefficients[axis][selected[axis]];
          axes[axis].value=coefficient.value; axes[axis].directional=coefficient.directional;
          for (unsigned center=0; center<4; ++center) { axes[axis].gradient[3*center+axis]=coefficient.gradient[center]; axes[axis].jvp[3*center+axis]=coefficient.jvp[center]; }
        }
        const auto state=coulomb_state(first,second,third,geometry,difference_direction);
        Jet<12> coulomb{}; coulomb.value=state.value; coulomb.directional=state.directional;
        for (unsigned center=0; center<4; ++center)
          for (unsigned axis=0; axis<3; ++axis) {
            const double scale=(center<2 ? 1 : -1)*geometry.product_scales[center];
            coulomb.gradient[3*center+axis]=scale*state.gradient[axis]; coulomb.jvp[3*center+axis]=scale*state.jvp[axis];
          }
        const auto term=multiply(multiply(multiply(axes[0],axes[1]),axes[2]),coulomb);
        result.value+=term.value; result.directional+=term.directional;
        for (unsigned index=0; index<12; ++index) { result.gradient[index]+=term.gradient[index]; result.jvp[index]+=term.jvp[index]; }
      }
  result=multiply(gaussian_prefactor(geometry,motion),result);
  output[0]=result.value;
  for (unsigned index=0; index<6*centers; ++index) output[index+1]=0;
  for (unsigned center=0; center<4; ++center)
    for (unsigned axis=0; axis<3; ++axis) {
      output[1+3*incidence[center]+axis]+=result.gradient[3*center+axis];
      output[1+3*centers+3*incidence[center]+axis]+=result.jvp[3*center+axis];
    }
  for (unsigned index=0; index<1+6*centers; ++index) if (!std::isfinite(output[index])) return false;
  return true;
}
}  // namespace generativeqc::scf::factorized_second
#undef GQC_FACTOR_FUNCTION
"""
    )


def emit_factorized_resident() -> str:
    """Bind the finite factor library to the same ordered radial contraction.

    Metadata is a bounded by-value kernel argument, not a wildcard capability.
    This launch owns no allocation, synchronization, runtime compiler or oracle.
    """
    return (
        emit_factorized_second(backend="cuda")
        + r"""
#include "dft/factorized_hessian_integral.hpp"
#include <cuda_runtime.h>
#include <limits>
namespace generativeqc::dft::hessian_aot {
namespace {
struct FactorizedSelection {
  unsigned family, radial, angular[4], components[16], component_count, centers, stride;
};
__global__ void factorized_integral_kernel(FactorizedSelection selection, const double* records,
    size_t count, const long long* atoms, const double* molecular_direction,
    double* output, int* status) {
  const unsigned packed=blockIdx.x;
  if (packed>=selection.component_count) return;
  __shared__ double terms[25][64];
  double direction[12]{}, result[25]{};
  const unsigned outputs=1+6*selection.centers;
  for (unsigned center=0; center<selection.centers; ++center)
    for (unsigned axis=0; axis<3; ++axis) direction[3*center+axis]=molecular_direction[3*atoms[center]+axis];
  for (size_t begin=0; begin<count;) {
    const size_t active=count-begin<64 ? count-begin : 64;
    bool valid=true;
    if (threadIdx.x<active) {
      const double* record=records+(begin+threadIdx.x)*selection.stride;
      double values[25]{};
      valid=generativeqc::scf::factorized_second::primitive(selection.family,selection.radial,
          selection.angular,selection.components[packed],record,direction,values);
      for (unsigned index=0; index<outputs; ++index) terms[index][threadIdx.x]=record[selection.stride-1]*values[index];
    }
    if (__syncthreads_or(!valid)) {
      if (threadIdx.x==0) atomicExch(status,1);
      return;
    }
    // Keep the original radial addition order even though primitives evaluate
    // concurrently. There is no floating atomic or reassociating reduction.
    if (threadIdx.x==0)
      for (unsigned index=0; index<outputs; ++index)
        for (size_t radial=0; radial<active; ++radial) result[index]+=terms[index][radial];
    __syncthreads();
    begin+=active;
  }
  if (threadIdx.x!=0) return;
  for (unsigned index=0; index<outputs; ++index) {
    if (!std::isfinite(result[index])) atomicExch(status,2);
    output[packed*outputs+index]=result[index];
  }
}
}  // namespace
int launch_factorized_integral(unsigned family, unsigned radial, const unsigned* angular,
    const std::uint32_t* components, size_t component_count, const double* records, size_t count,
    const long long* atoms, const double* direction, double* output, int* status,
    cudaStream_t stream) noexcept {
  if (family>2 || radial>2 || (family && radial) || !angular || !components ||
      !component_count || component_count>16 || !records || !count || !atoms ||
      !direction || !output || !status) return static_cast<int>(cudaErrorInvalidValue);
  FactorizedSelection selection{};
  selection.family=family; selection.radial=radial;
  selection.centers=family==0 ? 4 : family==1 ? 3 : 2;
  selection.stride=4*selection.centers+1;
  selection.component_count=static_cast<unsigned>(component_count);
  size_t size=1;
  for (unsigned center=0; center<selection.centers; ++center) {
    if (angular[center]>(family && (family==2 || center==2) ? 4u : 3u))
      return static_cast<int>(cudaErrorInvalidValue);
    selection.angular[center]=angular[center];
    size*=(angular[center]+1)*(angular[center]+2)/2;
  }
  if (count>std::numeric_limits<size_t>::max()/sizeof(double)/selection.stride)
    return static_cast<int>(cudaErrorInvalidValue);
  for (size_t index=0; index<component_count; ++index) {
    if (components[index]>=size || (index && components[index]<=components[index-1]))
      return static_cast<int>(cudaErrorInvalidValue);
    selection.components[index]=components[index];
  }
  factorized_integral_kernel<<<selection.component_count,64,0,stream>>>(selection,records,count,atoms,direction,output,status);
  return static_cast<int>(cudaGetLastError());
}
}  // namespace generativeqc::dft::hessian_aot
"""
    )
