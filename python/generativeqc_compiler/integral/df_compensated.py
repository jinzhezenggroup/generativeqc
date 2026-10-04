"""Compensated low-angular DF values from the existing component IR.

Diffuse s/p products are amplified by both a nearly dependent orbital frame
and metric whitening. Preserve FP64 residuals through geometry, Boys moments,
the primitive prefactor and contracted weights; publish ordinary FP64 values.
Higher angular classes retain their strict common Rys/polynomial lowering.
"""

from fractions import Fraction
from itertools import product

from ..common.fp64_expansion import emit_fp64_expansion
from .cuda import CudaEmitter
from .df_derivatives_cuda import emit_df_boys_cuda, emit_df_geometry_cuda
from .df_values import build_df_component_kernel, build_df_value_ir
from .shell_spec import cartesian_components


def emit_df_compensated() -> str:
    """Use the shared geometry, Boys evaluator and Hermite/Coulomb component DAG.

    The degree <=2 DAG has only exactly representable dyadic constants. Its
    ordinary scalar emitter can therefore retain literals while the temporary
    type selects the expansion arithmetic. No second integral algebra is added.
    """
    lines = [
        emit_fp64_expansion(),
        "namespace compensated {",
        "using fp64_expansion::Wide;",
        "__device__ inline Wide component(Vec3 p,unsigned i) {return i==0?p.x:(i==1?p.y:p.z);}",
        "struct Geometry {Wide sx,sy,ip,iq,pa[3],pb[3],dx[3],f[3],prefactor;};",
        emit_df_boys_cuda(compensated=True),
        emit_df_geometry_cuda(
            moments="boys_values(total,rho*distance,g.f,work);", compensated=True
        ),
        "__device__ inline Wide value(double alpha,Vec3 A,Angular a,double beta,Vec3 B,Angular b,double gamma,Vec3 C,Angular c) {",
        "  const unsigned total=a.x+a.y+a.z+b.x+b.y+b.z+c.x+c.y+c.z;",
        "  Geometry g;prepare_geometry(alpha,A,beta,B,gamma,C,total,g);",
    ]
    for angular in product(range(3), repeat=3):
        if sum(angular) > 2:
            continue
        integral = build_df_value_ir("three_center_eri", angular)
        for components in product(*(cartesian_components(l) for l in angular)):
            kernel = build_df_component_kernel(integral, components)
            # The scalar emitter writes one FP64 literal per DAG constant.
            # Fail closed if extending the supported IR would round one before
            # expansion arithmetic gets a chance to preserve its residual.
            for identifier in kernel.graph.topological_order((kernel.value,)):
                node = kernel.graph.nodes[identifier]
                if node.operation == "constant" and (
                    node.payload is None
                    or Fraction(float(node.payload)) != node.payload
                ):
                    raise ValueError("compensated DF IR needs exact FP64 constants")
            variables = {
                "inverse_two_p": "g.ip",
                "inverse_two_q": "g.iq",
                "rho": "(0.5/(g.ip+g.iq))",
                **{f"boys_{n}": f"g.f[{n}]" for n in range(sum(angular) + 1)},
                **{
                    f"difference_{name}": f"g.dx[{axis}]"
                    for axis, name in enumerate("xyz")
                },
                **{
                    f"p{slot}_{name}": f"g.{field}[{axis}]"
                    for slot, field in enumerate(("pa", "pb"))
                    for axis, name in enumerate("xyz")
                },
            }
            condition = " && ".join(
                f"{slot}.{axis}=={component.count(axis)}"
                for slot, component in zip("abc", components, strict=True)
                for axis in "xyz"
            )
            emitter = CudaEmitter(kernel.graph, variables)
            emitter.emit((kernel.value,))
            lines += [f"  if({condition}) {{"]
            lines += [
                line.replace("const double", "const Wide") for line in emitter.lines
            ]
            lines += [
                f"    return g.prefactor*{emitter.reference(kernel.value)};",
                "  }",
            ]
    lines += ["  return NAN;", "}", "} // namespace compensated"]
    return "\n".join(lines) + "\n"
