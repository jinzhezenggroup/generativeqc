"""Bounded DF Gaussian-moment polynomial lowering with analytic basis response."""

import typing
from itertools import product

from .cuda import CudaEmitter
from .df_derivatives import axis_polynomial, build_df_derivative_ir
from .ir_serialization import integral_to_payload


def df_derivative_inventory(
    *, auxiliary_g: bool = False, scalar_type: str = "double"
) -> typing.Any:
    """Separate the external-response contract from native tiling decisions."""
    if scalar_type not in ("double", "float"):
        raise ValueError("scalar type must be double or float")
    inventory = {
        "schema": "generativeqc.df_derivatives",
        "version": 1,
        "precision": "fp64",
        "maximum_boys_order": 11 if auxiliary_g else 10,
        "lowering": "gaussian_moment_polynomials",
        "programs": [
            integral_to_payload(
                build_df_derivative_ir(family, angular, weighted=weighted)
            )
            for family, extents in (
                ("coulomb_metric", (5 if auxiliary_g else 4,) * 2),
                ("three_center_eri", (4, 4, 5 if auxiliary_g else 4)),
            )
            for angular in product(*(range(size) for size in extents))
            for weighted in (False, True)
        ],
    }
    if scalar_type == "float":
        inventory.update(
            precision="mixed-fp32-polynomials",
            arithmetic={
                "geometry": "fp64",
                "boys": "fp64",
                "moment_polynomials": "fp32",
                "moment_convolution": "fp32",
                "primitive_derivative": "fp32",
                "returned_channels": "fp64",
                "contracted_accumulation": "consumer-owned",
            },
        )
    return inventory


def emit_df_geometry_cuda(
    name: typing.Any = "prepare_geometry",
    *,
    moments: typing.Any = "boys_values(total+1,rho*distance,g.f,work);",
    compensated: bool = False,
) -> typing.Any:
    """Share primitive geometry between polynomial and Rys derivative lowering.

    ``moments`` injects a compiler-owned quadrature call into the same Gaussian
    product/decay preparation. Keep its position before the prefactor identical
    to the existing polynomial path, including floating-point operation order.
    The emitted function lives where Vec3, Geometry, BoysWork and component
    already name the scalar derivative contract; no native equations are copied.
    """
    source = r"""__device__ __forceinline__ void prepare_geometry(double alpha,Vec3 A,double beta,Vec3 B,
    double gamma,Vec3 C,unsigned total,Geometry& g,BoysWork* work=nullptr) {
  const double p=alpha+beta,q=gamma,rho=p*q/(p+q);
  g.sx=q/(p+q);g.sy=p/(p+q);g.ip=0.5/p;g.iq=0.5/q;
  double distance=0,ab2=0;
  for(unsigned axis=0;axis<3;++axis) {
    const double ab=component(A,axis)-component(B,axis);
    g.pa[axis]=-beta/p*ab;g.pb[axis]=alpha/p*ab;
    g.dx[axis]=component(A,axis)-component(C,axis)+g.pa[axis];
    distance+=g.dx[axis]*g.dx[axis];ab2+=ab*ab;
  }
  boys_values(total+1,rho*distance,g.f,work);
  g.prefactor=34.986836655249725694/(p*q*sqrt(p+q))*exp(-alpha*beta/p*ab2);
}"""
    source = source.replace("prepare_geometry(", name + "(", 1).replace(
        "boys_values(total+1,rho*distance,g.f,work);", moments, 1
    )
    if compensated:
        # The geometry equations and order stay shared. component() in this
        # namespace returns Wide, so coordinate subtraction also retains its
        # residual. The split prefactor is 2*pi**(5/2), rounded only at output.
        source = source.replace("double", "Wide").replace(
            "34.986836655249725694", "Wide(34.986836655249725,7.2256294425378e-16)"
        )
    return source


def emit_df_boys_cuda(*, compensated: bool = False) -> str:
    """Emit the shared FP64 positive-series/downward Boys evaluation.

    Callers own order+1 output slots. Value and derivative consumers use the
    same arithmetic and convergence rule; no fitted high-order Rys rule is
    introduced by the g-auxiliary value consumer.
    """
    source = r"""/** Diagnostic metadata describes the actual positive-series branch, not FLOPs.
 * A null sink is a compile-time constant in the normal inlined callers.
 * Zero arguments still execute the series; small_argument is a subdomain of
 * that branch, never a claim that a separate asymptotic formula was used.
 */
struct BoysWork {
  unsigned series_iterations{},series{},large_argument{},small_argument{};
};
/** Positive-term series/downward recurrence avoids small-T cancellation. */
__device__ __forceinline__ void boys_values(unsigned order,double argument,double* f,
                                           BoysWork* work=nullptr) {
  if(work) *work={};
  const double decay=exp(-argument);
  if (argument<30.0) {
    if(work) {work->series=1;work->small_argument=argument<1e-8;}
    double term=1.0/(2*order+1),sum=term;
    for(unsigned k=1;k<180;++k) {
      term*=2*argument/(2*order+2*k+1); sum+=term;
      if(work) ++work->series_iterations;
      if(term<1e-17*sum) break;
    }
    f[order]=decay*sum;
    for(unsigned n=order;n>0;--n) f[n-1]=(2*argument*f[n]+decay)/(2*n-1);
  } else {
    if(work) work->large_argument=1;
    f[0]=0.88622692545275801365*erf(sqrt(argument))/sqrt(argument);
    for(unsigned n=1;n<=order;++n) f[n]=((2*n-1)*f[n-1]-decay)/(2*argument);
  }
}
"""
    if compensated:
        source = (
            source.replace("double", "Wide")
            .replace("1.0/(2*order+1)", "Wide(1)/(2*order+1)")
            .replace("1e-17*sum", "1e-30*sum")
            .replace(
                "0.88622692545275801365",
                "Wide(0.886226925452758,-3.8332932499128993e-17)",
            )
        )
    return source


def emit_df_polynomial_dot_cuda(*, scalar_type: str = "double") -> str:
    """Integrate the common axis polynomials against exact Boys moments."""
    if scalar_type not in ("double", "float"):
        raise ValueError("scalar type must be double or float")
    source = r"""/** Integrate the product of three polynomials using exact Boys moments. */
__device__ double dot(unsigned da,const double* a,unsigned db,const double* b,
    unsigned dc,const double* c,const double* f) {
  double value=0.0;
  for(unsigned i=0;i<=da;++i)
    for(unsigned j=0;j<=db;++j)
      for(unsigned k=0;k<=dc;++k) value+=a[i]*b[j]*c[k]*f[i+j+k];
  return value;
}
"""
    return (
        source
        if scalar_type == "double"
        else source.replace("double", "float").replace("0.0", "0.0f")
    )


def emit_df_derivatives_cuda(
    *, auxiliary_g: bool = False, scalar_type: str = "double"
) -> str:
    """Share base axis moments and Boys values across all independent centers.

    The derivative of an unnormalized basis factor is
    2*alpha*g_(a+1) - a*g_(a-1). This is applied to the same moment DAG as the
    value generator, at generation time. A branch owns at most eleven scalar
    coefficients; the runtime holds bounded coefficient arrays, never AD state.
    The explicit auxiliary-g variant shares this algebra but owns twelve slots,
    F_11 and role-checked internal raising. It is emitted into a separate
    namespace; requesting it does not promote any public force capability.

    The optional float candidate narrows polynomial inputs, DAG temporaries,
    moment products and derivative sums, not just the output storage. Gaussian
    geometry and Boys preparation retain FP64 before an explicit narrowing
    boundary. Returned channels widen once into the existing response ABI.
    This compiler candidate alone does not admit a production force path; its
    consumer must own qualification and pre-scatter strict recovery.
    """
    if scalar_type not in ("double", "float"):
        raise ValueError("scalar type must be double or float")
    prefix = r"""// Generated DF metric/three-center first derivatives.
#ifndef GENERATIVEQC_GENERATED_DF_DERIVATIVES_CUH
#define GENERATIVEQC_GENERATED_DF_DERIVATIVES_CUH
#include <cuda_runtime.h>
#include <cmath>
namespace generativeqc::scf::generated_df_derivatives {
struct Vec3 { double x,y,z; };
struct Angular { unsigned x,y,z; };
struct Response { double value; Vec3 first,second,third; };
__device__ __forceinline__ unsigned order(Angular a) { return a.x+a.y+a.z; }
__device__ __forceinline__ double component(Vec3 a,unsigned i) { return i==0?a.x:i==1?a.y:a.z; }
__device__ __forceinline__ unsigned power(Angular a,unsigned i) { return i==0?a.x:i==1?a.y:a.z; }
__DF_BOYS__/** Exact coefficients in u=t^2 of the shared Gaussian-moment value DAG. */
__device__ __noinline__ void axis_polynomial(unsigned a,unsigned b,unsigned c,
    __DF_ARITHMETIC__ pa,__DF_ARITHMETIC__ pb,__DF_ARITHMETIC__ dx,__DF_ARITHMETIC__ sx,__DF_ARITHMETIC__ sy,__DF_ARITHMETIC__ ip,__DF_ARITHMETIC__ iq,__DF_ARITHMETIC__* out) {
  switch(a*20U+b*4U+c) {
"""
    lines = [prefix.replace("__DF_BOYS__", emit_df_boys_cuda())]
    for a, b, c in product(
        range(6 if auxiliary_g else 5), range(5), range(5 if auxiliary_g else 4)
    ):
        if a == b == 4 or (a == 5 and b != 0):
            continue
        graph, roots = axis_polynomial(a, b, c, auxiliary_g_derivative=auxiliary_g)
        emitter = CudaEmitter(graph, {}, scalar_type=scalar_type)
        emitter.emit(roots)
        key = a * 25 + b * 5 + c if auxiliary_g else a * 20 + b * 4 + c
        lines += [f"    case {key}U: {{", *emitter.lines]
        lines += [
            f"      out[{i}]={emitter.reference(root)};" for i, root in enumerate(roots)
        ]
        lines += ["      return;", "    }"]
    lines += [
        "  }",
        "  out[0]=NAN;",
        "}",
        r"""
__DF_POLYNOMIAL_DOT__/** One primitive product, independent of the AO components using it. */
struct Geometry {
  double pa[3],pb[3],dx[3],sx,sy,ip,iq,prefactor,f[11];
};
__DF_GEOMETRY_PREPARATION__
__device__ __noinline__ Response evaluate(bool metric,double alpha,Vec3 A,Angular a,
    double beta,Vec3 B,Angular b,double gamma,Vec3 C,Angular c) {
  const double invalid=NAN;
  if (order(a)>3 || order(b)>3 || order(c)>3 || !(alpha>0) || !(gamma>0) ||
      (!metric && !(beta>0))) return {invalid,{invalid,invalid,invalid},{invalid,invalid,invalid},{invalid,invalid,invalid}};
  Geometry geometry;
  prepare_geometry(alpha,A,beta,B,gamma,C,order(a)+order(b)+order(c),geometry);
  const auto& pa=geometry.pa;const auto& pb=geometry.pb;const auto& dx=geometry.dx;
  __DF_MOMENT_INPUTS__
  const __DF_ARITHMETIC__ sx=geometry.sx,sy=geometry.sy,ip=geometry.ip,iq=geometry.iq;
  const __DF_ARITHMETIC__ prefactor=geometry.prefactor;
  __DF_ARITHMETIC__ base[3][11];
  unsigned degree[3];
  for(unsigned axis=0;axis<3;++axis) {
    degree[axis]=power(a,axis)+power(b,axis)+power(c,axis);
    axis_polynomial(power(a,axis),power(b,axis),power(c,axis),pa[axis],pb[axis],dx[axis],sx,sy,ip,iq,base[axis]);
  }
  Response result{};
  result.value=prefactor*dot(degree[0],base[0],degree[1],base[1],degree[2],base[2],f);
  __DF_ARITHMETIC__ first[3]{},second[3]{};
  for(unsigned axis=0;axis<3;++axis) {
    const unsigned other=(axis+1)%3,last=(axis+2)%3;
    const unsigned na=power(a,axis),nb=power(b,axis),nc=power(c,axis),d=degree[axis];
    __DF_ARITHMETIC__ raised[11],lowered[11];
    for(unsigned center=0;center<(metric?1U:2U);++center) {
      const unsigned n=center==0?na:nb;
      axis_polynomial(na+(center==0),nb+(center==1),nc,pa[axis],pb[axis],dx[axis],sx,sy,ip,iq,raised);
      if(n) axis_polynomial(na-(center==0),nb-(center==1),nc,pa[axis],pb[axis],dx[axis],sx,sy,ip,iq,lowered);
      // Fuse the analytically differentiated Gaussian factor before integrating
      // the remaining axes; exponents and supplied response weights stay fixed.
      for(unsigned i=0;i<=d+1;++i)
        raised[i]=2*(center==0?__DF_ALPHA__:__DF_BETA__)*raised[i]-(n && i<d ? n*lowered[i] : __DF_ZERO__);
      const __DF_ARITHMETIC__ response=prefactor*dot(d+1,raised,degree[other],base[other],degree[last],base[last],f);
      if(center==0) first[axis]=response; else second[axis]=response;
    }
  }
  result.first={first[0],first[1],first[2]};
  result.second={second[0],second[1],second[2]};
  result.third={-first[0]-second[0],-first[1]-second[1],-first[2]-second[2]};
  return result;
}
/** A metric has two real auxiliary centers; the internal absent B has no basis normalization. */
__device__ __forceinline__ Response metric(double alpha,Vec3 A,Angular a,double gamma,Vec3 C,Angular c) {
  return evaluate(true,alpha,A,a,0.0,A,{0,0,0},gamma,C,c);
}
__device__ __forceinline__ Response three_center(double alpha,Vec3 A,Angular a,
    double beta,Vec3 B,Angular b,double gamma,Vec3 C,Angular c) {
  return evaluate(false,alpha,A,a,beta,B,b,gamma,C,c);
}
} // namespace generativeqc::scf::generated_df_derivatives
#endif
""",
    ]
    # Raw and weighted consumers compile this same definition in separate TUs.
    # Device functions need internal linkage, including their NVCC host stubs.
    source = (
        "\n".join(lines)
        .replace(
            "__DF_POLYNOMIAL_DOT__",
            emit_df_polynomial_dot_cuda(scalar_type=scalar_type),
        )
        .replace("__DF_ARITHMETIC__", scalar_type)
        .replace("__DF_ZERO__", "0.0f" if scalar_type == "float" else "0.0")
        .replace(
            "__DF_MOMENT_INPUTS__",
            "float f[11];\n  for(unsigned index=0;index<=order(a)+order(b)+order(c)+1;++index)\n"
            "    f[index]=static_cast<float>(geometry.f[index]);\n"
            "  const float primitive_alpha=alpha,primitive_beta=beta;"
            if scalar_type == "float"
            else "const auto& f=geometry.f;",
        )
        .replace(
            "__DF_ALPHA__", "primitive_alpha" if scalar_type == "float" else "alpha"
        )
        .replace("__DF_BETA__", "primitive_beta" if scalar_type == "float" else "beta")
        .replace("__DF_GEOMETRY_PREPARATION__", emit_df_geometry_cuda())
        .replace("__device__", "static __device__")
    )
    if auxiliary_g:
        source = (
            source.replace(
                "GENERATIVEQC_GENERATED_DF_DERIVATIVES",
                "GENERATIVEQC_GENERATED_DF_AUXILIARY_G_DERIVATIVES",
            )
            .replace("generated_df_derivatives", "generated_df_auxiliary_g_derivatives")
            .replace("a*20U+b*4U+c", "a*25U+b*5U+c")
            # Resize declarations only: out[11] is a real coefficient,
            # not a storage bound, in the newly admitted F_11 branches.
            .replace("f[11]", "f[12]")
            .replace("base[3][11]", "base[3][12]")
            .replace("raised[11],lowered[11]", "raised[12],lowered[12]")
            .replace(
                "order(a)>3 || order(b)>3 || order(c)>3",
                "order(a)>(metric?4U:3U) || order(b)>3 || order(c)>4",
            )
        )
    if scalar_type == "float":
        source = source.replace(
            "GENERATIVEQC_GENERATED_DF_", "GENERATIVEQC_GENERATED_DF_FP32_"
        ).replace("generated_df_", "generated_df_fp32_")
    return source


def emit_df_derivatives_cpu(
    *, auxiliary_g: bool = False, scalar_type: str = "double"
) -> str:
    """Lower the same generated DF derivative algebra to ordinary host C++.

    Keep one scientific expression source for CUDA and CPU.  The CUDA emitter
    already contains only scalar math after code generation; this adapter removes
    device-only qualifiers/includes without changing any generated arithmetic.
    """

    source = emit_df_derivatives_cuda(auxiliary_g=auxiliary_g, scalar_type=scalar_type)
    if auxiliary_g:
        source = source.replace(
            "GENERATIVEQC_GENERATED_DF_AUXILIARY_G_DERIVATIVES_CUH",
            "GENERATIVEQC_GENERATED_DF_AUXILIARY_G_DERIVATIVES_CPU_HPP",
        )
    source = source.replace(
        "GENERATIVEQC_GENERATED_DF_DERIVATIVES_CUH",
        "GENERATIVEQC_GENERATED_DF_DERIVATIVES_CPU_HPP",
    )
    source = source.replace(
        "GENERATIVEQC_GENERATED_DF_FP32_DERIVATIVES_CUH",
        "GENERATIVEQC_GENERATED_DF_FP32_DERIVATIVES_CPU_HPP",
    ).replace(
        "GENERATIVEQC_GENERATED_DF_FP32_AUXILIARY_G_DERIVATIVES_CUH",
        "GENERATIVEQC_GENERATED_DF_FP32_AUXILIARY_G_DERIVATIVES_CPU_HPP",
    )
    source = source.replace("#include <cuda_runtime.h>\n", "")
    source = source.replace("static __device__ __forceinline__", "inline")
    source = source.replace("static __device__ __noinline__", "inline")
    source = source.replace("static __device__", "inline")
    source = source.replace("__forceinline__", "inline")
    source = source.replace("__noinline__", "")
    return source
