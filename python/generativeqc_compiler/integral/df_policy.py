"""Typed DF policies connecting shared scientific lowering to basis traversal.

Rank is the number of real Gaussian factors: two for M and three for A.
The native traversal knows no integral family or Gaussian derivative identity.
Raw coordinate projections and externally weighted consumers share these exact
center channels, including the translation-derived auxiliary contribution.
"""

import typing

from .df_derivatives_cuda import emit_df_derivatives_cuda


def _auxiliary_g_policy_adapter() -> str:
    """Preserve the f evaluator while binding g responses to the same policy ABI.

    The conversion only copies value/center channels. The explicit g evaluator
    owns its F11 work arrays; lower angular classes keep their existing stack
    and arithmetic, even when another shell in the same source contains g.
    """
    return r"""namespace generativeqc::scf::generated_df_g_adapter {
using Vec3=generated_df_derivatives::Vec3;
using Angular=generated_df_derivatives::Angular;
using Response=generated_df_derivatives::Response;
__device__ __forceinline__ Response convert(generated_df_auxiliary_g_derivatives::Response r) {
  return {r.value,{r.first.x,r.first.y,r.first.z},{r.second.x,r.second.y,r.second.z},
                 {r.third.x,r.third.y,r.third.z}};
}
__device__ __forceinline__ Response metric(double alpha,Vec3 A,Angular a,double gamma,Vec3 C,Angular c) {
  if(generated_df_derivatives::order(a)<=3 && generated_df_derivatives::order(c)<=3)
    return generated_df_derivatives::metric(alpha,A,a,gamma,C,c);
  return convert(generated_df_auxiliary_g_derivatives::metric(alpha,{A.x,A.y,A.z},{a.x,a.y,a.z},
                  gamma,{C.x,C.y,C.z},{c.x,c.y,c.z}));
}
__device__ __forceinline__ Response three_center(double alpha,Vec3 A,Angular a,
    double beta,Vec3 B,Angular b,double gamma,Vec3 C,Angular c) {
  if(generated_df_derivatives::order(c)<=3)
    return generated_df_derivatives::three_center(alpha,A,a,beta,B,b,gamma,C,c);
  return convert(generated_df_auxiliary_g_derivatives::three_center(alpha,{A.x,A.y,A.z},{a.x,a.y,a.z},
       beta,{B.x,B.y,B.z},{b.x,b.y,b.z},gamma,{C.x,C.y,C.z},{c.x,c.y,c.z}));
}
} // namespace generativeqc::scf::generated_df_g_adapter
""".replace("__device__", "static __device__")


def emit_df_value_source_schedule_cuda() -> str:
    """Separate raw public tiles from outputs that reduce over auxiliaries.

    A raw tile has one source auxiliary per output. The transformed consumer
    has many and benefits from its existing cooperative primitive/source warp.
    These mappings change work placement, not the scalar integral equations.
    Explicit mappings remain stable controls for endpoint qualification.
    """
    return """struct ValueSourceSchedule {
  static constexpr unsigned primitive_lanes = 4;
  static constexpr unsigned auxiliary_mapping = 0;
  static constexpr unsigned component_mapping = 1;
  static constexpr unsigned primitive_mapping = 2;
  static constexpr unsigned automatic_mapping = 3;
  static constexpr unsigned resolve(unsigned requested, bool transformed) {
    return requested == automatic_mapping
        ? (transformed ? primitive_mapping : auxiliary_mapping) : requested;
  }
};
"""


def emit_df_derivative_schedule_cuda() -> typing.Any:
    """Emit weighted-response scheduling independently of scalar mathematics.

    Four lanes split long primitive products while keeping eight independent
    outputs per warp. A separate artifact lets measured scheduling changes
    leave raw value/derivative consumers and their mathematical policy intact.
    """
    return """// Generated weighted DF schedule; generic runtime performs the reduction.
#ifndef GENERATIVEQC_GENERATED_DF_DERIVATIVE_SCHEDULE_CUH
#define GENERATIVEQC_GENERATED_DF_DERIVATIVE_SCHEDULE_CUH
namespace generativeqc::scf::generated_df_policy {
struct WeightedSchedule {
  static constexpr unsigned block_threads = 32;
  static constexpr unsigned lanes_per_element = 4;
};
} // namespace generativeqc::scf::generated_df_policy
#endif
"""


def emit_df_policy_cuda(*, derivatives: typing.Any = False) -> typing.Any:
    """Emit one consumer's policy without registering unused device tables.

    CUDA emits host registration symbols even for device definitions. Keep
    values separate so a derivative-only TU carries no unused Rys value data;
    scalar headers themselves use internal linkage for safe multi-TU reuse.
    """
    guard = (
        "GENERATIVEQC_GENERATED_DF_"
        + ("DERIVATIVE" if derivatives else "VALUE")
        + "_POLICY_CUH"
    )
    header = (
        "generated_df_derivatives.cuh"
        if derivatives
        else "generated_df_value_candidates.cuh"
    )
    value = (
        emit_df_value_source_schedule_cuda()
        + r"""
template<unsigned Math=0> struct ValueMath {
  using Vec3 = generated_df::Vec3;
  using Angular = generated_df::Angular;
  using Accumulator = double;
  template <unsigned Rank>
  __device__ static void accumulate(double& out, const double* e, const Vec3* r,
                                    const Angular* a, double weight) {
    static_assert(Rank == 2 || Rank == 3);
    // Scalar evaluators take references. Isolate their small arguments so an
    // escaped reference does not force the complete traversal state to local
    // memory; the runtime's arrays can then be scalarized into registers.
    const Vec3 first=r[0], second=r[1];
    const Angular first_angular=a[0], second_angular=a[1];
    if constexpr (Rank == 2)
      out += weight * generated_df::metric(e[0],first,first_angular,e[1],second,second_angular);
    else {
      const Vec3 third=r[2]; const Angular third_angular=a[2];
      out += weight * generated_df_value_candidates::three_center<Math>(e[0],first,first_angular,e[1],second,second_angular,e[2],third,third_angular);
    }
  }
};
using Value=ValueMath<0>;

/** Retain raw-value residuals until the first orbital contraction. Ordinary
 * HF values and all derivative policies keep their existing arithmetic.
 */
template<unsigned Math=0> struct CompensatedValue {
  using Vec3 = generated_df::Vec3;
  using Angular = generated_df::Angular;
  using Weight = generated_df::fp64_expansion::Wide;
  using Accumulator = Weight;
  template<unsigned Rank>
  __device__ static void accumulate(Accumulator& out,const double* e,const Vec3* r,
                                   const Angular* a,Weight weight) {
    static_assert(Rank==3);
    const unsigned total=generated_df::order(a[0])+generated_df::order(a[1])+generated_df::order(a[2]);
    if(total<=2)
      out+=weight*generated_df::compensated::value(e[0],r[0],a[0],e[1],r[1],a[1],e[2],r[2],a[2]);
    else
      out+=weight*generated_df_value_candidates::three_center<Math>(e[0],r[0],a[0],e[1],r[1],a[1],e[2],r[2],a[2]);
  }
};
"""
    )
    derivative = r"""struct Derivative {
  using Vec3 = generated_df_derivatives::Vec3;
  using Angular = generated_df_derivatives::Angular;
  struct Accumulator { double gradient[3][3]{}; };
  template <unsigned Rank>
  __device__ static void accumulate(Accumulator& out, const double* e, const Vec3* r,
                                    const Angular* a, double weight) {
    static_assert(Rank == 2 || Rank == 3);
    generated_df_derivatives::Response result;
    // Keep callee reference arguments independent of the traversal aggregate.
    const Vec3 first=r[0], second=r[1];
    const Angular first_angular=a[0], second_angular=a[1];
    if constexpr (Rank == 2)
      result = generated_df_derivatives::metric(e[0],first,first_angular,e[1],second,second_angular);
    else {
      const Vec3 third=r[2]; const Angular third_angular=a[2];
      result = generated_df_derivatives::three_center(e[0],first,first_angular,e[1],second,second_angular,e[2],third,third_angular);
    }
    const Vec3 channels[3]{result.first, Rank == 2 ? result.third : result.second, result.third};
#pragma unroll
    for (unsigned slot=0;slot<Rank;++slot) {
      out.gradient[slot][0] += weight*channels[slot].x;
      out.gradient[slot][1] += weight*channels[slot].y;
      out.gradient[slot][2] += weight*channels[slot].z;
    }
  }
};
"""
    g_support = (
        emit_df_derivatives_cuda(auxiliary_g=True) + _auxiliary_g_policy_adapter()
        if derivatives
        else ""
    )
    g_policy = (
        derivative.replace("struct Derivative", "struct AuxiliaryGDerivative").replace(
            "generated_df_derivatives::", "generated_df_g_adapter::"
        )
        if derivatives
        else ""
    )
    return (
        "// Generated DF policy; runtime owns normalized basis traversal.\n"
        f"#ifndef {guard}\n#define {guard}\n"
        f'#include "{header}"\n'
        + g_support
        + "namespace generativeqc::scf::generated_df_policy {\n"
        + (derivative if derivatives else value)
        + g_policy
        + "} // namespace generativeqc::scf::generated_df_policy\n#endif\n"
    )
