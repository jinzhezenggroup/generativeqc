"""Stationary RHF pullback from energy to ordered integral source fields.

The native primal supplies a converged spin-summed density D and its physical
energy-weighted density W. Differentiating E(D,h,g) - W:S gives the complete
first derivative of the constrained minimum, including basis/Pulay motion.
It does not differentiate a frozen unconstrained density or SCF iterations.
"""

from generativeqc_compiler.tensor import (
    Program,
    TensorSpec,
    add,
    input_tensor,
    multiply,
    transpose_program,
)
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp

from .df_hf_response_contract import RHF_COULOMB_COEFFICIENT, RHF_EXCHANGE_COEFFICIENT


def stationary_integral_program() -> Program:
    """Return a generated VJP of the normalized-orbital RHF Lagrangian."""
    fixed = TensorSpec(role="input", differentiable=False)
    parameter = TensorSpec(role="parameter", differentiable=True)
    dab, dcd, dac, dbd, wab = (
        input_tensor(name, fixed) for name in ("dab", "dcd", "dac", "dbd", "wab")
    )
    overlap, hcore, eri = (
        input_tensor(name, parameter) for name in ("overlap", "hcore", "eri")
    )
    pair = add(
        multiply(dab, dcd),
        multiply(dac, dbd),
        coefficients=(RHF_COULOMB_COEFFICIENT / 2, -RHF_EXCHANGE_COEFFICIENT),
    )
    lagrangian = add(
        multiply(dab, hcore),
        multiply(wab, overlap),
        multiply(pair, eri),
        coefficients=(1, -1, 1),
    )
    return transpose_program(
        Program({"energy": lagrangian}), ("energy",), inputs=("overlap", "hcore", "eri")
    ).program


def emit_stationary_integral_weights() -> str:
    """Emit source weights; contractions and normalization remain integral-owned."""
    program = stationary_integral_program()
    helper = emit_scalar_cpp(
        program,
        function_name="rhf_source_weights",
        input_order=("dab", "dcd", "dac", "dbd", "wab", "bar_energy"),
        output_order=("bar_overlap", "bar_hcore", "bar_eri"),
    )
    return (
        helper
        + r"""
extern "C" int integral_ad_rhf_weights(std::size_t n, const double* d, const double* w, double* out) {
  double inputs[6], weights[3];
  inputs[5] = 1.0;
  for (std::size_t a=0; a<n; ++a) for (std::size_t b=0; b<n; ++b) {
    inputs[0]=d[a*n+b]; inputs[4]=w[a*n+b];
    inputs[1]=inputs[2]=inputs[3]=0;
    if (!rhf_source_weights(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], weights[0], weights[1], weights[2])) return 1;
    out[a*n+b]=weights[0]; out[n*n+a*n+b]=weights[1];
    for (std::size_t c=0; c<n; ++c) for (std::size_t e=0; e<n; ++e) {
      inputs[1]=d[c*n+e]; inputs[2]=d[a*n+c]; inputs[3]=d[b*n+e];
      if (!rhf_source_weights(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], weights[0], weights[1], weights[2])) return 1;
      out[2*n*n+((a*n+b)*n+c)*n+e]=weights[2];
    }
  }
  out[2*n*n+n*n*n*n]=1.0;
  return 0;
}
"""
    )


def emit_spin_integral_weights() -> str:
    """Generate HF integral weights and their density/metric-response JVP.

    Unit-occupation spin densities unify RHF and UHF. Only the upstream solve
    decides whether orbital motion is restricted; this source algebra is shared.
    """
    from fractions import Fraction

    from generativeqc_compiler.tensor import linearize

    state_names = (
        "daab",
        "dbab",
        "dacd",
        "dbcd",
        "daac",
        "dbac",
        "dabd",
        "dbbd",
        "waab",
        "wbab",
    )
    parameter = TensorSpec(role="parameter", differentiable=True)
    daab, dbab, dacd, dbcd, daac, dbac, dabd, dbbd, waab, wbab = (
        input_tensor(name, parameter) for name in state_names
    )
    overlap, hcore, eri = (
        input_tensor(name, parameter) for name in ("overlap", "hcore", "eri")
    )
    density = add(daab, dbab)
    pair = add(
        multiply(density, add(dacd, dbcd)),
        multiply(daac, dabd),
        multiply(dbac, dbbd),
        coefficients=(Fraction(1, 2), Fraction(-1, 2), Fraction(-1, 2)),
    )
    lagrangian = add(
        multiply(density, hcore),
        multiply(add(waab, wbab), overlap),
        multiply(pair, eri),
        coefficients=(1, -1, 1),
    )
    primal = Program({"energy": lagrangian})
    weights = transpose_program(
        primal, ("energy",), inputs=("overlap", "hcore", "eri")
    ).program
    tangent = linearize(weights, state_names).program
    combined = Program({**weights.outputs, **tangent.outputs})
    # linearize retains primal roots in its output inventory. Select the six
    # weight outputs explicitly rather than relying on internal ordering.
    outputs = (
        "bar_overlap",
        "bar_hcore",
        "bar_eri",
        "d_bar_overlap",
        "d_bar_hcore",
        "d_bar_eri",
    )
    source = emit_scalar_cpp(
        combined,
        function_name="hf_source_weights",
        input_order=(
            *state_names,
            "bar_energy",
            *("d_" + name for name in state_names),
        ),
        output_order=outputs,
    )
    return (
        source
        + r"""
extern "C" int integral_ad_hf_weights(std::size_t n, const double* state,
    const double* direction, double* out, double* dout) {
  const auto n2=n*n;
  const double* da=state; const double* db=state+n2;
  const double* wa=state+2*n2; const double* wb=state+3*n2;
  for (std::size_t a=0;a<n;++a) for (std::size_t b=0;b<n;++b)
    for (std::size_t c=0;c<n;++c) for (std::size_t d=0;d<n;++d) {
      const auto ab=a*n+b, cd=c*n+d, ac=a*n+c, bd=b*n+d;
      double x[10]={da[ab],db[ab],da[cd],db[cd],da[ac],db[ac],da[bd],db[bd],wa[ab],wb[ab]};
      double dx[10]={};
      if (direction) {
        const std::size_t indices[10]={ab,n2+ab,cd,n2+cd,ac,n2+ac,bd,n2+bd,2*n2+ab,3*n2+ab};
        for (unsigned i=0;i<10;++i) dx[i]=direction[indices[i]];
      }
      double y[6];
      if (!hf_source_weights(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],1.0,
          dx[0],dx[1],dx[2],dx[3],dx[4],dx[5],dx[6],dx[7],dx[8],dx[9],
          y[0],y[1],y[2],y[3],y[4],y[5])) return 1;
      const auto index=2*n2+(a*n+b)*n2+c*n+d;
      out[ab]=y[0]; out[n2+ab]=y[1]; out[index]=y[2];
      if (dout) { dout[ab]=y[3]; dout[n2+ab]=y[4]; dout[index]=y[5]; }
    }
  out[2*n2+n2*n2]=1;
  if (dout) dout[2*n2+n2*n2]=0;
  return 0;
}
"""
    )


def emit_dense_hf_terms() -> str:
    """Generate raw ordered J/K summands for the explicit-integral CUDA provider."""
    spec = TensorSpec(role="input")
    a, b, g, x = (
        input_tensor(name, spec) for name in ("alpha", "beta", "coulomb", "exchange")
    )
    program = Program(
        {"j": multiply(add(a, b), g), "ka": multiply(a, x), "kb": multiply(b, x)}
    )
    return emit_scalar_cpp(
        program,
        function_name="explicit_hf_terms",
        input_order=("alpha", "beta", "coulomb", "exchange"),
        output_order=("j", "ka", "kb"),
    ).replace("inline bool", "__device__ inline bool")
