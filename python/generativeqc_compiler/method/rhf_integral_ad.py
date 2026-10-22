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
