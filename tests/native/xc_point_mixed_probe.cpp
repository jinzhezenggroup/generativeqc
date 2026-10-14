#include "dft/xc_point_mixed.hpp"

extern "C" int mixed_point(const double* inputs, double* output, int mixed) {
  double rho[2]{inputs[0], inputs[4]};
  double gradient[2][3]{{inputs[1], inputs[2], inputs[3]}, {inputs[5], inputs[6], inputs[7]}};
  const auto candidate = mixed == 2 ? generativeqc::dft::point::evaluate_pbe_full_mixed(
                                          true, rho, gradient, inputs[8], inputs[9])
                                    : generativeqc::dft::point::evaluate_pbe_restricted_mixed(
                                          rho, gradient, inputs[8], inputs[9]);
  const auto value =
      mixed ? candidate.value
            : generativeqc::dft::point::evaluate(true, rho, gradient, inputs[8], inputs[9]);
  output[0] = value.energy;
  for (unsigned spin = 0; spin < 2; ++spin) {
    output[1 + spin * 4] = value.rho[spin];
    for (unsigned axis = 0; axis < 3; ++axis)
      output[2 + spin * 4 + axis] = value.gradient[spin][axis];
  }
  return !value.valid ? -1 : mixed && candidate.fp32_correlation ? 1 : 0;
}
