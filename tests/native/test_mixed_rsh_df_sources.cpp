#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/mixed_rsh_df_sources.hpp"

namespace {
void require(bool accepted, const char* reason) {
  if (!accepted) throw std::runtime_error(reason);
}
void near(double actual, double expected, const char* reason) {
  require(std::abs(actual - expected) < 1.0e-13, reason);
}
void verify_spin(double factor) {
  // WB97M-V c_short=0.15, c_long=1.0. Fock coefficients include
  // -1/2 in RKS and -1 in UKS; the source algebra is spin-agnostic.
  const double primary = -factor * 0.15;
  const double correction = -factor * 0.85;
  const std::vector<double> fitted{1., 2., 3., 4., 5., 6., 7., 8.};
  const std::vector<double> lr{9., 10.};
  std::vector<double> sources;
  std::string detail;
  require(generativeqc::dft::compose_mixed_rsh_df_integral_sources(
              fitted, lr, 2, primary, correction, sources, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "valid mixed source composition was rejected");
  require(sources.size() == 10 && detail.empty(), "canonical five-row layout not published");
  for (std::size_t i = 0; i < 6; ++i)
    near(sources[i], fitted[i], "one-electron/Pulay/Coulomb source changed");
  const double share = primary / correction;
  for (std::size_t i = 0; i < 2; ++i) {
    near(sources[6 + i], fitted[6 + i] - share * lr[i], "short-range source");
    near(sources[8 + i], (1.0 + share) * lr[i], "long-range source");
    near(sources[6 + i] + sources[8 + i], fitted[6 + i] + lr[i],
         "full DF plus exact LR correction identity");
  }
  sources = {123.};
  require(generativeqc::dft::compose_mixed_rsh_df_integral_sources(
              fitted, lr, 2, primary, 0.0, sources, detail) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              sources.empty(), "zero LR correction silently became a canonical source");
  require(generativeqc::dft::compose_mixed_rsh_df_integral_sources(
              fitted, {lr.data(), 1}, 2, primary, correction, sources, detail) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              sources.empty(), "truncated LR source was admitted");
  require(generativeqc::dft::compose_mixed_rsh_df_integral_sources(
              fitted, lr, 0, primary, correction, sources, detail) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              sources.empty(), "empty coordinate source was admitted");
  auto invalid = fitted;
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  require(generativeqc::dft::compose_mixed_rsh_df_integral_sources(
              invalid, lr, 2, primary, correction, sources, detail) ==
              GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
              sources.empty(), "nonfinite DF source was published");
  invalid = fitted;
  auto invalid_lr = lr;
  invalid_lr[0] = std::numeric_limits<double>::infinity();
  require(generativeqc::dft::compose_mixed_rsh_df_integral_sources(
              invalid, invalid_lr, 2, primary, correction, sources, detail) ==
              GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
              sources.empty(), "nonfinite LR source was published");
}
}  // namespace

int main() {
  try {
    verify_spin(0.5);
    verify_spin(1.0);
    std::cout << "mixed RSH-DF source algebra passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
