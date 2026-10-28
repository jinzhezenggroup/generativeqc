#pragma once

namespace generativeqc::dft {

/** Physical-spin point transport shared by XC and stationary geometry.
 * Kinetic coefficients contain half-vtau, matching the AO product convention.
 * Values are unweighted; consumers apply the actual quadrature and raw Becke
 * measures. A producer must publish valid=false on invalid point arithmetic.
 * This trivially copied tile lives on the producer/consumer's borrowed stream.
 */
struct DevicePointValue {
  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};
  bool valid{true};
};

}  // namespace generativeqc::dft
