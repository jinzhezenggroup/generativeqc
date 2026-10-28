#pragma once

namespace generativeqc::dft {
/** Shared scalar recomposition for host and device source transports. */
#if defined(__CUDACC__)
__host__ __device__
#endif
    inline void mixed_rsh_exchange_sources(double full, double correction, double fraction,
                                           double& short_range, double& long_range) {
  short_range = full - fraction * correction;
  long_range = (1.0 + fraction) * correction;
}

}  // namespace generativeqc::dft
