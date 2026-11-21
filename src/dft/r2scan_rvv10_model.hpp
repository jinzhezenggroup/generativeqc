#pragma once

#include "dft/semilocal_family.hpp"
#include "generated_nonlocal_method_parameters.hpp"
#include "r2scan_cuda_domain.hpp"

namespace generativeqc::dft {

/** Match the compiler-owned electronic model, not its selector. Quadrature,
 * backend and precision are checked by the caller that owns those inputs. */
constexpr bool r2scan_rvv10_model_matches(SemilocalFamily family,
                                          const nlc::Vv10Parameters& parameters,
                                          double exchange_scale, double correlation_scale,
                                          bool has_exact_exchange) noexcept {
  return kCudaR2scanVariational && family == SemilocalFamily::R2scan &&
         parameters == nlc::kR2scanRvv10 && exchange_scale == 1 && correlation_scale == 1 &&
         !has_exact_exchange;
}

}  // namespace generativeqc::dft
