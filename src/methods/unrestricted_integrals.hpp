#pragma once

#include <stdexcept>

#include "integrals/correlation_backend.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/cuda_one_electron_gradient.hpp"

namespace generativeqc::methods::detail {

/** Bind selected CUDA adapters at method composition, not inside post-HF.
 * The retained SCF adapters own their existing integral schedules and stream
 * drains; correlation code depends only on the neutral synchronous contract. */
inline integrals::CorrelationIntegralBackend unrestricted_integral_backend() {
#if GENERATIVEQC_HAS_CUDA
  return {
      [](int device, const core::System& orbital, const core::System& auxiliary, bool derivatives) {
        integrals::DensityFittingIntegralData cartesian;
        std::string detail;
        const auto status = scf::build_cuda_density_fitting_integrals(
            device, orbital, auxiliary, cartesian, detail, derivatives);
        if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::length_error(detail);
        if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(detail);
        if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
        return integrals::transform_density_fitting_integrals(cartesian, orbital, auxiliary);
      },
      [](int device, const core::System& system, std::span<const double> overlap,
         std::span<const double> hcore, std::size_t budget) {
        std::vector<double> gradient;
        std::string detail;
        const auto status = scf::execute_cuda_one_electron_gradient(
            device, system, overlap, hcore, hcore,
            scf::cuda_policy::one_electron_derivative_mapping_requested(), budget, gradient,
            detail);
        if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::length_error(detail);
        if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
        return gradient;
      }};
#else
  return {};
#endif
}

}  // namespace generativeqc::methods::detail
