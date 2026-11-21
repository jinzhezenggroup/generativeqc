#pragma once

#include <cstddef>

#include "dft/grid.hpp"
#include "r2scan_cuda_domain.hpp"
#include "scf/types.hpp"

namespace generativeqc::dft {

/** Select the variational model where its native derivative inventory is qualified.
 * Unsupported scientific topologies retain the incumbent energy and force;
 * requested properties and transient resource availability cannot change it. */
inline bool use_cuda_r2scan_variational(const scf::ScfOptions& options, const GridSpec& grid,
                                        std::size_t maximum_angular) noexcept {
  return options.semilocal_exchange_scale == 1.0 && options.semilocal_correlation_scale == 1.0 &&
         grid.partition_iterations == 3 && maximum_angular <= 3U &&
         use_cuda_r2scan_variational(
             options.precision_mode.value_or(GENERATIVEQC_PRECISION_FP64) ==
                 GENERATIVEQC_PRECISION_FP64,
             options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused,
             options.application_precision_budget[0]);
}

}  // namespace generativeqc::dft
