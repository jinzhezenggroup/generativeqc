#pragma once

#include "scf/cuda_density_fitting.hpp"

namespace generativeqc::scf {

/** Reserve complete optional device storage before sizing raw fallback tiles.
 * Pure native topology census; no GPU call, integral work or large allocation.
 * Unsupported angular domains and insufficient capacity return zero. */
std::size_t cuda_df_md_coulomb_reservation(const core::System& orbital,
                                           const core::System& auxiliary, std::size_t value_budget);

/** Optional installation on a newly created single-system streamed value plan.
 * The normalized systems must be the exact inputs used to create its source.
 * This internal seam is called before any SCF state/graph can borrow the plan.
 * Full-rank metrics and orbital <= f / auxiliary <= g are admitted. Unsupported
 * domains and capacity refusals return success with an explanatory detail and
 * leave the bounded incumbent intact; invalid calls and device errors propagate.
 * All added storage is charged to diagnostics. The caller supplies the complete
 * value-device cap and a separate cap for resident host state plus setup scratch.
 * The preparation adapter restricts admission to pure Coulomb consumers.
 */
generativeqc_status try_install_cuda_df_md_coulomb(
    CudaDensityFittingJkPlan* plan, const core::System& orbital, const core::System& auxiliary,
    std::size_t device_budget, std::size_t host_budget,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail);

}  // namespace generativeqc::scf
