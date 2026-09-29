#pragma once

#include <string>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace generativeqc::scf {
struct CudaDensityFittingJkPlan;
namespace cuda_df {
struct PersistentScfState;

/** Allocate the requested history within the planner's explicit reservation.
 * RHF and joined-spin UHF reuse the existing shared device DIIS arithmetic. */
generativeqc_status allocate_scf_diis(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                      unsigned history, std::string& detail);

/** Refresh the current overlap and clear history generations on every solve. */
generativeqc_status reset_scf_diis(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                   const std::vector<double>& overlap, std::string& detail);

/** Build the physical FDS-SDF residual and replace F only with a DIIS proposal.
 * Call after physical energy evaluation and before eigen/density generation.
 * UHF shares one coefficient vector across its two spin residuals. Inactive
 * items preserve their histories; strict final selection still checks F[D]. */
generativeqc_status apply_scf_diis(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                   std::string& detail);
}  // namespace cuda_df
}  // namespace generativeqc::scf
