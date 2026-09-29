#pragma once

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_scf_state.hpp"

namespace generativeqc::scf::cuda_df {
/** Invalidate before argument checks too, so a failed replay cannot expose
 * the preceding successful solve. Epoch saturation fails closed. */
generativeqc_status begin_scf_final_state_solve(CudaDensityFittingJkPlan& plan,
                                                std::string& detail);
generativeqc_status allocate_scf_final_frames(CudaDensityFittingJkPlan& plan,
                                              PersistentScfState& state, std::string& detail);
generativeqc_status reset_scf_final_frames(CudaDensityFittingJkPlan& plan,
                                           PersistentScfState& state,
                                           const std::vector<std::int32_t>& alpha,
                                           const std::vector<std::int32_t>& beta,
                                           std::string& detail);
/** Called before convergence commits D and clears active; UHF alpha is saved
 * before beta overwrites shared temporary coefficients. Capture records the
 * same masked copy that ordinary execution launches. */
void store_scf_final_frame(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                           const double* coefficients, bool beta);
void publish_scf_final_frames(PersistentScfState& state,
                              const std::vector<CudaDensityFittingDeviceScfItem>& results);
}  // namespace generativeqc::scf::cuda_df
