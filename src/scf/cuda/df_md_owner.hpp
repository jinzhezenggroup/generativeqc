#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

#include "core/types.hpp"
#include "generativeqc/generativeqc.h"

namespace generativeqc::scf::cuda_execution {

struct CudaDfMdOwner;

/** Optional primal omission certificate. The enclosing full-rank plan supplies
 * its actual device eigenvectors/eigenvalues and an absolute per-J-entry budget.
 * Zero, missing factors or an unrepresentable bound retain exact-nonzero work.
 * The factors are borrowed only during preparation, before graph capture. */
struct CudaDfMdScreening {
  double max_j_error{};
  const double *eigenvectors{}, *eigenvalues{};
};

/** Optional owner's complete additional storage. The caller must reserve this
 * in its enclosing budget; these values exclude the borrowed input/output
 * vectors, metric factor/solver, stream and other source/SCF state. */
struct CudaDfMdResources {
  std::size_t device_bytes{}, host_resident_bytes{}, host_peak_bytes{};
  std::size_t orbital_primitives{}, auxiliary_primitives{}, nbf{}, naux{};
  double max_j_omission{}, source_norm_limit{}, projection_error_per_primitive{};
};

/** Prepare a strict-FP64 single-system DF charge/J owner.
 * Inputs carry native radial normalization. No integral oracle, public-AO
 * three-center tensor, metric solve or force calculation occurs in preparation.
 * Optional primal screening certifies a cumulative final-J omission allowance
 * using the supplied metric factors; derivatives retain every nonzero source.
 * Raised Hermite potentials and current-source compaction are budgeted and
 * refreshed on every charge/force call, including changed-density graph replay.
 * The borrowed stream must outlive the owner and all captured graphs using it.
 * Creation is forbidden inside capture; replay methods are allocation-free.
 * On refusal/failure, *owner remains null and all partial allocations retire.
 * The optional prepared pure-J adapter owns admission and enclosing budgets. */
generativeqc_status create_cuda_df_md_owner(int device, cudaStream_t stream,
                                            const core::System& orbital,
                                            const core::System& auxiliary,
                                            std::size_t device_budget, std::size_t host_budget,
                                            CudaDfMdOwner** owner, CudaDfMdResources& resources,
                                            std::string& detail, CudaDfMdScreening screening = {});

/** Drain the borrowed stream before releasing its resident storage; restore
 * the caller's previous current device. The stream must still be alive. */
void destroy_cuda_df_md_owner(CudaDfMdOwner* owner) noexcept;

/** rho_P=sum_ij (ij|P) D_ij. Overwrites all naux public outputs and refreshes
 * the entire current-density Hermite state. D may be nonsymmetric/indefinite.
 * Caller-owned device buffers must be disjoint and valid for this stream. */
cudaError_t cuda_df_md_charge(CudaDfMdOwner* owner, const double* density, double* charge);

/** J_ij=sum_P (ij|P) v_P for the caller's metric-applied vector. Overwrites
 * the complete row-major J and is independent of any prior charge invocation.
 * Neither entry point retains borrowed input/output pointer identity. */
cudaError_t cuda_df_md_project(CudaDfMdOwner* owner, const double* potential, double* coulomb);

/** Add coefficient * sum_ijP D_ij v_P d(ij|P)/dR to an existing device
 * atom gradient. D and v are fixed physical inputs, independently refreshed on
 * every call; no metric derivative is included. The caller owns the full-rank
 * metric response and clears/retains the output as appropriate. Allocation-free
 * and stream-ordered, including captured replay. Nonsymmetric D is supported. */
cudaError_t cuda_df_md_three_center_gradient(CudaDfMdOwner* owner, const double* density,
                                             const double* potential, double coefficient,
                                             double* gradient);

}  // namespace generativeqc::scf::cuda_execution
