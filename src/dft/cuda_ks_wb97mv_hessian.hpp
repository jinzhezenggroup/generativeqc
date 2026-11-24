#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "dft/cuda_grid_geometry.hpp"
#include "dft/cuda_ks_one_electron_response.hpp"
#include "dft/cuda_ks_wb97mv_df_nuclear.hpp"
#include "dft/cuda_ks_wb97mv_df_response.hpp"
#include "dft/cuda_ks_wb97mv_ecp_nuclear.hpp"
#include "dft/cuda_ks_wb97mv_nuclear_jk.hpp"
#include "dft/cuda_ks_wb97mv_response.hpp"

namespace generativeqc::dft {

struct CudaKsWb97mvHessianOptions {
  /** Independent response resources only; immutable borrowed SCF storage is
   * accounted by its producer. Every response owner and candidate is charged. */
  std::size_t max_device_bytes{2ULL << 30};
  std::size_t max_assembly_terms{1ULL << 30};
  std::uint64_t max_nuclear_pairs{16'777'216};
  /** Optional repeated-RHS CPKS representation, additionally constrained by
   * the response/aggregate budgets and the rotation-to-RHS amortization gate.
   * The extended ceiling accommodates up to 724 FP64 rotations in eligible
   * Direct RKS; other paths retain their smaller admission limit. Zero
   * retains physical GMRES. */
  std::size_t max_resident_cpks_bytes{4ULL << 20};
  /** Optional full-Hessian frozen exact derivatives, inside max_device_bytes.
   * Stores O(3N*spin*AO^2) first Fock derivatives plus frozen HVP columns,
   * never a four-index derivative tensor. Zero keeps directional replay. */
  std::size_t max_frozen_exact_bytes{8ULL << 20};
  /** Optional dense cache of Cartesian derivatives of H and S. Zero keeps
   * the per-column weighted one-electron replay; HVPs always use that path. */
  std::size_t max_frozen_core_bytes{8ULL << 20};
  double symmetry_absolute_tolerance{2e-7};
  double symmetry_relative_tolerance{2e-7};
  response::CudaCpksOptions cpks{};
  CudaKsWb97mvResponseOptions induced{};
  CudaGridGeometryOptions geometry{.max_ao_replay_terms = 1ULL << 33};
  CudaWb97mvGridOptions grid{};
  integrals::CudaOneElectronOptions one_electron{};
  CudaWb97mvJkOptions exact{};
  CudaWb97mvDfOptions fitted{};
  integrals::CudaDfNuclearOptions raw_df{};
  integrals::CudaEcpNuclearOptions ecp{};
};

/** Select the storage contract at preparation, before any workspace is admitted. */
enum class CudaKsWb97mvHessianMode { action_only, dense_and_action };

enum class CudaKsWb97mvHessianStatus { failed, response_not_converged, converged };

struct CudaKsWb97mvHessianResult {
  CudaKsWb97mvHessianStatus status{CudaKsWb97mvHessianStatus::failed};
  std::size_t completed_directions{};
  response::CudaCpksResult response{};
  const char* failure_check{};  // Static diagnostic; never part of the public ABI.
  bool converged() const noexcept { return status == CudaKsWb97mvHessianStatus::converged; }
};

struct CudaKsWb97mvHessianWork {
  std::size_t device_bytes{}, assembly_terms_per_direction{}, nuclear_pairs{};
  std::size_t cpks_operator_bytes{}, cpks_operator_columns{}, cpks_cached_actions{},
      cpks_physical_checks{};
  std::size_t cpks_proposal_iterations{}, cpks_proposal_seeds{}, cpks_strict_iterations{};
  std::size_t frozen_exact_bytes{}, frozen_exact_columns{}, frozen_exact_contractions{};
  std::size_t frozen_core_bytes{}, frozen_core_columns{}, frozen_core_contractions{};
  std::uint64_t attempted_directions{}, completed_directions{}, published_hvps{},
      published_hessians{}, correction_directions{};
  CudaKsWb97mvResponseWork induced{};
  CudaGridGeometryWork geometry{}, vv10_geometry{};
  CudaWb97mvGridWork grid{}, separate_vv10{};
  CudaKsOneElectronWork one_electron{};
  CudaWb97mvJkWork exact{};
  CudaWb97mvDfWork fitted{};
  integrals::CudaDfNuclearWork raw_df{};
  CudaWb97mvDfWork fitted_range{};
  integrals::CudaDfNuclearWork raw_df_range{};
  integrals::CudaEcpNuclearWork ecp{};
};

/** Stationary compiler-AD nuclear HVP/Hessian on a live accepted KS state.
 * Canonical LDA/PBE/PBE0/B3LYP and WB97M-V can use qualified Direct, full-rank
 * fitted and residual ECP sources. WB97M-V retains the accepted exact or fitted
 * LR operator. RKS/UKS and basis representation follow the producer bindings.
 * Linked closure is mandatory:
 * no Python, runtime compilation, finite differences or exact-for-DF fallback.
 * Preparation allocates explicitly charged independent arenas. Actions allocate
 * nothing and never overwrite ordinary SCF/DF/occupied-projection scratch.
 * Action-only owners omit dense staging and repeated-RHS caches; they reject
 * hessian() without attempting a response or modifying caller storage.
 *
 * Retain the batch and exclude concurrent access for this nonmovable owner's
 * entire lifetime. All calls use the producer device and ordinary stream. HVP
 * output is [atom,xyz]; Hessian is row-major [left_coordinate,right_coordinate].
 * Complete candidates are private until response convergence, finite/symmetry
 * gates and live-owner reauthentication pass. A successful CUDA return alone
 * does not imply response convergence: inspect result.converged(). No symmetry
 * repair is performed. Rejected calls preserve published output, except that a
 * CUDA fault during the final copy invalidates that output.
 */
class CudaKsWb97mvHessian {
 public:
  CudaKsWb97mvHessian();
  ~CudaKsWb97mvHessian();
  CudaKsWb97mvHessian(const CudaKsWb97mvHessian&) = delete;
  CudaKsWb97mvHessian& operator=(const CudaKsWb97mvHessian&) = delete;
  CudaKsWb97mvHessian(CudaKsWb97mvHessian&&) = delete;
  CudaKsWb97mvHessian& operator=(CudaKsWb97mvHessian&&) = delete;

  generativeqc_status prepare(methods::PreparedBatch& batch, std::size_t item,
                              const CudaKsFinalStateToken& expected, CudaKsWb97mvHessianMode mode,
                              const CudaKsWb97mvHessianOptions& options, std::string& detail);
  cudaError_t apply(const double* direction, double* output, cudaStream_t stream,
                    CudaKsWb97mvHessianResult& result) noexcept;
  cudaError_t hessian(double* output, cudaStream_t stream,
                      CudaKsWb97mvHessianResult& result) noexcept;
  CudaKsWb97mvHessianWork work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
