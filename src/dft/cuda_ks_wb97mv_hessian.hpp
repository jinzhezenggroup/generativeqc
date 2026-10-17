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
   * the existing response/aggregate budgets. Zero retains physical GMRES. */
  std::size_t max_resident_cpks_bytes{128ULL << 10};
  double symmetry_absolute_tolerance{2e-7};
  double symmetry_relative_tolerance{2e-7};
  response::CudaCpksOptions cpks{};
  CudaKsWb97mvResponseOptions induced{};
  CudaGridGeometryOptions geometry{};
  CudaWb97mvGridOptions grid{};
  integrals::CudaOneElectronOptions one_electron{};
  CudaWb97mvJkOptions exact{};
  CudaWb97mvDfOptions fitted{};
  integrals::CudaDfNuclearOptions raw_df{};
  integrals::CudaEcpNuclearOptions ecp{};
};

enum class CudaKsWb97mvHessianStatus { failed, response_not_converged, converged };

struct CudaKsWb97mvHessianResult {
  CudaKsWb97mvHessianStatus status{CudaKsWb97mvHessianStatus::failed};
  std::size_t completed_directions{};
  response::CudaCpksResult response{};
  bool converged() const noexcept { return status == CudaKsWb97mvHessianStatus::converged; }
};

struct CudaKsWb97mvHessianWork {
  std::size_t device_bytes{}, assembly_terms_per_direction{}, nuclear_pairs{};
  std::size_t cpks_operator_bytes{}, cpks_operator_columns{}, cpks_cached_actions{},
      cpks_physical_checks{};
  std::uint64_t attempted_directions{}, completed_directions{}, published_hvps{},
      published_hessians{};
  CudaKsWb97mvResponseWork induced{};
  CudaGridGeometryWork geometry{};
  CudaWb97mvGridWork grid{};
  CudaKsOneElectronWork one_electron{};
  CudaWb97mvJkWork exact{};
  CudaWb97mvDfWork fitted{};
  integrals::CudaDfNuclearWork raw_df{};
  integrals::CudaEcpNuclearWork ecp{};
};

/** Stationary compiler-AD WB97M-V nuclear HVP/Hessian on a live accepted KS state.
 * RKS/UKS, exact/DF primary J/K, residual ECP and public basis representations
 * follow the actual producer bindings. Linked derivative closure is mandatory:
 * no Python, runtime compilation, finite differences or exact-for-DF fallback.
 * Preparation allocates explicitly charged independent arenas. Actions allocate
 * nothing and never overwrite ordinary SCF/DF/occupied-projection scratch.
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
                              const CudaKsFinalStateToken& expected,
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
