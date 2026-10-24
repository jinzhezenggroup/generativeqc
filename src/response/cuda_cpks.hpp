#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "response/cuda_gmres.hpp"

namespace generativeqc::response {

enum class CudaCpksLayout { row_major, column_major };
enum class CudaCpksStatus { running, converged, numerical_failure, linear_solve_failed };

struct CudaCpksShape {
  std::size_t nao{};
  unsigned spins{1};
  std::size_t occupied[2]{};
};

struct CudaCpksOptions {
  CudaGmresOptions gmres{};
  std::size_t max_workspace_bytes{512ULL << 20};
  double symmetry_absolute_tolerance{1e-12};
  double symmetry_relative_tolerance{1e-10};
};

/** Authenticate a live prepared owner/token, without exporting numerical panels.
 * Called before any device work and again before publication. The owner must
 * keep its lease, device, stream and immutable arrays alive throughout the call.
 * A detached token label or a callback that merely accepts labels is NOT a
 * production state proof. The component probe uses its own immutable test owner.
 */
using CudaCpksAuthenticate = cudaError_t (*)(void*) noexcept;

struct CudaCpksReference {
  const double* coefficients{};
  const double* orbital_energies{};
  CudaCpksLayout coefficient_layout{CudaCpksLayout::column_major};
  cudaStream_t stream{};
  CudaCpksAuthenticate authenticate{};
  void* owner{};
};

/** One native, coupled-spin induced-Fock action, on the borrowed stream.
 * Density and Fock are distinct contiguous [spin,AO,AO] FP64 panels (logical
 * row-major, symmetric). Both spins are passed together, with no spin average.
 * All numerical work must execute on CUDA. No Python callback, CPU contraction,
 * allocation, device/stream switch, or retention of borrowed pointers is allowed.
 * The provider must overwrite the complete output or return a CUDA error.
 */
using CudaCpksInducedFock = cudaError_t (*)(void*, const double*, double*, const CudaCpksShape&,
                                            cudaStream_t) noexcept;

struct CudaCpksResult {
  CudaCpksStatus status{CudaCpksStatus::running};
  CudaGmresResult gmres{};
  std::size_t workspace_bytes{};
  std::size_t induced_fock_actions{};
  bool converged() const noexcept { return status == CudaCpksStatus::converged; }
};

/** Optional bounded representation of one immutable, linear CPKS operator.
 * The caller owns the device storage and keeps the exact reference/provider
 * alive and unchanged. Reset explicitly before changing either source. Columns
 * are built on CUDA from physical unit rotations, never from fitted residuals.
 * Every cached solve still checks its true residual with the physical provider.
 * This noncopyable lease is for repeated RHSs, not a detached numerical oracle. */
class CudaCpksOperatorCache {
 public:
  CudaCpksOperatorCache() = default;
  CudaCpksOperatorCache(const CudaCpksOperatorCache&) = delete;
  CudaCpksOperatorCache& operator=(const CudaCpksOperatorCache&) = delete;
  void reset(double* values, std::size_t bytes) noexcept {
    values_ = values;
    bytes_ = bytes;
    ready_ = false;
    columns_ = actions_ = checks_ = 0;
  }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t actions() const noexcept { return actions_; }
  std::size_t physical_checks() const noexcept { return checks_; }

 private:
  double* values_{};
  std::size_t bytes_{}, dimension_{}, columns_{}, actions_{}, checks_{};
  CudaCpksShape shape_{};
  CudaCpksReference reference_{};
  CudaCpksInducedFock provider_{};
  void* context_{};
  bool ready_{};
  friend cudaError_t solve_cuda_cpks(const CudaCpksShape&, const CudaCpksOptions&,
                                     const CudaCpksReference&, const double*, const double*,
                                     CudaCpksInducedFock, void*, double*, double*, void*,
                                     std::size_t, cudaStream_t, CudaCpksResult&,
                                     CudaCpksOperatorCache*) noexcept;
};

/** Host-only storage requirement; zero denotes an empty rotation space. */
cudaError_t cuda_cpks_operator_workspace(const CudaCpksShape& shape, const CudaCpksOptions& options,
                                         std::size_t& bytes) noexcept;

/** Pure host sizing/admission, with overflow and complete workspace budget gates. */
cudaError_t cuda_cpks_workspace(const CudaCpksShape& shape, const CudaCpksOptions& options,
                                std::size_t& bytes) noexcept;

/** Compiler-owned reference W for the same accepted canonical frame as CPKS.
 * Reuses the primal underlying the weighted-density JVP, with physical RKS/UKS
 * occupations and empty/full spin populations. Output is positive-convention
 * W, not the negative total overlap cotangent used by stationary assembly.
 * The existing cuda_cpks_workspace arena is reused exclusively before/after a
 * solve; no second persistent allocation, induced Fock action or GMRES runs.
 * Numerical, capture, alias, stream and revocation failures preserve output;
 * a CUDA publication fault may leave a partial copy. All numerical work stays
 * on the original CUDA stream, with no Python/JIT/host contraction.
 */
cudaError_t cuda_cpks_reference_weights(const CudaCpksShape& shape, const CudaCpksOptions& options,
                                        const CudaCpksReference& reference, double* output,
                                        void* workspace, std::size_t workspace_bytes,
                                        cudaStream_t stream) noexcept;

/** Native RKS/UKS CPKS plus compiler-AD density and weighted-density response.
 * References must be complete S-orthonormal canonical frames with integer
 * occupations and separated occupied/virtual gaps. Authentication of this
 * stationary-state proof belongs to the live prepared owner. One-spin RKS has
 * occupation two; each UKS block has occupation one. Empty/full spin populations
 * and degenerate same-space energies are supported without gauge divisions.
 * Frozen Fock and overlap directions must be finite and symmetric. Output D/W
 * directions are [spin,AO,AO]. Matrix symmetry is checked, never repaired.
 *
 * Caller-owned workspace, outputs and inputs must be aligned/disjoint as
 * appropriate. No allocations, runtime compiler/interpreter, or CPU numerical
 * fallback. Only discrete status and terminal GMRES scalar receipts reach host.
 * Ordinary noncapturing streams only; capture is rejected before submission.
 * Synchronizes at failure/convergence gates. Numerical or CUDA/provider failure
 * never publishes candidates; both outputs remain unchanged before publication.
 * A CUDA failure during the final copies invalidates BOTH outputs. This is a
 * response component, not an advertised complete molecular Hessian endpoint.
 */
cudaError_t solve_cuda_cpks(const CudaCpksShape& shape, const CudaCpksOptions& options,
                            const CudaCpksReference& reference, const double* frozen_fock,
                            const double* overlap_direction, CudaCpksInducedFock induced_fock,
                            void* context, double* density_direction,
                            double* weighted_density_direction, void* workspace,
                            std::size_t workspace_bytes, cudaStream_t stream,
                            CudaCpksResult& result,
                            CudaCpksOperatorCache* operator_cache = nullptr) noexcept;

}  // namespace generativeqc::response
