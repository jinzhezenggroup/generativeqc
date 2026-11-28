#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "response/cuda_gmres.hpp"

namespace generativeqc::response {

enum class CudaCpksLayout { row_major, column_major };
enum class CudaCpksStatus { running, converged, numerical_failure, linear_solve_failed };
enum class CudaCpksResidentPrecision { fp64, fp32_proposal };

struct CudaCpksShape {
  std::size_t nao{};
  unsigned spins{1};
  std::size_t occupied[2]{};
};

struct CudaCpksOptions {
  CudaGmresOptions gmres{};
  CudaCpksResidentPrecision resident_precision{CudaCpksResidentPrecision::fp64};
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

/** Optional FP64 cached-operator seed (e.g. a batched local XC projection).
 * Column-major [rotation,rotation]. The matching build provider supplies
 * exactly the missing terms. NotSupported is a settled refusal: no work may
 * still borrow the target and the ordinary build overwrites every entry. */
using CudaCpksOperatorSeed = cudaError_t (*)(void*, const CudaCpksShape&, const CudaCpksReference&,
                                             double*, cudaStream_t) noexcept;

/** Optional initial guess from the authenticated, immutable FP64 operator.
 * The strict solver still checks the residual, and CPKS still replays the
 * physical operator. NotSupported must leave no unsettled borrowed work. */
using CudaCpksRotationGuess = cudaError_t (*)(void*, const double*, const double*, double*,
                                              std::size_t, bool, cudaStream_t) noexcept;

struct CudaCpksResult {
  CudaCpksStatus status{CudaCpksStatus::running};
  CudaGmresResult gmres{};
  std::size_t workspace_bytes{};
  std::size_t induced_fock_actions{};
  std::size_t proposal_iterations{}, proposal_operator_actions{};
  bool proposal_seeded{};
  bool converged() const noexcept { return status == CudaCpksStatus::converged; }
};

/** Sticky receipt for one dense transaction. A caller zeroes this device
 * storage before submission and reads it once with the outer assembly audit. */
struct CudaCpksDenseReceipt {
  CudaGmresDeviceReceipt last;
  std::size_t iterations, cached_actions, physical_checks;
  CudaCpksStatus failure;
  int revoke_cache, upstream_flags;
};

/** Registered by the dense owner for private FP64 output only. Every solve
 * must use the same reference, provider, stream, workspace, options and panel
 * addresses. On failure the outer status becomes sticky for that column.
 * No published storage may be bound here. The owner drains all queued work
 * before releasing any captured buffers or the aggregate receipt. */
struct CudaCpksDenseReplay {
  CudaCpksDenseReceipt* receipt;
  int* column_status;
  const int* upstream_detail;
  CudaGmresReplay solver;
};

/** Optional bounded representation of one immutable, linear CPKS operator.
 * The caller owns the device storage and keeps the exact reference/provider
 * alive and unchanged. Reset explicitly before changing either source. Columns
 * are built on CUDA from physical unit rotations, never from fitted residuals.
 * The FP32 format compresses only this projection. Its optional proposal
 * always hands off to physical GMRES, and both formats audit final residuals.
 * This noncopyable lease is for repeated RHSs, not a detached numerical oracle. */
class CudaCpksOperatorCache {
 public:
  CudaCpksOperatorCache() = default;
  CudaCpksOperatorCache(const CudaCpksOperatorCache&) = delete;
  CudaCpksOperatorCache& operator=(const CudaCpksOperatorCache&) = delete;
  void reset(void* values, std::size_t bytes,
             CudaCpksResidentPrecision precision = CudaCpksResidentPrecision::fp64) noexcept {
    values_ = values;
    bytes_ = bytes;
    precision_ = precision;
    ready_ = false;
    build_refused_ = false;
    columns_ = actions_ = checks_ = 0;
    seed_ = nullptr;
    complement_ = nullptr;
    guess_ = nullptr;
    guess_owner_ = nullptr;
    clear_metric_projection();
  }
  /** Install a bounded FP64 builder before the lease builds any columns. An
   * absent complement denotes a complete projection: if refused, use the
   * physical operator directly instead of building expensive unit columns. */
  void set_build_decomposition(CudaCpksOperatorSeed seed, CudaCpksInducedFock complement) noexcept {
    if (!ready_) {
      seed_ = seed;
      complement_ = complement;
    }
  }
  void set_rotation_guess(CudaCpksRotationGuess guess, void* owner) noexcept {
    if (!ready_) {
      guess_ = guess;
      guess_owner_ = owner;
    }
  }
  /** Borrow the projected XC response to this exact overlap direction for
   * the next solve's initial RHS only. The final Fock still uses provider_. */
  void set_metric_projection(const double* overlap, const double* xc_ov,
                             CudaCpksInducedFock complement) noexcept {
    metric_overlap_ = overlap;
    metric_xc_ov_ = xc_ov;
    metric_complement_ = complement;
  }
  void clear_metric_projection() noexcept { set_metric_projection(nullptr, nullptr, nullptr); }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t actions() const noexcept { return actions_; }
  std::size_t physical_checks() const noexcept { return checks_; }
  void collect_dense(const CudaCpksDenseReceipt& receipt) noexcept {
    actions_ += receipt.cached_actions;
    checks_ += receipt.physical_checks;
    if (receipt.revoke_cache) ready_ = false;
  }

 private:
  void* values_{};
  CudaCpksResidentPrecision precision_{CudaCpksResidentPrecision::fp64};
  std::size_t bytes_{}, dimension_{}, columns_{}, actions_{}, checks_{};
  CudaCpksShape shape_{};
  CudaCpksReference reference_{};
  CudaCpksInducedFock provider_{};
  CudaCpksOperatorSeed seed_{};
  CudaCpksInducedFock complement_{};
  CudaCpksRotationGuess guess_{};
  void* guess_owner_{};
  const double *metric_overlap_{}, *metric_xc_ov_{};
  CudaCpksInducedFock metric_complement_{};
  void* context_{};
  bool ready_{}, build_refused_{};
  friend cudaError_t solve_cuda_cpks(const CudaCpksShape&, const CudaCpksOptions&,
                                     const CudaCpksReference&, const double*, const double*,
                                     CudaCpksInducedFock, void*, double*, double*, void*,
                                     std::size_t, cudaStream_t, CudaCpksResult&,
                                     CudaCpksOperatorCache*, const int*,
                                     CudaCpksDenseReplay*) noexcept;
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
 * If supplied, upstream_status is an aligned, disjoint device audit on stream.
 * A nonzero value rejects the RHS at the existing setup checkpoint.
 * The internal dense_replay mode defers numerical validation and publication
 * to the private transaction receipt. In that mode a successful return means
 * submitted, not converged, and the CPU result contains only setup metadata.
 * A CUDA failure during the final copies invalidates BOTH outputs. This is a
 * response component, not an advertised complete molecular Hessian endpoint.
 */
cudaError_t solve_cuda_cpks(const CudaCpksShape& shape, const CudaCpksOptions& options,
                            const CudaCpksReference& reference, const double* frozen_fock,
                            const double* overlap_direction, CudaCpksInducedFock induced_fock,
                            void* context, double* density_direction,
                            double* weighted_density_direction, void* workspace,
                            std::size_t workspace_bytes, cudaStream_t stream,
                            CudaCpksResult& result, CudaCpksOperatorCache* operator_cache = nullptr,
                            const int* upstream_status = nullptr,
                            CudaCpksDenseReplay* dense_replay = nullptr) noexcept;

}  // namespace generativeqc::response
