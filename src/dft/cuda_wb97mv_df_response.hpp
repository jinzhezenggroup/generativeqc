#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace generativeqc::dft {

struct CudaWb97mvDfShape {
  std::size_t nao{}, naux{}, spins{};
};

struct CudaWb97mvDfOptions {
  std::size_t max_workspace_bytes{512ULL << 20};
  std::size_t max_affine_term_visits{100'000'000};
  double symmetry_absolute_tolerance{1e-12};
  double symmetry_relative_tolerance{1e-10};
};

/** Immutable physical DF sources on the original owner stream. Density is
 * [spin,AO,AO], raw A is [AO,AO,aux], W is [out_aux,source_aux], and metric
 * eigenvalues are ascending. Authentication binds all pointers, shapes, cutoff
 * and the accepted KS state to their live owners. In particular W need not be
 * symmetric: the physical inverse is W^T W, never W W^T.
 *
 * Full rank alone is insufficient for nuclear response: the smallest eigenvalue
 * must exceed both 1.1 times the forward relative cutoff and 64 eps times the
 * largest eigenvalue. Rank-truncated/near-cutoff metrics fail closed. */
struct CudaWb97mvDfReference {
  const double *density{}, *three_center{}, *whitening{}, *metric_eigenvalues{};
  double metric_relative_threshold{};
  int device_id{-1};
  cudaStream_t stream{};
  cudaError_t (*authenticate)(void*) noexcept {};
  void* owner{};
};

/** Seven jointly staged outputs. Frozen-D Fock motion includes only dA,dV;
 * source-weight motion includes dD as well. Raw metric cotangents differentiate
 * the full inverse solve, not a frozen fitting metric or whitening gauge.
 * These are fitted-primary components, excluding exact LR, XC and nuclear
 * integral derivatives. No stationary Hessian is supplied by this component. */
struct CudaWb97mvDfOutput {
  double *energy{}, *fock{}, *fock_direction{}, *weight_three_center{}, *weight_metric{},
      *weight_direction_three_center{}, *weight_direction_metric{};
};

struct CudaWb97mvDfWork {
  /** Scheduled scalar-term bounds, including primal validation dependencies;
   * affine_terms charges a full action when its first map is submitted. This
   * is semantic admission evidence, not FLOPs or measured endpoint timing. */
  std::size_t workspace_bytes{}, reference_terms{}, action_terms{};
  std::uint64_t attempted_actions{}, completed_actions{}, affine_terms{};
};

/** Host-only checked admission using the literal compiler-emitted schedule. */
cudaError_t cuda_wb97mv_df_workspace(const CudaWb97mvDfShape& shape,
                                     const CudaWb97mvDfOptions& options,
                                     std::size_t& bytes) noexcept;

/** Native full-rank fitted-primary AD, with borrowed explicitly charged arena.
 * Preparation computes the immutable inverse and reference VJPs. Actions never
 * allocate, compile, select devices, or execute Python/CPU numerical work.
 * Arena and outputs must be disjoint from inputs and each other; signed
 * symmetric D/dD are legal. Read-only inputs may share storage.
 * All calls are synchronous on the original stream, reject capture and recheck
 * authentication before publication. Numerical/admission/revocation failures
 * preserve all outputs. CUDA errors during publication may leave partial data.
 * The source owner and arena must outlive this exclusive, nonmovable consumer.
 */
class CudaWb97mvDfResponse {
 public:
  CudaWb97mvDfResponse();
  ~CudaWb97mvDfResponse();
  CudaWb97mvDfResponse(const CudaWb97mvDfResponse&) = delete;
  CudaWb97mvDfResponse& operator=(const CudaWb97mvDfResponse&) = delete;
  CudaWb97mvDfResponse(CudaWb97mvDfResponse&&) = delete;
  CudaWb97mvDfResponse& operator=(CudaWb97mvDfResponse&&) = delete;

  cudaError_t prepare(const CudaWb97mvDfShape& shape, const CudaWb97mvDfOptions& options,
                      const CudaWb97mvDfReference& reference, void* workspace,
                      std::size_t workspace_bytes) noexcept;
  cudaError_t apply(const double* delta_density, const double* delta_three_center,
                    const double* delta_metric, const CudaWb97mvDfOutput& output,
                    cudaStream_t stream) noexcept;
  const CudaWb97mvDfWork& work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::dft
