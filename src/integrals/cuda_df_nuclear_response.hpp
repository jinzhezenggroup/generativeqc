#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "scf/cuda_density_fitting_basis.hpp"

namespace generativeqc::integrals {

struct CudaDfNuclearShape {
  std::size_t atoms{}, primitives{}, nao{}, naux{}, cartesian_nao{}, cartesian_naux{};
};

/** Integer-only topology in normalized source order. Primitive ranges refer
 * to the combined orbital+auxiliary packing; public and Cartesian ranges are
 * local to each basis. Device metadata is checked against every field before
 * use. Orbital l<=3 and auxiliary l<=4 preserve their separate public ABIs. */
struct CudaDfNuclearShell {
  std::size_t atom{}, angular{}, primitive_begin{}, primitive_count{}, ao_begin{}, ao_count{},
      cartesian_begin{};
};

struct CudaDfNuclearOptions {
  std::size_t max_workspace_bytes{512ULL << 20}, max_groups{100'000}, radial_tile{64};
  std::uint64_t max_component_evaluations{67'108'864}, max_projection_terms{268'435'456},
      max_reduction_terms_bound{1'073'741'824};
  double symmetry_absolute_tolerance{1e-12}, symmetry_relative_tolerance{1e-10};
};

/** The actual immutable DF basis and its enclosing original-stream capability.
 * No numerical basis reconstruction, whitening, occupied factors or J/K scratch
 * is consumed. Authentication must prove the complete view/topology/lifetime. */
struct CudaDfNuclearReference {
  scf::CudaDfReadonlyBasis basis;
  cudaStream_t stream{};
  cudaError_t (*authenticate)(void*) noexcept {};
  void* owner{};
};

/** Full ordered raw arrays: A[AO,AO,aux] and V[aux,aux]. Directions differentiate
 * only geometry, never a projection/normalization gauge. No coordinate-major
 * molecular derivative tensor or fitted four-center tensor is materialized. */
struct CudaDfNuclearValues {
  double *three_center{}, *metric{}, *direction_three_center{}, *direction_metric{};
};

/** Full ordered raw source cotangents, including both AO/metric incidences.
 * No implicit triangular factor, spin convention or frozen-metric assumption.
 * The method's AD supplies weight motion retaining the accepted right dD. */
struct CudaDfNuclearWeights {
  const double *three_center{}, *metric{}, *direction_three_center{}, *direction_metric{};
};

struct CudaDfNuclearVectors {
  double *gradient{}, *hvp{};
};

/** Per-pass admission and cumulative submitted work, not endpoint timings or
 * FLOPs. Records include component-tile replay; primitive-component evaluations
 * do not depend on radial tiling. Projection terms bound sparse mathematical-
 * coordinate contractions. Reduction terms bound primitive-field sums, cache
 * additions, projection accumulation and deterministic scatter for either pass.
 * Failed/revoked actions can submit work without completing a pass. */
struct CudaDfNuclearWork {
  std::size_t workspace_bytes{}, groups{}, radial_records_per_pass{};
  std::uint64_t component_evaluations_per_pass{}, projection_terms_bound_per_pass{},
      reduction_terms_bound_per_pass{}, attempted_passes{}, value_passes{}, contraction_passes{},
      radial_records{}, component_evaluations{}, projection_terms_bound{}, reduction_terms_bound{};
};

/** Host-only exact ordered AOT closure and checked work/storage admission. */
cudaError_t cuda_df_nuclear_workspace(const CudaDfNuclearShape& shape,
                                      std::span<const CudaDfNuclearShell> orbital,
                                      std::span<const CudaDfNuclearShell> auxiliary,
                                      const CudaDfNuclearOptions& options,
                                      std::size_t& bytes) noexcept;

/** Bounded two-pass raw DF nuclear AD, not a stationary Hessian endpoint.
 * values supplies A_R,V_R for the physical inverse/Fock response. After CPKS
 * and moving-weight AD, contract streams the source again to form w I' and
 * w I''R + w_R I'. Both passes are explicitly counted. A mandatory shell cache
 * evaluates each primitive component once per pass; public expansion gathers
 * never repeat primitive work or introduce a left-coordinate source traversal.
 *
 * All numerical work uses linked compiler AD on CUDA; actions allocate nothing,
 * never compile, switch devices, invoke Python or contract numerics on CPU.
 * Original-stream/capture/device/live-owner, finite/symmetry and alias checks
 * precede joint publication. Rejected numerical/revocation paths preserve all
 * outputs; a CUDA fault during publication can leave partial data. The source
 * and external arena outlive this exclusive, nonmovable synchronous consumer.
 */
class CudaDfNuclearResponse {
 public:
  CudaDfNuclearResponse();
  ~CudaDfNuclearResponse();
  CudaDfNuclearResponse(const CudaDfNuclearResponse&) = delete;
  CudaDfNuclearResponse& operator=(const CudaDfNuclearResponse&) = delete;
  CudaDfNuclearResponse(CudaDfNuclearResponse&&) = delete;
  CudaDfNuclearResponse& operator=(CudaDfNuclearResponse&&) = delete;
  cudaError_t prepare(const CudaDfNuclearShape& shape, std::span<const CudaDfNuclearShell> orbital,
                      std::span<const CudaDfNuclearShell> auxiliary,
                      const CudaDfNuclearOptions& options, const CudaDfNuclearReference& reference,
                      void* workspace, std::size_t workspace_bytes) noexcept;
  cudaError_t values(const double* direction, const CudaDfNuclearValues& output,
                     cudaStream_t stream) noexcept;
  cudaError_t contract(const double* direction, const CudaDfNuclearWeights& weights,
                       const CudaDfNuclearVectors& output, cudaStream_t stream) noexcept;
  const CudaDfNuclearWork& work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::integrals
