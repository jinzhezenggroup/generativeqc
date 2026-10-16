#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "core/types.hpp"

namespace generativeqc::integrals {

struct CudaEcpNuclearShape {
  std::size_t atoms{}, primitives{}, nao{}, terms{};
};

struct CudaEcpNuclearOptions {
  std::size_t max_workspace_bytes{512ULL << 20}, max_terms{10000};
  std::uint64_t max_ao_evaluations{268'435'456}, max_affine_terms{1ULL << 34},
      max_reduction_terms{1ULL << 34};
  double symmetry_absolute_tolerance{1e-12}, symmetry_relative_tolerance{1e-10};
};

/** Actual immutable normalized AoBasis and residual Gaussian ECP terms.
 * The Coulomb tail belongs to the separate effective-charge attraction source.
 * Orbital Cartesian/spherical components through f use the existing 16-double
 * basis ABI. A live owner must authenticate all arrays, topology and stream;
 * detached pointer labels alone do not establish an accepted molecular state. */
struct CudaEcpNuclearReference {
  const double* basis{};
  const core::EcpTerm* terms{};
  int device_id{-1};
  cudaStream_t stream{};
  cudaError_t (*authenticate)(void*) noexcept {};
  void* owner{};
};

/** Full ordered [AO,AO] local/nonlocal matrices and their geometry directions. */
struct CudaEcpNuclearValues {
  double *local{}, *nonlocal{}, *direction_local{}, *direction_nonlocal{};
};

/** Total physical D/dD, not occupied factors or a spin-specific half density. */
struct CudaEcpNuclearWeights {
  const double *density{}, *direction_density{};
};

/** Four separately gated atom-major vectors, with energy-gradient signs. */
struct CudaEcpNuclearVectors {
  double *gradient_local{}, *gradient_nonlocal{}, *hvp_local{}, *hvp_nonlocal{};
};

struct CudaEcpNuclearWork {
  std::size_t workspace_bytes{}, centers{}, radial_layers_per_action{};
  std::uint64_t ao_evaluations_per_action{}, projection_terms_per_action{}, matrix_affine_terms{},
      contraction_affine_terms{}, reduction_terms_bound{}, attempted_actions{}, value_actions{},
      contraction_actions{}, radial_layers{}, ao_evaluations{}, projection_terms{}, affine_terms{};
};

/** Host-only checked storage/work/AOT admission for both fixed quadratures. */
cudaError_t cuda_ecp_nuclear_workspace(const CudaEcpNuclearShape& shape,
                                       std::span<const unsigned> centers,
                                       const CudaEcpNuclearOptions& options,
                                       std::size_t& bytes) noexcept;

/** Native residual local/projector second response with build-time quadrature.
 * Each radial layer collocates normalized AO jets once, then reuses them for
 * local nodes and nonlocal projections. Compiler AO Taylor AD and stationary
 * bilinear VJP/JVP maps own all derivatives; no left-coordinate source replay,
 * coordinate-major AO-pair tensor, Python, JIT or host numerical contraction.
 * The incumbent coarse/fine matrix/derivative gates precede joint publication.
 * Actions allocate nothing and reauthenticate the original noncapturing stream
 * and owner before publication. Earlier failures preserve every output; CUDA
 * publication faults may leave partial copies. Source/arena outlive this
 * exclusive synchronous nonmovable consumer. This is not a stationary Hessian.
 */
class CudaEcpNuclearResponse {
 public:
  CudaEcpNuclearResponse();
  ~CudaEcpNuclearResponse();
  CudaEcpNuclearResponse(const CudaEcpNuclearResponse&) = delete;
  CudaEcpNuclearResponse& operator=(const CudaEcpNuclearResponse&) = delete;
  CudaEcpNuclearResponse(CudaEcpNuclearResponse&&) = delete;
  CudaEcpNuclearResponse& operator=(CudaEcpNuclearResponse&&) = delete;
  cudaError_t prepare(const CudaEcpNuclearShape& shape, std::span<const unsigned> centers,
                      const CudaEcpNuclearOptions& options,
                      const CudaEcpNuclearReference& reference, void* workspace,
                      std::size_t workspace_bytes) noexcept;
  cudaError_t values(const double* direction, const CudaEcpNuclearValues& output,
                     cudaStream_t stream) noexcept;
  cudaError_t contract(const double* direction, const CudaEcpNuclearWeights& weights,
                       const CudaEcpNuclearVectors& output, cudaStream_t stream) noexcept;
  const CudaEcpNuclearWork& work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::integrals
