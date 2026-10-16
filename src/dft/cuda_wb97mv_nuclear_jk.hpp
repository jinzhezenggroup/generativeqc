#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "integrals/cuda_one_electron_response.hpp"

namespace generativeqc::dft {

/** Full+LR is the exact WB97M-V operator. Fitted primary J/K is NOT supplied
 * here: a DF owner may consume only the LR correction beside its fitted source.
 * Separate radial parts are exposed for independent scientific qualification. */
enum class CudaWb97mvJkPart { exact, primary, correction, coulomb, short_range, long_range };

struct CudaWb97mvJkShape {
  std::size_t atoms{}, primitives{}, nao{}, spins{};
};

struct CudaWb97mvJkOptions {
  CudaWb97mvJkPart part{CudaWb97mvJkPart::exact};
  std::size_t max_workspace_bytes{512ULL << 20};
  std::size_t max_groups{100'000};
  std::size_t radial_tile{64};  ///< Primitive streaming bound, in [1,64].
  std::uint64_t max_primitive_component_evaluations{67'108'864};
  std::uint64_t max_projection_terms{268'435'456};
  std::uint64_t max_reduction_terms{1'073'741'824};
  double symmetry_absolute_tolerance{1e-12};
  double symmetry_relative_tolerance{1e-10};
};

/** Borrow the actual normalized AoBasis and an immutable live owner. Geometry,
 * radial normalization and public Cartesian/spherical projection coefficients
 * remain on device. The owner authenticates storage/topology/generation as well
 * as its original device and stream. It and the arena outlive synchronous calls. */
struct CudaWb97mvJkReference {
  const double* basis{};
  int device_id{-1};
  cudaStream_t stream{};
  cudaError_t (*authenticate)(void*) noexcept {};
  void* owner{};
};

/** Five jointly published outputs. Fock panels are [spin,AO,AO], vectors are
 * [atom,xyz]. Fock direction is the frozen-density nuclear partial, whereas HVP
 * retains the supplied right density response in the stationary weight JVP.
 * This component alone is not a stationary Hessian or total energy gradient. */
struct CudaWb97mvJkOutput {
  double *energy{}, *fock{}, *fock_direction{}, *gradient{}, *hvp{};
};

struct CudaWb97mvJkWork {
  std::size_t workspace_bytes{}, groups{}, primitive_records_per_action{};
  std::uint64_t component_evaluations_per_action{}, projection_terms_bound_per_action{},
      reduction_terms_per_action{};
  std::uint64_t attempted_actions{}, completed_actions{}, component_evaluations{},
      projection_terms_bound{}, reduction_terms{};
};

/** Host-only checked closure/work/arena admission, without CUDA initialization.
 * Every ordered shell class and raw component must be linked exactly once for
 * each requested radial kernel/omega. No name parsing, discovery or JIT exists. */
cudaError_t cuda_wb97mv_jk_workspace(const CudaWb97mvJkShape& shape,
                                     std::span<const integrals::CudaOneElectronShell> shells,
                                     const CudaWb97mvJkOptions& options,
                                     std::size_t& bytes) noexcept;

/** Native compiler-AD J/K nuclear source, with signed symmetric D and dD.
 * Raw value/gradient/gradient-JVP are evaluated once per primitive component and
 * cached only for the current shell quartet (<=25*10^4 FP64 values through f).
 * Public transforms gather that mandatory bounded cache, never repeat integral
 * work per AO or left coordinate. All numerical contraction and deterministic
 * reductions run on CUDA; actions allocate nothing and invoke no Python/CPU
 * numerical work. Earlier failures preserve all public outputs. A CUDA failure
 * during publication invalidates all five outputs; callers must discard them.
 */
class CudaWb97mvNuclearJk {
 public:
  CudaWb97mvNuclearJk();
  ~CudaWb97mvNuclearJk();
  CudaWb97mvNuclearJk(const CudaWb97mvNuclearJk&) = delete;
  CudaWb97mvNuclearJk& operator=(const CudaWb97mvNuclearJk&) = delete;
  CudaWb97mvNuclearJk(CudaWb97mvNuclearJk&&) = delete;
  CudaWb97mvNuclearJk& operator=(CudaWb97mvNuclearJk&&) = delete;

  cudaError_t prepare(const CudaWb97mvJkShape& shape,
                      std::span<const integrals::CudaOneElectronShell> shells,
                      const CudaWb97mvJkOptions& options, const CudaWb97mvJkReference& reference,
                      void* workspace, std::size_t workspace_bytes) noexcept;
  cudaError_t apply(const double* direction, const double* density, const double* delta_density,
                    const CudaWb97mvJkOutput& output, cudaStream_t stream) noexcept;
  const CudaWb97mvJkWork& work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
