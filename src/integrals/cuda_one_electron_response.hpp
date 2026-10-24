#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace generativeqc::integrals {

struct CudaOneElectronShape {
  std::size_t atoms{}, primitives{}, nao{};
};

/** Integer-only shell view of the immutable normalized AoBasis packing.
 * Shells cover primitive and public-AO ranges exactly once, in native order.
 * Cartesian and real-spherical public counts are admitted through f; actual
 * projection coefficients remain in the borrowed device basis, not this view. */
struct CudaOneElectronShell {
  std::size_t atom{}, angular{}, primitive_begin{}, primitive_count{}, ao_begin{}, ao_count{};
};

struct CudaOneElectronOptions {
  std::size_t max_workspace_bytes{512ULL << 20};
  std::size_t max_jobs{100'000};
  std::size_t radial_tile{64};  ///< Bounded streaming tile, in [1,64].
  /** Optional independent primitive chains with an ordered publication fold.
   * These limits are subordinate to max_workspace_bytes; zero streams or zero
   * extra bytes retains the original scalar CUDA schedule. No runtime codegen. */
  unsigned parallel_streams{32};
  std::size_t max_parallel_tasks{2048};
  std::size_t max_parallel_workspace_bytes{64ULL << 20};
  std::uint64_t max_primitive_component_evaluations{67'108'864};
  std::uint64_t max_projection_terms{268'435'456};
  double symmetry_absolute_tolerance{1e-12};
  double symmetry_relative_tolerance{1e-10};
};

/** Actual immutable normalized basis and effective point-charge source.
 * Basis uses AoBasis's centers/primitives/16-double public-AO records. Charges
 * are device [atom] effective nuclear charges, including any ECP core removal.
 * The original live owner must authenticate exact storage, topology, device,
 * stream and generation. It and the arena outlive all synchronous calls.
 * This source does not include local/nonlocal ECP residual projectors. */
struct CudaOneElectronReference {
  const double *basis{}, *charges{};
  int device_id{-1};
  cudaStream_t stream{};
  cudaError_t (*authenticate)(void*) noexcept {};
  void* owner{};
};

/** Externally supplied physical stationary weights; signed/indefinite allowed.
 * Every panel is a finite symmetric public-AO matrix. A method normally supplies
 * total D / dD and negative total W / dW. No HF factors or spin policy are
 * introduced by this integral owner. */
struct CudaOneElectronWeights {
  const double *core{}, *core_direction{}, *overlap{}, *overlap_direction{};
};

struct CudaOneElectronOutput {
  double *core{}, *overlap{}, *core_direction{}, *overlap_direction{}, *gradient{}, *hvp{};
};

struct CudaOneElectronWork {
  std::size_t workspace_bytes{}, jobs{}, primitive_records_per_action{};
  std::uint64_t component_evaluations_per_action{}, projection_terms_per_action{};
  std::uint64_t attempted_actions{}, completed_actions{}, component_evaluations{},
      projection_terms{};
  std::size_t parallel_workspace_bytes{}, parallel_tasks{};
  std::uint64_t parallel_actions{};
};

/** Checked host-only closure/work/arena admission; does not initialize CUDA.
 * Every needed ordered shell class/component must already be in the linked AOT
 * inventory. Missing or overlapping closure is rejected before kernel work.
 * All mathematical centers, including coincident physical atoms, are retained. */
cudaError_t cuda_one_electron_workspace(const CudaOneElectronShape& shape,
                                        std::span<const CudaOneElectronShell> shells,
                                        const CudaOneElectronOptions& options,
                                        std::size_t& bytes) noexcept;

/** Compiler-AD S and T+effective-charge V nuclear source, not a Hessian endpoint.
 * Preparation owns bounded integer scheduling metadata only. All primitive
 * products, public/spherical transforms, derivatives and weighted contractions
 * run on CUDA using build-time linked compiler AD. Actions allocate nothing,
 * never switch devices, JIT, call Python, or contract numerical data on CPU.
 * Public matrix values/directions and atom-major gradient/HVP are staged and
 * jointly reauthenticated before publication. A CUDA publication failure
 * invalidates all six outputs; earlier failures leave all six unchanged.
 */
class CudaOneElectronResponse {
 public:
  CudaOneElectronResponse();
  ~CudaOneElectronResponse();
  CudaOneElectronResponse(const CudaOneElectronResponse&) = delete;
  CudaOneElectronResponse& operator=(const CudaOneElectronResponse&) = delete;
  CudaOneElectronResponse(CudaOneElectronResponse&&) = delete;
  CudaOneElectronResponse& operator=(CudaOneElectronResponse&&) = delete;

  cudaError_t prepare(const CudaOneElectronShape& shape,
                      std::span<const CudaOneElectronShell> shells,
                      const CudaOneElectronOptions& options,
                      const CudaOneElectronReference& reference, void* workspace,
                      std::size_t workspace_bytes) noexcept;
  cudaError_t apply(const double* direction, const CudaOneElectronWeights& weights,
                    const CudaOneElectronOutput& output, cudaStream_t stream) noexcept;
  const CudaOneElectronWork& work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::integrals
