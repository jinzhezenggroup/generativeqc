#pragma once

#include <cuda_runtime_api.h>

#include <functional>

#include "runtime/residency_observer.hpp"
#include "scf/provider_initialization_teardown.hpp"
#include "scf/types.hpp"

namespace generativeqc::scf::cuda_execution {

/** Result of constructing one reusable RHF/UHF host-controlled CUDA Graph. */
struct RhfGraphCaptureResult {
  generativeqc_status body_status{GENERATIVEQC_STATUS_SUCCESS};
  cudaError_t cuda_error{cudaSuccess};

  bool ok() const noexcept {
    return body_status == GENERATIVEQC_STATUS_SUCCESS && cuda_error == cudaSuccess;
  }
};

/**
 * Own the reusable CUDA Graph executables for one direct-HF bucket.
 *
 * Scientific launch order remains in the HF driver. This owner is limited to
 * capture lifecycle, graph instantiation/upload, replay, and teardown so graph
 * mechanics can change without rebuilding the direct numerical orchestration.
 * Callback exceptions end capture and release any abandoned graph before rethrow;
 * the borrowed stream remains reusable by teardown or a later retry.
 */
class RhfIterationGraphs {
 public:
  RhfIterationGraphs() = default;
  RhfIterationGraphs(const RhfIterationGraphs&) = delete;
  RhfIterationGraphs& operator=(const RhfIterationGraphs&) = delete;
  ~RhfIterationGraphs();

  /** Retire captured borrowers after their stream has been drained. */
  void reset() noexcept;
  static ProviderOwnerSchema provider_schema();
  ProviderOwnerInventory provider_inventory() const;
  ProviderInitializationTeardownStep close_provider_operation(const std::string&);
  void enter_provider_retirement(
      std::function<void()> require_current,
      std::function<void(const std::string&, std::optional<std::uint32_t>)> failure);
  void mark_provider_closed();
  /** A phase-local execution may initialize the bucket without a retained graph. */
  bool has_iteration() const noexcept { return iteration_graph_exec_ != nullptr; }
  /** Read-only lifetime identity for verifying replay without exposing handles. */
  std::uint64_t iteration_generation() const noexcept { return iteration_generation_; }
  /** Successful submissions, not inferred iteration counts or capture counts. */
  std::uint64_t iteration_launches() const noexcept { return iteration_launches_; }
  std::uint64_t post_eigensolver_launches() const noexcept { return post_eigensolver_launches_; }

  RhfGraphCaptureResult capture_iteration(int device_id, cudaStream_t stream, bool device_launch,
                                          const std::function<generativeqc_status()>& body);
  RhfGraphCaptureResult capture_post_eigensolver(int device_id, cudaStream_t stream,
                                                 const std::function<generativeqc_status()>& body);

  cudaError_t launch_iteration(cudaStream_t stream) const noexcept;
  cudaError_t launch_post_eigensolver(cudaStream_t stream) const noexcept;

 private:
  RhfGraphCaptureResult capture(cudaStream_t stream, cudaGraph_t& graph,
                                cudaGraphExec_t& executable, std::uint64_t& generation,
                                runtime::ResidencyGraphRole role,
                                unsigned long long instantiate_flags, bool synchronize_before,
                                const std::function<generativeqc_status()>& body);
  static void destroy(cudaGraph_t& graph, cudaGraphExec_t& executable, std::uint64_t& generation,
                      runtime::ResidencyGraphRole role) noexcept;
  cudaError_t launch(cudaGraph_t graph, cudaGraphExec_t executable, std::uint64_t generation,
                     runtime::ResidencyGraphRole role, cudaStream_t stream) const noexcept;

  int device_id_{-1};
  cudaGraph_t iteration_graph_{};
  cudaGraphExec_t iteration_graph_exec_{};
  std::uint64_t iteration_generation_{};
  mutable std::uint64_t iteration_launches_{};
  mutable std::uint64_t post_eigensolver_launches_{};
  unsigned long long iteration_flags_{};
  cudaGraph_t post_eigensolver_graph_{};
  cudaGraphExec_t post_eigensolver_graph_exec_{};
  std::uint64_t post_eigensolver_generation_{};
  // Identity-only aliases for an observation emitted after both actual destroys
  // succeed; these fields never acquire ownership of another CUDA object.
  cudaGraph_t retiring_iteration_graph_{}, retiring_post_graph_{};
  cudaGraphExec_t retiring_iteration_exec_{}, retiring_post_exec_{};
  bool provider_managed_{}, provider_closed_{};
  std::function<void()> provider_current_;
  std::function<void(const std::string&, std::optional<std::uint32_t>)> provider_failure_;
};

}  // namespace generativeqc::scf::cuda_execution
