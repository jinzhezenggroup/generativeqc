#include "scf/cuda/rhf_graph.hpp"

#include <mutex>

#include "runtime/allocation_measurement.hpp"
#include "runtime/residency_cuda.cuh"

namespace generativeqc::scf::cuda_execution {

void RhfIterationGraphs::destroy(cudaGraph_t& graph, cudaGraphExec_t& executable,
                                 std::uint64_t& generation,
                                 runtime::ResidencyGraphRole role) noexcept {
  if (generation) {
    runtime::observe_residency_graph(runtime::ResidencyGraphEvent::destruction, generation, role, 0,
                                     graph, executable, nullptr);
    generation = 0;
  }
  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
    executable = nullptr;
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
    graph = nullptr;
  }
}

RhfIterationGraphs::~RhfIterationGraphs() { reset(); }

void RhfIterationGraphs::reset() noexcept {
  std::lock_guard<std::mutex> allocation_lock(runtime::allocation_measurement_mutex);
  if (device_id_ >= 0) (void)cudaSetDevice(device_id_);
  destroy(post_eigensolver_graph_, post_eigensolver_graph_exec_, post_eigensolver_generation_,
          runtime::ResidencyGraphRole::hf_post_eigensolver);
  destroy(iteration_graph_, iteration_graph_exec_, iteration_generation_,
          runtime::ResidencyGraphRole::hf_iteration);
}

RhfGraphCaptureResult RhfIterationGraphs::capture(
    cudaStream_t stream, cudaGraph_t& graph, cudaGraphExec_t& executable, std::uint64_t& generation,
    runtime::ResidencyGraphRole role, unsigned long long instantiate_flags, bool synchronize_before,
    const std::function<generativeqc_status()>& body) {
  const runtime::ResidencyExecution source_execution(runtime::ResidencyOwner::hf_graph_setup);
  destroy(graph, executable, generation, role);

  cudaError_t error = synchronize_before
                          ? runtime::residency_stream_synchronize(
                                source_execution, runtime::ResidencyRole::prepare,
                                runtime::ResidencySite::hf_graph_capture_fence, stream)
                          : cudaSuccess;
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  }
  if (error != cudaSuccess) return {GENERATIVEQC_STATUS_SUCCESS, error};

  generativeqc_status body_status = GENERATIVEQC_STATUS_SUCCESS;
  try {
    body_status = body();
  } catch (...) {
    // Preserve the original exception while releasing the stream's capture
    // state. Teardown and a later retry must not inherit an abandoned capture.
    cudaGraph_t abandoned_graph = nullptr;
    (void)cudaStreamEndCapture(stream, &abandoned_graph);
    if (abandoned_graph != nullptr) (void)cudaGraphDestroy(abandoned_graph);
    throw;
  }
  if (body_status != GENERATIVEQC_STATUS_SUCCESS) {
    cudaGraph_t abandoned_graph = nullptr;
    (void)cudaStreamEndCapture(stream, &abandoned_graph);
    if (abandoned_graph != nullptr) (void)cudaGraphDestroy(abandoned_graph);
    return {body_status, cudaSuccess};
  }

  error = cudaStreamEndCapture(stream, &graph);
  if (error != cudaSuccess || graph == nullptr) {
    return {GENERATIVEQC_STATUS_SUCCESS,
            error != cudaSuccess ? error : cudaErrorInvalidResourceHandle};
  }

  error = cudaGraphInstantiate(&executable, graph, instantiate_flags);
  if (error == cudaSuccess) {
    generation = runtime::next_residency_graph_generation();
    runtime::observe_residency_graph(runtime::ResidencyGraphEvent::definition, generation, role,
                                     instantiate_flags, graph, executable, stream);
  }
  if (error == cudaSuccess) error = cudaGraphUpload(executable, stream);
  if (error == cudaSuccess)
    error = runtime::residency_stream_synchronize(source_execution, runtime::ResidencyRole::prepare,
                                                  runtime::ResidencySite::hf_graph_upload_fence,
                                                  stream);
  return {GENERATIVEQC_STATUS_SUCCESS, error};
}

RhfGraphCaptureResult RhfIterationGraphs::capture_iteration(
    int device_id, cudaStream_t stream, bool device_launch,
    const std::function<generativeqc_status()>& body) {
  device_id_ = device_id;
  iteration_flags_ =
      device_launch ? static_cast<unsigned long long>(cudaGraphInstantiateFlagDeviceLaunch) : 0ULL;
  return capture(stream, iteration_graph_, iteration_graph_exec_, iteration_generation_,
                 runtime::ResidencyGraphRole::hf_iteration, iteration_flags_, true, body);
}

RhfGraphCaptureResult RhfIterationGraphs::capture_post_eigensolver(
    int device_id, cudaStream_t stream, const std::function<generativeqc_status()>& body) {
  device_id_ = device_id;
  return capture(stream, post_eigensolver_graph_, post_eigensolver_graph_exec_,
                 post_eigensolver_generation_, runtime::ResidencyGraphRole::hf_post_eigensolver,
                 0ULL, false, body);
}

/** Host submissions are source-owned observations, not proof that a graph or
 * its device-side tail launches completed. Keep the exact CUDA return status. */
cudaError_t RhfIterationGraphs::launch(cudaGraph_t graph, cudaGraphExec_t executable,
                                       std::uint64_t generation, runtime::ResidencyGraphRole role,
                                       cudaStream_t stream) const noexcept {
  const auto flags = role == runtime::ResidencyGraphRole::hf_iteration ? iteration_flags_ : 0ULL;
  runtime::observe_residency_graph(runtime::ResidencyGraphEvent::launch_begin, generation, role,
                                   flags, graph, executable, stream);
  const auto error = cudaGraphLaunch(executable, stream);
  runtime::observe_residency_graph(runtime::ResidencyGraphEvent::launch_end, generation, role,
                                   flags, graph, executable, stream, error);
  return error;
}

cudaError_t RhfIterationGraphs::launch_iteration(cudaStream_t stream) const noexcept {
  return launch(iteration_graph_, iteration_graph_exec_, iteration_generation_,
                runtime::ResidencyGraphRole::hf_iteration, stream);
}

cudaError_t RhfIterationGraphs::launch_post_eigensolver(cudaStream_t stream) const noexcept {
  return launch(post_eigensolver_graph_, post_eigensolver_graph_exec_, post_eigensolver_generation_,
                runtime::ResidencyGraphRole::hf_post_eigensolver, stream);
}

}  // namespace generativeqc::scf::cuda_execution
