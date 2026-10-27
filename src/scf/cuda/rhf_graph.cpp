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

RhfIterationGraphs::~RhfIterationGraphs() {
  if (provider_managed_) {
    if (!provider_closed_) std::terminate();
    return;
  }
  reset();
}

void RhfIterationGraphs::reset() noexcept {
  if (provider_managed_) {
    if (!provider_closed_) provider_failure_("graph/reset refused before explicit close", {});
    return;
  }
  std::lock_guard<std::mutex> allocation_lock(runtime::allocation_measurement_mutex);
  if (device_id_ >= 0) (void)cudaSetDevice(device_id_);
  destroy(post_eigensolver_graph_, post_eigensolver_graph_exec_, post_eigensolver_generation_,
          runtime::ResidencyGraphRole::hf_post_eigensolver);
  destroy(iteration_graph_, iteration_graph_exec_, iteration_generation_,
          runtime::ResidencyGraphRole::hf_iteration);
}

ProviderOwnerSchema RhfIterationGraphs::provider_schema() {
  using K = ProviderOwnedResourceKind;
  return {{{"post_graph_exec", K::status_resource},
           {"post_graph", K::status_resource},
           {"iteration_graph_exec", K::status_resource},
           {"iteration_graph", K::status_resource}}};
}

ProviderOwnerInventory RhfIterationGraphs::provider_inventory() const {
  using K = ProviderOwnedResourceKind;
  const auto token = [](auto pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
  return {{{"post_graph_exec", K::status_resource, token(post_eigensolver_graph_exec_)},
           {"post_graph", K::status_resource, token(post_eigensolver_graph_)},
           {"iteration_graph_exec", K::status_resource, token(iteration_graph_exec_)},
           {"iteration_graph", K::status_resource, token(iteration_graph_)}}};
}

void RhfIterationGraphs::enter_provider_retirement(
    std::function<void()> current,
    std::function<void(const std::string&, std::optional<std::uint32_t>)> failure) {
  for (const auto& resource : provider_inventory().resources)
    if (resource.token) throw std::logic_error("adopt graph owner before first creation");
  provider_current_ = std::move(current);
  provider_failure_ = std::move(failure);
  provider_managed_ = true;
}

ProviderInitializationTeardownStep RhfIterationGraphs::close_provider_operation(
    const std::string& operation) {
  const auto name = operation.substr(0, operation.find('/'));
  const bool post = name.starts_with("post_");
  auto& saved_graph = post ? retiring_post_graph_ : retiring_iteration_graph_;
  auto& saved_exec = post ? retiring_post_exec_ : retiring_iteration_exec_;
  auto& generation = post ? post_eigensolver_generation_ : iteration_generation_;
  if (generation && !saved_graph && !saved_exec) {
    saved_graph = post ? post_eigensolver_graph_ : iteration_graph_;
    saved_exec = post ? post_eigensolver_graph_exec_ : iteration_graph_exec_;
  }
  std::optional<cudaError_t> status;
  if (name == "post_graph_exec" && post_eigensolver_graph_exec_) {
    status = cudaGraphExecDestroy(post_eigensolver_graph_exec_);
    if (*status == cudaSuccess) post_eigensolver_graph_exec_ = nullptr;
  } else if (name == "post_graph" && post_eigensolver_graph_) {
    status = cudaGraphDestroy(post_eigensolver_graph_);
    if (*status == cudaSuccess) post_eigensolver_graph_ = nullptr;
  } else if (name == "iteration_graph_exec" && iteration_graph_exec_) {
    status = cudaGraphExecDestroy(iteration_graph_exec_);
    if (*status == cudaSuccess) iteration_graph_exec_ = nullptr;
  } else if (name == "iteration_graph" && iteration_graph_) {
    status = cudaGraphDestroy(iteration_graph_);
    if (*status == cudaSuccess) iteration_graph_ = nullptr;
  }
  if (!status) return {operation, ProviderTeardownStepState::unavailable, {}};
  if (*status != cudaSuccess) provider_failure_(operation, static_cast<std::uint32_t>(*status));
  const bool gone = post ? !post_eigensolver_graph_ && !post_eigensolver_graph_exec_
                         : !iteration_graph_ && !iteration_graph_exec_;
  if (gone && generation) {
    runtime::observe_residency_graph(runtime::ResidencyGraphEvent::destruction, generation,
                                     post ? runtime::ResidencyGraphRole::hf_post_eigensolver
                                          : runtime::ResidencyGraphRole::hf_iteration,
                                     post ? 0ULL : iteration_flags_, saved_graph, saved_exec,
                                     nullptr);
    generation = 0;
    saved_graph = nullptr;
    saved_exec = nullptr;
  }
  return {operation, ProviderTeardownStepState::returned, static_cast<std::uint32_t>(*status)};
}

void RhfIterationGraphs::mark_provider_closed() {
  for (const auto& resource : provider_inventory().resources)
    if (resource.token) throw std::logic_error("provider graphs remain live");
  provider_closed_ = true;
}

RhfGraphCaptureResult RhfIterationGraphs::capture(
    cudaStream_t stream, cudaGraph_t& graph, cudaGraphExec_t& executable, std::uint64_t& generation,
    runtime::ResidencyGraphRole role, unsigned long long instantiate_flags, bool synchronize_before,
    const std::function<generativeqc_status()>& body) {
  std::unique_lock<std::mutex> provider_lock(runtime::allocation_measurement_mutex,
                                             std::defer_lock);
  if (provider_managed_) {
    provider_current_();
    provider_lock.lock();
    const bool post = role == runtime::ResidencyGraphRole::hf_post_eigensolver;
    for (const auto& slot : {post ? "post_graph_exec" : "iteration_graph_exec",
                             post ? "post_graph" : "iteration_graph"}) {
      const auto state = provider_inventory();
      const auto resource = std::find_if(state.resources.begin(), state.resources.end(),
                                         [&](const auto& item) { return item.slot == slot; });
      if (resource->token) {
        const auto step = close_provider_operation(std::string(slot) + "/release");
        if (!step.status || *step.status != 0)
          return {GENERATIVEQC_STATUS_SUCCESS,
                  step.status ? static_cast<cudaError_t>(*step.status) : cudaErrorUnknown};
      }
    }
  }
  const runtime::ResidencyExecution source_execution(runtime::ResidencyOwner::hf_graph_setup);
  if (!provider_managed_) destroy(graph, executable, generation, role);

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
    if (provider_managed_) {
      const auto end = cudaStreamEndCapture(stream, &graph);
      if (end != cudaSuccess)
        provider_failure_("graph/end_capture", static_cast<std::uint32_t>(end));
      throw;
    }
    // Preserve the original exception while releasing the stream's capture
    // state. Teardown and a later retry must not inherit an abandoned capture.
    cudaGraph_t abandoned_graph = nullptr;
    (void)cudaStreamEndCapture(stream, &abandoned_graph);
    if (abandoned_graph != nullptr) (void)cudaGraphDestroy(abandoned_graph);
    throw;
  }
  if (body_status != GENERATIVEQC_STATUS_SUCCESS) {
    if (provider_managed_) {
      const auto end = cudaStreamEndCapture(stream, &graph);
      if (end != cudaSuccess)
        provider_failure_("graph/end_capture", static_cast<std::uint32_t>(end));
      return {body_status, end};
    }
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
  const auto error = launch(iteration_graph_, iteration_graph_exec_, iteration_generation_,
                            runtime::ResidencyGraphRole::hf_iteration, stream);
  if (error == cudaSuccess) ++iteration_launches_;
  return error;
}

cudaError_t RhfIterationGraphs::launch_post_eigensolver(cudaStream_t stream) const noexcept {
  const auto error =
      launch(post_eigensolver_graph_, post_eigensolver_graph_exec_, post_eigensolver_generation_,
             runtime::ResidencyGraphRole::hf_post_eigensolver, stream);
  if (error == cudaSuccess) ++post_eigensolver_launches_;
  return error;
}

}  // namespace generativeqc::scf::cuda_execution
