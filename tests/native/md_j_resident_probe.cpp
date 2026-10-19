#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "api/handles.hpp"
#include "scf/cuda_fock_execution.hpp"

namespace {
thread_local std::string last_error;

void require_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/** Graphs borrow the prepared owner and the Python-owned input/output storage.
 * The owner fences before replacing or destroying this graph lease. */
struct CapturedGraph {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  std::size_t nodes{}, kernels{}, device_copies{};

  ~CapturedGraph() {
    if (executable) (void)cudaGraphExecDestroy(executable);
    if (graph) (void)cudaGraphDestroy(graph);
  }
};

/** Always leave capture mode, including failed native enqueue/instantiation. */
struct CaptureScope {
  cudaStream_t stream;
  bool active{};

  explicit CaptureScope(cudaStream_t value) : stream(value) {
    require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    active = true;
  }
  void finish(cudaGraph_t& graph) {
    const auto status = cudaStreamEndCapture(stream, &graph);
    active = false;
    require_cuda(status);
  }
  ~CaptureScope() {
    if (active) {
      cudaGraph_t abandoned{};
      (void)cudaStreamEndCapture(stream, &abandoned);
      if (abandoned) (void)cudaGraphDestroy(abandoned);
    }
  }
};

/** Test-only borrower of the same resident value seam used by native KS.
 * Public host Fock evaluation intentionally uses a different compatibility
 * route and therefore cannot qualify the resident MD contraction kernels.
 */
struct ResidentProbe {
  generativeqc::scf::PreparedFockPlan source;
  generativeqc::scf::PreparedCudaFockBinding binding;
  cudaEvent_t started{}, finished{};
  std::unique_ptr<CapturedGraph> captured;
  bool diagnostic_downloads{};

  ResidentProbe(const generativeqc::core::System& system, bool unrestricted, bool exchange,
                double screening, std::size_t budget)
      : source(
            system, nullptr,
            generativeqc::scf::resolve_fock_build(specification(unrestricted, exchange),
                                                  generativeqc::scf::FockBackend::Cuda, screening),
            0, budget),
        binding(generativeqc::scf::prepared_cuda_fock_binding(source)) {
    if (!binding) throw std::runtime_error("resident Fock binding is unavailable");
    const char* counts = std::getenv("GENERATIVEQC_MD_J_WORK_COUNTS");
    diagnostic_downloads = counts && std::strcmp(counts, "1") == 0;
  }

  static generativeqc::scf::FockBuildSpec specification(bool unrestricted, bool exchange) {
    auto spec = generativeqc::scf::make_hf_fock_spec(unrestricted
                                                         ? generativeqc::scf::FockSpin::Unrestricted
                                                         : generativeqc::scf::FockSpin::Restricted);
    spec.derivative_order = 0;
    spec.exchange.present = exchange;
    return spec;
  }

  ~ResidentProbe() {
    if (binding.stream) (void)cudaStreamSynchronize(binding.stream);
    captured.reset();
    if (started) (void)cudaEventDestroy(started);
    if (finished) (void)cudaEventDestroy(finished);
  }
};

ResidentProbe& checked_probe(void* handle) {
  if (!handle) throw std::invalid_argument("null resident probe");
  return *static_cast<ResidentProbe*>(handle);
}

void finish_timing(ResidentProbe& probe, double* seconds) {
  require_cuda(cudaEventRecord(probe.finished, probe.binding.stream));
  require_cuda(cudaStreamSynchronize(probe.binding.stream));
  float milliseconds = 0.0F;
  require_cuda(cudaEventElapsedTime(&milliseconds, probe.started, probe.finished));
  *seconds = milliseconds / 1000.0;
}
}  // namespace

/** Construct from a caller-owned native system, retained until probe teardown. */
extern "C" void* md_j_probe_create(const generativeqc_system* system, int unrestricted,
                                   int exchange, double screening, std::size_t budget) {
  try {
    if (!system) throw std::invalid_argument("null resident probe system");
    auto probe =
        std::make_unique<ResidentProbe>(system->data, unrestricted, exchange, screening, budget);
    require_cuda(cudaEventCreate(&probe->started));
    require_cuda(cudaEventCreate(&probe->finished));
    last_error.clear();
    return probe.release();
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

/** Enqueue raw resident J/K on caller-owned device arrays and fence only this
 * owner's stream. Event time excludes Python copies and native preparation.
 */
extern "C" int md_j_probe_execute(void* handle, const double* density, const double* beta,
                                  std::size_t elements, double* coulomb, double* alpha_exchange,
                                  double* beta_exchange, int* numerical_error, double* seconds) {
  try {
    auto& probe = checked_probe(handle);
    if (!seconds) throw std::invalid_argument("null resident probe timing output");
    require_cuda(cudaEventRecord(probe.started, probe.binding.stream));
    std::string detail;
    const auto status = generativeqc::scf::enqueue_prepared_cuda_fock(
        probe.source, density, beta, elements, coulomb, alpha_exchange, beta_exchange,
        numerical_error, false, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    finish_timing(probe, seconds);
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return 1;
  }
}

/** Capture the actual native resident value enqueue on its prepared stream.
 * No wrapper timing events, upload, download or host callbacks enter the graph.
 * Work-count diagnostics deliberately fence/download and are not capturable.
 * A failed recapture preserves the previous executable and ends capture before
 * cleanup, so ordinary execution and a later recapture remain available.
 */
extern "C" int md_j_probe_capture(void* handle, const double* density, const double* beta,
                                  std::size_t elements, double* coulomb, double* alpha_exchange,
                                  double* beta_exchange, int* numerical_error) {
  bool used_cuda = false;
  try {
    auto& probe = checked_probe(handle);
    if (probe.diagnostic_downloads)
      throw std::invalid_argument(
          "graph capture requires GENERATIVEQC_MD_J_WORK_COUNTS disabled "
          "before resident probe creation");
    used_cuda = true;
    require_cuda(cudaStreamSynchronize(probe.binding.stream));
    auto candidate = std::make_unique<CapturedGraph>();
    {
      CaptureScope capture(probe.binding.stream);
      std::string detail;
      const auto status = generativeqc::scf::enqueue_prepared_cuda_fock(
          probe.source, density, beta, elements, coulomb, alpha_exchange, beta_exchange,
          numerical_error, false, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
      capture.finish(candidate->graph);
    }
    require_cuda(cudaGraphGetNodes(candidate->graph, nullptr, &candidate->nodes));
    std::vector<cudaGraphNode_t> nodes(candidate->nodes);
    require_cuda(cudaGraphGetNodes(candidate->graph, nodes.data(), &candidate->nodes));
    for (const auto node : nodes) {
      cudaGraphNodeType type{};
      require_cuda(cudaGraphNodeGetType(node, &type));
      if (type == cudaGraphNodeTypeKernel)
        ++candidate->kernels;
      else if (type == cudaGraphNodeTypeMemcpy) {
        // Canonical spin staging and AO output reconstruction legitimately
        // copy device-to-device. No captured host transfer or implicit pointer
        // direction may masquerade as a device-only resident graph.
        cudaMemcpy3DParms copy{};
        require_cuda(cudaGraphMemcpyNodeGetParams(node, &copy));
        if (copy.kind != cudaMemcpyDeviceToDevice || copy.srcArray || copy.dstArray)
          throw std::runtime_error("resident graph contains a non-device memory transfer");
        for (const auto pointer : {copy.srcPtr.ptr, copy.dstPtr.ptr}) {
          cudaPointerAttributes attributes{};
          require_cuda(cudaPointerGetAttributes(&attributes, pointer));
          if (attributes.type != cudaMemoryTypeDevice ||
              attributes.device != probe.binding.device_id)
            throw std::runtime_error("resident graph copy does not use prepared-device buffers");
        }
        ++candidate->device_copies;
      } else if (type != cudaGraphNodeTypeMemset)
        throw std::runtime_error("resident graph contains a non-device operation");
    }
    if (!candidate->kernels) throw std::runtime_error("resident graph contains no kernels");
    require_cuda(cudaGraphInstantiateWithFlags(&candidate->executable, candidate->graph, 0));
    probe.captured = std::move(candidate);
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    // Direct's failure drain can invalidate an active capture. Scope teardown
    // above always ends it; consume that handled CUDA error after retaining the
    // original failure so a later native launch cannot inherit stale state.
    if (used_cuda) (void)cudaGetLastError();
    return 1;
  }
}

/** Only launch the retained graph; never re-enter native Fock host dispatch.
 * The final owner-stream fence is outside capture and publishes test results.
 */
extern "C" int md_j_probe_replay(void* handle, double* seconds) {
  try {
    auto& probe = checked_probe(handle);
    if (!seconds) throw std::invalid_argument("null resident probe timing output");
    if (!probe.captured) throw std::invalid_argument("resident graph has not been captured");
    require_cuda(cudaEventRecord(probe.started, probe.binding.stream));
    require_cuda(cudaGraphLaunch(probe.captured->executable, probe.binding.stream));
    finish_timing(probe, seconds);
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return 1;
  }
}

extern "C" std::size_t md_j_probe_graph_nodes(void* handle) {
  const auto* probe = static_cast<ResidentProbe*>(handle);
  return probe && probe->captured ? probe->captured->nodes : 0;
}

extern "C" std::size_t md_j_probe_graph_kernels(void* handle) {
  const auto* probe = static_cast<ResidentProbe*>(handle);
  return probe && probe->captured ? probe->captured->kernels : 0;
}

extern "C" std::size_t md_j_probe_graph_device_copies(void* handle) {
  const auto* probe = static_cast<ResidentProbe*>(handle);
  return probe && probe->captured ? probe->captured->device_copies : 0;
}

extern "C" const char* md_j_probe_schedule(void* handle) {
  return static_cast<ResidentProbe*>(handle)->source.diagnostic().direct.schedule;
}

extern "C" std::size_t md_j_probe_device_bytes(void* handle) {
  return static_cast<ResidentProbe*>(handle)->source.diagnostic().device_bytes;
}

extern "C" const char* md_j_probe_error() { return last_error.c_str(); }
extern "C" void md_j_probe_destroy(void* handle) { delete static_cast<ResidentProbe*>(handle); }
