#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "api/handles.hpp"
#include "scf/cuda_fock_execution.hpp"

namespace {
thread_local std::string last_error;

void require_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/** Test-only borrower of the same resident value seam used by native KS.
 * Public host Fock evaluation intentionally uses a different compatibility
 * route and therefore cannot qualify the resident MD contraction kernels.
 */
struct ResidentProbe {
  generativeqc::scf::PreparedFockPlan source;
  generativeqc::scf::PreparedCudaFockBinding binding;
  cudaEvent_t started{}, finished{};

  ResidentProbe(const generativeqc::core::System& system, bool unrestricted, bool exchange,
                double screening, std::size_t budget)
      : source(
            system, nullptr,
            generativeqc::scf::resolve_fock_build(specification(unrestricted, exchange),
                                                  generativeqc::scf::FockBackend::Cuda, screening),
            0, budget),
        binding(generativeqc::scf::prepared_cuda_fock_binding(source)) {
    if (!binding) throw std::runtime_error("resident Fock binding is unavailable");
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
    if (started) (void)cudaEventDestroy(started);
    if (finished) (void)cudaEventDestroy(finished);
  }
};
}  // namespace

/** Construct from a caller-owned native system, retained until probe teardown. */
extern "C" void* md_j_probe_create(const generativeqc_system* system, int unrestricted,
                                   int exchange, double screening, std::size_t budget) {
  try {
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
    auto& probe = *static_cast<ResidentProbe*>(handle);
    require_cuda(cudaEventRecord(probe.started, probe.binding.stream));
    std::string detail;
    const auto status = generativeqc::scf::enqueue_prepared_cuda_fock(
        probe.source, density, beta, elements, coulomb, alpha_exchange, beta_exchange,
        numerical_error, false, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    require_cuda(cudaEventRecord(probe.finished, probe.binding.stream));
    require_cuda(cudaStreamSynchronize(probe.binding.stream));
    float milliseconds = 0.0F;
    require_cuda(cudaEventElapsedTime(&milliseconds, probe.started, probe.finished));
    *seconds = milliseconds / 1000.0;
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return 1;
  }
}

extern "C" const char* md_j_probe_schedule(void* handle) {
  return static_cast<ResidentProbe*>(handle)->source.diagnostic().direct.schedule;
}

extern "C" std::size_t md_j_probe_device_bytes(void* handle) {
  return static_cast<ResidentProbe*>(handle)->source.diagnostic().device_bytes;
}

extern "C" const char* md_j_probe_error() { return last_error.c_str(); }
extern "C" void md_j_probe_destroy(void* handle) { delete static_cast<ResidentProbe*>(handle); }
