#include <stdexcept>
#include <utility>
#include <vector>

#include "generativeqc/generativeqc.hpp"
#include "scf/cuda/rhf_source_handoff.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::scf {

// Host-only single-system adapters share the bucket execution and error
// contract. Keep them outside the kernel translation unit so host changes
// do not require adding more code to the large CUDA implementation.
static ScfResult run_rhf_cuda_impl(const core::System& system, const ScfOptions& options,
                                   int device_id, const std::vector<double>* initial_density,
                                   CudaRhfSourceHandoff* handoff) {
  if (handoff) *handoff = {};
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");

  std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  CudaRhfBucketPlan* raw_plan = nullptr;
  // The guard also owns a plan published before a throwing bucket execution.
  struct PlanGuard {
    CudaRhfBucketPlan*& plan;
    ~PlanGuard() { destroy_rhf_cuda_bucket_plan(plan); }
  } guard{raw_plan};
  auto result = handoff ? run_rhf_cuda_bucket_cached(&raw_plan, systems, options, initial_densities,
                                                     device_id)
                        : run_rhf_cuda_bucket(systems, options, initial_densities, device_id);
  if (result.empty()) throw std::runtime_error("CUDA RHF returned no result");
  const generativeqc_status status = result.front().status;
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) {
    throw std::invalid_argument("CUDA RHF received invalid arguments");
  }
  // Preserve the structural-only diagnostic status through the single-system
  // adapter; translating it to runtime_error would turn it into INTERNAL_ERROR.
  if (status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
    throw Error(status, "CUDA RHF numerical endpoint is unavailable in this mode");
  }
  if (status != GENERATIVEQC_STATUS_SUCCESS && status != GENERATIVEQC_STATUS_SCF_NOT_CONVERGED) {
    throw std::runtime_error("CUDA RHF execution failed");
  }
  auto& reference = result.front().scf;
  if (handoff && reference.converged && reference.reference)
    *handoff = detach_rhf_cuda_source(*raw_plan, std::move(systems.front()),
                                      reference.reference->numeric_capacity_bytes,
                                      options.reference_memory_budget_bytes);
  return std::move(reference);
}

ScfResult run_rhf_cuda(const core::System& system, const ScfOptions& options, int device_id,
                       const std::vector<double>* initial_density) {
  return run_rhf_cuda_impl(system, options, device_id, initial_density, nullptr);
}

ScfResult run_rhf_cuda_with_source(const core::System& system, const ScfOptions& options,
                                   int device_id, const std::vector<double>* initial_density,
                                   CudaRhfSourceHandoff& handoff) {
  return run_rhf_cuda_impl(system, options, device_id, initial_density, &handoff);
}

ScfResult run_uhf_cuda(const core::System& system, const ScfOptions& options, int device_id,
                       const std::vector<double>* initial_density) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");

  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  std::vector<RhfBucketItem> result =
      run_uhf_cuda_bucket(systems, options, initial_densities, device_id);
  if (result.empty()) throw std::runtime_error("CUDA UHF returned no result");
  const generativeqc_status status = result.front().status;
  if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) {
    throw std::invalid_argument("CUDA UHF received invalid arguments");
  }
  if (status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
    throw Error(status, "CUDA UHF numerical endpoint is unavailable in this mode");
  }
  if (status != GENERATIVEQC_STATUS_SUCCESS && status != GENERATIVEQC_STATUS_SCF_NOT_CONVERGED) {
    throw std::runtime_error("CUDA UHF execution failed");
  }
  return std::move(result.front().scf);
}

}  // namespace generativeqc::scf
