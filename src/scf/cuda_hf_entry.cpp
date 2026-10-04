#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "generativeqc/generativeqc.hpp"
#include "integrals/electron_interaction_source.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::scf {
namespace {

using RhfPlanOwner =
    std::unique_ptr<CudaRhfBucketPlan, void (*)(CudaRhfBucketPlan*)>;

bool exact_reference_source_identity(const CudaRhfBucketPlan* plan, const core::System& system,
                                     const ScfOptions& requested_options, int device_id) noexcept {
  try {
    if (plan == nullptr || !plan->initialized || plan->batch_size != 1 || plan->unrestricted ||
      plan->resources.device_id_ != device_id || plan->resources.stream_ == nullptr ||
      plan->resources.reference_eri_ == nullptr) {
    return false;
  }
  if (!plan->options.export_physical_reference || plan->options.screening_tolerance != 0.0 ||
      plan->options.precision_mode.value_or(GENERATIVEQC_PRECISION_FP64) !=
          GENERATIVEQC_PRECISION_FP64) {
    return false;
  }

  auto expected_options = requested_options;
  if (expected_options.resolved_fock_build.has_value() &&
      expected_options.resolved_fock_build != plan->options.resolved_fock_build) {
    return false;
  }
  expected_options.resolved_fock_build = plan->options.resolved_fock_build;
  if (!same_hf_bucket_options(plan->options, expected_options)) return false;

  const auto n = molecule::ao_count(system);
  if (n != plan->nbf) return false;
  const auto n2 = posthf::checked_mul(n, n);
  const auto n4 = posthf::checked_mul(n2, n2);
  if (plan->resources.reference_eri_bytes_ != posthf::checked_mul(n4, sizeof(double)))
    return false;

  cuda_execution::HostBatch candidate;
  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> seeds{nullptr};
  if (!cuda_execution::pack_host_batch(systems, seeds, candidate, false, true, false))
    return false;
    return cuda_execution::same_topology(plan->topology, candidate) &&
           plan->cached_positions == candidate.positions;
  } catch (...) {
    return false;
  }
}

class CudaRhfReferenceInteractionSource final : public integrals::ElectronInteractionSource {
 public:
  CudaRhfReferenceInteractionSource(core::System system, RhfPlanOwner plan,
                                    std::size_t retained_bytes)
      : system_(std::move(system)),
        plan_(std::move(plan)),
        retained_bytes_(retained_bytes),
        nbf_(molecule::ao_count(system_)),
        device_(plan_ ? plan_->resources.device_id_ : -1) {}

  ~CudaRhfReferenceInteractionSource() override {
    if (device_ >= 0) (void)cudaSetDevice(device_);
    for (const auto& use : uses_) {
      if (use.event != nullptr) {
        (void)cudaEventSynchronize(use.event);
        (void)cudaEventDestroy(use.event);
      }
    }
  }

  const core::System& orbital() const override { return system_; }
  std::size_t nbf() const override { return nbf_; }
  std::size_t naux() const override { return 0; }
  std::size_t retained_numeric_bytes() const override { return retained_bytes_; }
  bool supports(Operator op) const noexcept override { return op == Operator::eri; }
  bool supports_host_read(Operator) const noexcept override { return false; }
  bool supports_device_read(Operator op, int device) const noexcept override {
    return op == Operator::eri && plan_ != nullptr && device == device_ &&
           plan_->resources.reference_eri_ != nullptr;
  }

  void read(Operator, const std::array<std::size_t, 4>&,
            const std::array<std::size_t, 4>&, double*, std::size_t) const override {
    throw std::invalid_argument("CUDA RHF reference interaction source is device-only");
  }

  void read_device(Operator op, const std::array<std::size_t, 4>& begin,
                   const std::array<std::size_t, 4>& count,
                   integrals::DeviceInteractionTarget target,
                   std::size_t elements) const override {
    if (!supports_device_read(op, target.device) || target.values == nullptr ||
        target.stream == nullptr || target.capacity < elements) {
      throw std::invalid_argument("invalid CUDA RHF reference interaction target");
    }
    std::size_t expected = 1;
    for (unsigned axis = 0; axis < 4; ++axis) {
      if (count[axis] == 0 || begin[axis] > nbf_ || count[axis] > nbf_ - begin[axis])
        throw std::invalid_argument("CUDA RHF reference interaction tile is out of range");
      expected = posthf::checked_mul(expected, count[axis]);
    }
    if (expected != elements)
      throw std::invalid_argument("CUDA RHF reference interaction tile size mismatch");
    if (cudaSetDevice(device_) != cudaSuccess)
      throw std::runtime_error("CUDA RHF reference source could not select its device");
    auto stream = static_cast<cudaStream_t>(target.stream);
    cuda_execution::launch_copy_resident_eri_tile(
        stream, plan_->resources.reference_eri_, nbf_, begin, count, elements, target.values);
    if (cudaPeekAtLastError() != cudaSuccess)
      throw std::runtime_error("CUDA RHF reference interaction copy launch failed");
    record_use(stream);
  }

 private:
  struct StreamUse {
    cudaStream_t stream{};
    cudaEvent_t event{};
  };

  void record_use(cudaStream_t stream) const {
    for (auto& use : uses_) {
      if (use.stream != stream) continue;
      if (cudaEventRecord(use.event, stream) != cudaSuccess)
        throw std::runtime_error("CUDA RHF reference interaction event record failed");
      return;
    }
    cudaEvent_t event{};
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess)
      throw std::runtime_error("CUDA RHF reference interaction event allocation failed");
    try {
      uses_.push_back({stream, event});
    } catch (...) {
      (void)cudaEventDestroy(event);
      throw;
    }
    if (cudaEventRecord(event, stream) != cudaSuccess)
      throw std::runtime_error("CUDA RHF reference interaction event record failed");
  }

  core::System system_;
  RhfPlanOwner plan_{nullptr, destroy_rhf_cuda_bucket_plan};
  std::size_t retained_bytes_{};
  std::size_t nbf_{};
  int device_{-1};
  mutable std::vector<StreamUse> uses_;
};

}  // namespace

// Host-only single-system adapters share the bucket execution and error
// contract. Keep them outside the kernel translation unit so host changes
// do not require adding more code to the large CUDA implementation.
ScfResult run_rhf_cuda(
    const core::System& system, const ScfOptions& options, int device_id,
    const std::vector<double>* initial_density,
    std::shared_ptr<const integrals::ElectronInteractionSource>* interaction_source) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");
  if (interaction_source) interaction_source->reset();

  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  CudaRhfBucketPlan* raw_plan = nullptr;
  std::vector<RhfBucketItem> result =
      interaction_source
          ? run_rhf_cuda_bucket_cached(&raw_plan, systems, options, initial_densities, device_id)
          : run_rhf_cuda_bucket(systems, options, initial_densities, device_id);
  RhfPlanOwner plan(raw_plan, destroy_rhf_cuda_bucket_plan);
  if (result.empty()) throw std::runtime_error("CUDA RHF returned no result");
  const generativeqc_status status = result.front().status;
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) {
    throw std::invalid_argument("CUDA RHF received invalid arguments");
  }
  if (status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
    throw Error(status, "CUDA RHF numerical endpoint is unavailable in this mode");
  }
  if (status != GENERATIVEQC_STATUS_SUCCESS && status != GENERATIVEQC_STATUS_SCF_NOT_CONVERGED) {
    throw std::runtime_error("CUDA RHF execution failed");
  }

  ScfResult scf = std::move(result.front().scf);
  if (interaction_source && scf.converged && scf.reference &&
      exact_reference_source_identity(plan.get(), system, options, device_id)) {
    try {
      auto retained = hf_cuda_owned_device_bytes(plan.get());
      retained = posthf::checked_add(retained, plan->resources.solver_host_workspace_bytes_);
      retained = posthf::checked_add(retained, posthf::source_capacity(system));
      core::System source_system = system;
      *interaction_source = std::make_shared<CudaRhfReferenceInteractionSource>(
          std::move(source_system), std::move(plan), retained);
    } catch (const std::bad_alloc&) {
      interaction_source->reset();
    }
  }
  return scf;
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
