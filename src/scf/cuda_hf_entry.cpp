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
#include "runtime/resource_usage.hpp"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::scf {
namespace {

using RhfPlanOwner = std::unique_ptr<CudaRhfBucketPlan, void (*)(CudaRhfBucketPlan*)>;

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
    // If the runtime cannot establish completion at all, keep the backing
    // allocation alive rather than releasing storage still borrowed by a GPU.
    if (!finish_uses()) (void)plan_.release();
    for (const auto& use : uses_)
      if (use.event != nullptr) (void)cudaEventDestroy(use.event);
  }

  CudaRhfBucketPlan* release_plan() const noexcept {
    return finish_uses() ? plan_.release() : nullptr;
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

  void read(Operator, const std::array<std::size_t, 4>&, const std::array<std::size_t, 4>&, double*,
            std::size_t) const override {
    throw std::invalid_argument("CUDA RHF reference interaction source is device-only");
  }

  void read_device(Operator op, const std::array<std::size_t, 4>& begin,
                   const std::array<std::size_t, 4>& count,
                   integrals::DeviceInteractionTarget target, std::size_t elements) const override {
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
    // All allocating bookkeeping precedes submission. Mark the use unfenced
    // before launch so even a launch diagnostic or failed event re-record
    // cannot leave a newer borrow hidden behind an older event generation.
    auto& use = prepare_use(stream);
    use.pending = true;
    use.recorded = false;
    try {
      cuda_execution::launch_copy_resident_eri_tile(stream, plan_->resources.reference_eri_, nbf_,
                                                    begin, count, elements, target.values);
      if (cudaPeekAtLastError() != cudaSuccess)
        throw std::runtime_error("CUDA RHF reference interaction copy launch failed");
      if (cudaEventRecord(use.event, stream) != cudaSuccess)
        throw std::runtime_error("CUDA RHF reference interaction event record failed");
      use.recorded = true;
    } catch (...) {
      (void)finish_use(use);
      throw;
    }
  }

 private:
  struct StreamUse {
    cudaStream_t stream{};
    cudaEvent_t event{};
    bool pending{};
    bool recorded{};
  };

  static bool finish_use(StreamUse& use) noexcept {
    if (!use.pending) return true;
    const bool complete = (use.recorded && cudaEventSynchronize(use.event) == cudaSuccess) ||
                          cudaStreamSynchronize(use.stream) == cudaSuccess ||
                          cudaDeviceSynchronize() == cudaSuccess;
    if (complete) use.pending = false;
    return complete;
  }

  bool finish_uses() const noexcept {
    if (device_ < 0 || cudaSetDevice(device_) != cudaSuccess) return uses_.empty();
    bool complete = true;
    for (auto& use : uses_) complete = finish_use(use) && complete;
    return complete;
  }

  StreamUse& prepare_use(cudaStream_t stream) const {
    for (auto& use : uses_)
      if (use.stream == stream) return use;
    cudaEvent_t event{};
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess)
      throw std::runtime_error("CUDA RHF reference interaction event allocation failed");
    try {
      uses_.push_back({stream, event});
    } catch (...) {
      (void)cudaEventDestroy(event);
      throw;
    }
    return uses_.back();
  }

  core::System system_;
  mutable RhfPlanOwner plan_{nullptr, destroy_rhf_cuda_bucket_plan};
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

  if (interaction_source) {
    struct LocalPlan {
      CudaRhfBucketPlan* plan{};
      ~LocalPlan() { destroy_rhf_cuda_bucket_plan(plan); }
    } owner;
    return run_rhf_cuda_cached(&owner.plan, system, options, device_id, initial_density, nullptr,
                               interaction_source);
  }
  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  auto result = run_rhf_cuda_bucket(systems, options, initial_densities, device_id);
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

  return std::move(result.front().scf);
}

ScfResult run_rhf_cuda_cached(
    CudaRhfBucketPlan** plan, const core::System& system, const ScfOptions& options, int device_id,
    const std::vector<double>* initial_density, bool* execution_plan_reused,
    std::shared_ptr<const integrals::ElectronInteractionSource>* interaction_source) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");
  if (!plan) throw std::invalid_argument("CUDA RHF cached execution requires a plan owner");
  if (execution_plan_reused) *execution_plan_reused = false;
  if (interaction_source) interaction_source->reset();

  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  std::vector<RhfBucketItem> result;
  try {
    result = run_rhf_cuda_bucket_cached(plan, systems, options, initial_densities, device_id);
  } catch (...) {
    destroy_rhf_cuda_bucket_plan(*plan);
    *plan = nullptr;
    throw;
  }
  if (result.empty()) {
    destroy_rhf_cuda_bucket_plan(*plan);
    *plan = nullptr;
    throw std::runtime_error("CUDA RHF returned no result");
  }
  if (execution_plan_reused) *execution_plan_reused = result.front().execution_plan_reused;
  const generativeqc_status status = result.front().status;
  if (status != GENERATIVEQC_STATUS_SUCCESS) {
    // A failed or nonconverged attempt must not publish a partially advanced
    // executable owner. The method layer may retry cold with a fresh plan.
    destroy_rhf_cuda_bucket_plan(*plan);
    *plan = nullptr;
  }
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT)
    throw std::invalid_argument("CUDA RHF received invalid arguments");
  if (status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED)
    throw Error(status, "CUDA RHF numerical endpoint is unavailable in this mode");
  if (status != GENERATIVEQC_STATUS_SUCCESS && status != GENERATIVEQC_STATUS_SCF_NOT_CONVERGED)
    throw std::runtime_error("CUDA RHF execution failed");
  ScfResult scf = std::move(result.front().scf);
  if (interaction_source && scf.converged && scf.reference &&
      exact_reference_source_identity(*plan, system, options, device_id)) {
    try {
      core::System source_system = system;
      auto retained = posthf::checked_add(hf_cuda_retained_numeric_bytes(*plan),
                                          posthf::source_capacity(source_system));
      // The generic source allowance covers atoms/shells/primitive data, but
      // this additional normalized system also owns its ECP scalar payload.
      retained = posthf::checked_add(retained, runtime::vector_bytes(source_system.ecp_terms));
      // The cached slot and the source are mutually exclusive owners. Moving
      // out before publication also leaves allocation failure exception-safe.
      RhfPlanOwner owner(std::exchange(*plan, nullptr), destroy_rhf_cuda_bucket_plan);
      *interaction_source = std::make_shared<CudaRhfReferenceInteractionSource>(
          std::move(source_system), std::move(owner), retained);
    } catch (const std::bad_alloc&) {
      interaction_source->reset();
    } catch (const std::length_error&) {
      interaction_source->reset();
    } catch (const std::overflow_error&) {
      interaction_source->reset();
    }
  }
  return scf;
}

bool reclaim_rhf_cuda_reference_plan(
    CudaRhfBucketPlan** plan,
    std::shared_ptr<const integrals::ElectronInteractionSource>& interaction_source) noexcept {
  if (!plan || *plan || !interaction_source || interaction_source.use_count() != 1) return false;
  const auto* source =
      dynamic_cast<const CudaRhfReferenceInteractionSource*>(interaction_source.get());
  if (!source) return false;
  auto* released = source->release_plan();
  if (!released) return false;
  interaction_source.reset();
  *plan = released;
  return true;
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
