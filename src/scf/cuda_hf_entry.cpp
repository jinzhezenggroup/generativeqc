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
#include "scf/cuda/rhf_source_handoff.hpp"
#include "scf/cuda/rhf_source_view.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::scf {
namespace {

using RhfPlanOwner = std::unique_ptr<CudaRhfBucketPlan, void (*)(CudaRhfBucketPlan*)>;

bool exact_reference_source_identity(const CudaRhfBucketPlan* plan, const core::System& system,
                                     const ScfOptions& requested_options, int device_id,
                                     bool borrowed_inputs = false) noexcept {
  try {
    if (plan == nullptr || !plan->initialized || plan->batch_size != 1 || plan->unrestricted ||
        plan->resources.device_id_ != device_id || plan->resources.stream_ == nullptr ||
        (!borrowed_inputs && plan->resources.reference_eri_ == nullptr)) {
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
    if (!borrowed_inputs) {
      const auto n2 = posthf::checked_mul(n, n);
      const auto n4 = posthf::checked_mul(n2, n2);
      if (plan->resources.reference_eri_bytes_ != posthf::checked_mul(n4, sizeof(double)))
        return false;
    }

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

/** Opt-in source owns host metadata/access only. CUDA storage, graphs and all
 * consumer events remain in the same pending native reference owner. */
class PendingRhfReferenceInteractionSource final : public integrals::ElectronInteractionSource {
 public:
  PendingRhfReferenceInteractionSource(core::System&& system,
                                       std::shared_ptr<PendingReferenceProviderOwner> owner,
                                       std::size_t retained)
      : system_(std::move(system)),
        owner_(std::move(owner)),
        lease_(owner_->acquire_source_lease()),
        view_(borrow_rhf_source_batch(*owner_->bucket())),
        resident_(owner_->bucket()->resources.reference_eri_),
        retained_(retained),
        nbf_(owner_->bucket()->nbf),
        device_(owner_->bucket()->resources.device_id_) {}
  ~PendingRhfReferenceInteractionSource() override { owner_->release_source_lease(lease_); }
  CudaRhfBucketPlan* release_plan() const noexcept {
    try {
      auto guard = owner_->acquire_borrow_lock();
      if (!lease_->load(std::memory_order_acquire) || !owner_->finish_uses_current())
        return nullptr;
      auto* plan = owner_->bucket();
      owner_->release_source_lease(lease_);
      return plan;
    } catch (...) {
      return nullptr;
    }
  }
  const core::System& orbital() const override { return system_; }
  std::size_t nbf() const override { return nbf_; }
  std::size_t naux() const override { return 0; }
  std::size_t retained_numeric_bytes() const override { return retained_; }
  bool supports(Operator op) const noexcept override { return op == Operator::eri; }
  bool supports_host_read(Operator) const noexcept override { return false; }
  bool supports_device_read(Operator op, int device) const noexcept override {
    return op == Operator::eri && device == device_ && lease_->load(std::memory_order_acquire);
  }
  void read(Operator, const std::array<std::size_t, 4>&, const std::array<std::size_t, 4>&, double*,
            std::size_t) const override {
    throw std::invalid_argument("pending native RHF source is device-only");
  }
  void read_device(Operator op, const std::array<std::size_t, 4>& begin,
                   const std::array<std::size_t, 4>& count,
                   integrals::DeviceInteractionTarget target, std::size_t elements) const override {
    auto guard = owner_->acquire_borrow_lock();
    owner_->require_current();
    if (!supports_device_read(op, target.device) || !target.values || !target.stream ||
        target.capacity < elements)
      throw std::invalid_argument("invalid pending RHF interaction target");
    std::size_t expected = 1;
    for (unsigned axis = 0; axis < 4; ++axis) {
      if (!count[axis] || begin[axis] > nbf_ || count[axis] > nbf_ - begin[axis])
        throw std::invalid_argument("pending RHF interaction tile exceeds its public basis");
      expected = posthf::checked_mul(expected, count[axis]);
    }
    if (expected != elements)
      throw std::invalid_argument("pending RHF interaction tile extent mismatch");
    auto stream = static_cast<cudaStream_t>(target.stream);
    auto& use = owner_->prepare_use(target.stream);
    use.pending = true;
    use.recorded = false;
    try {
      if (resident_) {
        cuda_execution::launch_copy_resident_eri_tile(stream, resident_, nbf_, begin, count,
                                                      elements, target.values);
      } else {
        cudaPointerAttributes attributes{};
        const auto status = cudaPointerGetAttributes(&attributes, target.values);
        if (status != cudaSuccess || attributes.type != cudaMemoryTypeDevice ||
            attributes.device != device_)
          throw std::invalid_argument("pending RHF ERI output is not on the original device");
        const bool full =
            std::all_of(begin.begin(), begin.end(), [](auto x) { return x == 0; }) &&
            std::all_of(count.begin(), count.end(), [&](auto x) { return x == nbf_; });
        // Same exact incumbent producer and orbit/rectangle selection as the
        // compact direct source; this view borrows the original live arena.
        if (full)
          cuda_execution::launch_build_eri_system_orbits(stream, view_, 0, elements, target.values);
        else
          cuda_execution::launch_independent_eri_tile(stream, view_, 0, begin, count, elements,
                                                      target.values);
      }
      const auto launch = cudaPeekAtLastError();
      if (launch != cudaSuccess)
        throw std::runtime_error("pending reference source launch failed: status=" +
                                 std::to_string(launch));
      const auto recorded = cudaEventRecord(static_cast<cudaEvent_t>(use.event), stream);
      if (recorded != cudaSuccess)
        throw std::runtime_error("pending reference source event record failed: status=" +
                                 std::to_string(recorded));
      use.recorded = true;
    } catch (...) {
      // Keep the original error; an unavailable drain leaves event/whole owner
      // retained. There is no context select or device-wide cleanup fallback.
      try {
        (void)owner_->finish_use_current(use);
      } catch (...) {
      }
      throw;
    }
  }

 private:
  core::System system_;
  std::shared_ptr<PendingReferenceProviderOwner> owner_;
  std::shared_ptr<std::atomic<bool>> lease_;
  cuda_execution::DeviceBatch view_;
  const double* resident_{};
  std::size_t retained_{}, nbf_{};
  int device_{-1};
};

}  // namespace

std::shared_ptr<const integrals::ElectronInteractionSource> make_pending_rhf_reference_source(
    core::System&& system, std::shared_ptr<PendingReferenceProviderOwner> owner,
    std::size_t retained_bytes) {
  if (!owner || !owner->bucket()) throw std::invalid_argument("pending source has no native owner");
  return std::make_shared<PendingRhfReferenceInteractionSource>(std::move(system), std::move(owner),
                                                                retained_bytes);
}

// Host-only single-system adapters share the bucket execution and error
// contract. Keep them outside the kernel translation unit so host changes
// do not require adding more code to the large CUDA implementation.
ScfResult run_rhf_cuda(
    const core::System& system, const ScfOptions& options, int device_id,
    const std::vector<double>* initial_density,
    std::shared_ptr<const integrals::ElectronInteractionSource>* interaction_source,
    CudaRhfSourceHandoff* handoff) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");
  if (interaction_source) interaction_source->reset();
  if (handoff) *handoff = {};

  if (interaction_source || handoff) {
    struct LocalPlan {
      CudaRhfBucketPlan* plan{};
      ~LocalPlan() { destroy_rhf_cuda_bucket_plan(plan); }
    } owner;
    return run_rhf_cuda_cached(&owner.plan, system, options, device_id, initial_density, nullptr,
                               interaction_source, handoff);
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
    std::shared_ptr<const integrals::ElectronInteractionSource>* interaction_source,
    CudaRhfSourceHandoff* handoff) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");
  if (!plan) throw std::invalid_argument("CUDA RHF cached execution requires a plan owner");
  if (execution_plan_reused) *execution_plan_reused = false;
  if (interaction_source) interaction_source->reset();
  if (handoff) *handoff = {};

  std::vector<core::System> systems{system};
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
  if (handoff && scf.reference) handoff->numeric_peak_bytes = scf.reference->numeric_capacity_bytes;
  if (options.reference_provider_initialization_owner) {
    // Keep one physical owner in both resident and bounded nonresident modes.
    // Borrowing incumbent inputs avoids a second compact CUDA plan/stream.
    if (interaction_source && scf.converged && scf.reference &&
        exact_reference_source_identity(*plan, system, options, device_id, true)) {
      try {
        auto& source_system = systems.front();
        const auto source_host = posthf::checked_add(
            posthf::source_capacity(source_system),
            posthf::checked_add(runtime::vector_bytes(source_system.ecp_terms),
                                sizeof(PendingRhfReferenceInteractionSource) + 128));
        const auto peak = posthf::checked_add(scf.reference->numeric_capacity_bytes, source_host);
        if (peak <= options.reference_memory_budget_bytes) {
          const auto retained =
              posthf::checked_add(hf_cuda_retained_numeric_bytes(*plan), source_host);
          auto source = make_pending_rhf_reference_source(
              std::move(source_system), options.reference_provider_initialization_owner, retained);
          *interaction_source = std::move(source);
          *plan = nullptr;  // logical reservation transfers; physical owner stays unchanged
          if (handoff) handoff->numeric_peak_bytes = peak;
        } else if (handoff)
          handoff->resource_fallback = true;
      } catch (const std::bad_alloc&) {
        if (handoff) handoff->resource_fallback = true;
      } catch (const std::length_error&) {
        if (handoff) handoff->resource_fallback = true;
      } catch (const std::overflow_error&) {
        if (handoff) handoff->resource_fallback = true;
      }
    }
    // If host publication cannot fit, the original cached bucket and ordinary
    // bounded post-HF producer remain available with their existing reservation.
    return scf;
  }
  const bool resident_eligible = interaction_source && scf.converged && scf.reference &&
                                 exact_reference_source_identity(*plan, system, options, device_id);
  if (resident_eligible) {
    try {
      auto& source_system = systems.front();
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
  // Resident values have priority. Otherwise detach only immutable metadata,
  // leaving a cached executable in its original slot for geometry replay. The
  // reference peak already covers that live plan; compaction adds its complete
  // source reservation before allocation. Resource rejection keeps RHF valid.
  if ((interaction_source || handoff) && !resident_eligible && *plan && scf.converged &&
      scf.reference) {
    auto compact = detach_rhf_cuda_source(**plan, std::move(systems.front()),
                                          scf.reference->numeric_capacity_bytes,
                                          options.reference_memory_budget_bytes);
    if (interaction_source) *interaction_source = compact.source;
    if (handoff) *handoff = std::move(compact);
  }
  return scf;
}

ScfResult run_rhf_cuda_with_source(const core::System& system, const ScfOptions& options,
                                   int device_id, const std::vector<double>* initial_density,
                                   CudaRhfSourceHandoff& handoff) {
  // This explicit compaction probe/API preserves the detached-source contract.
  // Normal correlated adapters request the preferred resident source as well.
  return run_rhf_cuda(system, options, device_id, initial_density, nullptr, &handoff);
}

bool reclaim_rhf_cuda_reference_plan(
    CudaRhfBucketPlan** plan,
    std::shared_ptr<const integrals::ElectronInteractionSource>& interaction_source) noexcept {
  if (!plan || *plan || !interaction_source || interaction_source.use_count() != 1) return false;
  if (const auto* pending =
          dynamic_cast<const PendingRhfReferenceInteractionSource*>(interaction_source.get())) {
    auto* released = pending->release_plan();
    if (!released) return false;
    interaction_source.reset();
    *plan = released;
    return true;
  }
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
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
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
