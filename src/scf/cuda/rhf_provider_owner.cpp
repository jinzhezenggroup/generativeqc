#include <algorithm>
#include <cstdlib>
#include <exception>
#include <type_traits>

#include "runtime/allocation_measurement.hpp"
#include "runtime/residency_cuda.cuh"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/direct_tile_validation.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/provider_initialization_owner.hpp"
#include "scf/provider_initialization_runtime.hpp"

namespace generativeqc::scf {
namespace {
class NativePendingReferenceOwner final
    : public PendingReferenceProviderOwner,
      public ProviderOwnerCloseExtension,
      public std::enable_shared_from_this<NativePendingReferenceOwner> {
 public:
  struct Generation {
    NativePendingReferenceOwner& owner;
    std::unique_ptr<PendingProviderInitializationGeneration> bound;
    std::size_t declared_allowance() const {
      if (runtime::active_device_resource_ledger)
        throw ProviderInitializationUnavailable("pending reference has unknown ledger basis");
      return owner.intent_.allowance_bytes();
    }
    bool has_bound_identity() const noexcept { return bound && bound->has_bound_identity(); }
    template <class Observe, class Close>
    ProviderInitializationTeardownReport retire_current(Observe&& observe, Close&& close) {
      if (!bound) return {{}, "owned hardware has no bound original identity"};
      return bound->retire_current(std::forward<Observe>(observe), std::forward<Close>(close));
    }
  };

  NativePendingReferenceOwner(ProviderInitializationIntent intent, int device,
                              std::size_t reference_budget, std::size_t endpoint_budget)
      : intent_(intent),
        device_(device),
        budget_{intent.allowance_bytes(), reference_budget, endpoint_budget},
        generation_{*this, {}} {
    if (device < 0) throw std::invalid_argument("pending reference requires a native device");
    (void)generation_.declared_allowance();
    if (intent.allowance_bytes() > reference_budget || intent.allowance_bytes() > endpoint_budget)
      throw std::length_error(
          "provider allowance exceeds the original reference/endpoint envelope");
  }
  ~NativePendingReferenceOwner() override {
    if (!cell_.destroy_closed()) std::terminate();
  }
  Generation& generation() noexcept { return generation_; }
  std::unique_lock<std::recursive_mutex> acquire_borrow_lock() override {
    return std::unique_lock<std::recursive_mutex>(mutex_);
  }
  void attach_bucket(CudaRhfBucketPlan* bucket) override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (closed_ || stale_ || bucket_ || !bucket)
      throw ProviderInitializationUnavailable(
          "pending reference cannot replace its original owner");
    bucket_ = bucket;
    // The method already owns this inert state. The manual cell is established
    // before this call returns and before the driver creates any CUDA object.
    if (!cell_.adopt(shared_from_this(), budget_))
      throw ProviderInitializationUnavailable("pending reference retention cell is occupied");
    retained_ = true;
  }
  CudaRhfBucketPlan* bucket() const noexcept override { return bucket_; }
  bool closed() const noexcept override { return closed_; }
  bool identity_bound() const noexcept override { return generation_.has_bound_identity(); }
  std::size_t allowance_bytes() const noexcept override { return intent_.allowance_bytes(); }
  bool source_lease_active() const noexcept override {
    return source_lease_ && source_lease_->load(std::memory_order_acquire);
  }
  void register_cache_slot(CudaRhfBucketPlan** slot, CudaRhfBucketPlan** previous) override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (cache_slot_ && cache_slot_ != previous && cache_slot_ != slot)
      throw std::logic_error("pending reference already has a cache owner");
    cache_slot_ = slot;
  }
  void bind_current(ReferenceProviders providers,
                    std::optional<ProviderInitializationEigenFamily> family) override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!retained_ || !bucket_ || closed_ || stale_)
      throw ProviderInitializationUnavailable("pending reference was not retained before binding");
    if (generation_.bound) {
      if (providers.mask() != providers_.mask() || family != family_) {
        stale_ = true;
        throw ProviderInitializationUnavailable("selected provider family changed");
      }
      require_current();
      return;
    }
    providers_ = providers;
    family_ = family;
    auto candidate = std::make_unique<PendingProviderInitializationGeneration>(intent_, providers,
                                                                               family, device_);
    const auto observation = observe();
    candidate->bind_current([&] { return observation; });
    pins_ = observation.library_pins;
    generation_.bound = std::move(candidate);
  }
  void require_current() override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    (void)generation_.declared_allowance();
    if (closed_ || stale_)
      throw ProviderInitializationUnavailable("pending reference generation is retired");
    if (generation_.bound) {
      try {
        generation_.bound->require_reuse([&] { return observe(); });
      } catch (...) {
        stale_ = true;
        throw;
      }
    }
  }
  void begin_reference_replay() override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_current();
    if (!finish_uses_current())
      throw ProviderInitializationUnavailable("prior reference borrowers did not complete");
    if (source_lease_) source_lease_->store(false, std::memory_order_release);
    if (cache_slot_ && !*cache_slot_) *cache_slot_ = bucket_;
  }
  const ResolvedProviderInitialization& resolved() const override {
    if (!generation_.bound) throw ProviderInitializationUnavailable("pending reference is unbound");
    return generation_.bound->resolved();
  }
  const std::array<std::shared_ptr<void>, 4>& pins() const override { return pins_; }
  void mark_uncertain(const std::string& operation, std::optional<std::uint32_t> status) override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    stale_ = true;
    // A refused void reset performs no hardware call. An actual nonzero status
    // may have consumed a handle, so its first result permanently forbids retry.
    if (status && !uncertain_) {
      uncertain_ = true;
      uncertainty_ = {{{operation, ProviderTeardownStepState::returned, status}},
                      "physical owner operation unavailable; original owner retained"};
      uncertainty_.detail += "; " + provider_teardown_failure_detail(uncertainty_);
    }
  }
  ProviderReferenceStreamUse& prepare_use(void* stream) override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_current();
    if (!generation_.has_bound_identity())
      throw ProviderInitializationUnavailable("source borrow precedes native identity binding");
    for (auto& use : uses_)
      if (use.stream == stream) return use;
    if (inventory_schema().slots.size() + 1 > kProviderOwnerInventoryLimit)
      throw ProviderInitializationUnavailable("finite consumer-stream inventory is exhausted");
    uses_.push_back({stream, nullptr, false, false});
    event_slots_ = uses_.size();
    cudaEvent_t event{};
    std::lock_guard<std::mutex> measurement(runtime::allocation_measurement_mutex);
    const auto status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    uses_.back().event = event;  // retain any actually returned handle, even on error
    if (status != cudaSuccess || !event)
      throw ProviderInitializationUnavailable(
          "native reference consumer event creation failed: status=" + std::to_string(status));
    return uses_.back();
  }
  bool finish_use_current(ProviderReferenceStreamUse& use) override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_current();
    if (!use.pending) return true;
    const auto status = use.recorded ? cudaEventSynchronize(static_cast<cudaEvent_t>(use.event))
                                     : cudaStreamSynchronize(static_cast<cudaStream_t>(use.stream));
    if (status == cudaSuccess) {
      use.pending = false;
      return true;
    }
    mark_uncertain("source consumer/drain", static_cast<std::uint32_t>(status));
    return false;
  }
  bool finish_uses_current() override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (auto& use : uses_)
      if (!finish_use_current(use)) return false;
    return true;
  }
  std::shared_ptr<std::atomic<bool>> acquire_source_lease() override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_current();
    if (!generation_.has_bound_identity())
      throw ProviderInitializationUnavailable(
          "source publication precedes native identity binding");
    if (source_lease_) source_lease_->store(false, std::memory_order_release);
    source_lease_ = std::make_shared<std::atomic<bool>>(true);
    return source_lease_;
  }
  void release_source_lease(const std::shared_ptr<std::atomic<bool>>& lease) noexcept override {
    try {
      std::lock_guard<std::recursive_mutex> lock(mutex_);
      if (lease) lease->store(false, std::memory_order_release);
      if (source_lease_ == lease && !closed_ && cache_slot_ && !*cache_slot_)
        *cache_slot_ = bucket_;
    } catch (...) {
    }
  }
  ProviderInitializationTeardownReport close_current() override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (closed_) return last_;
    if (uncertain_) return has_close_report_ ? last_ : uncertainty_;
    if (!retained_) {
      const auto before = inventory();
      if (!unbound_host_only_inventory())
        return {{}, "unretained candidate unexpectedly owns CUDA resources"};
      auto raw = close_current_context();
      const auto after = inventory();
      last_ = verify_provider_owner_teardown(inventory_schema(), before, after, std::move(raw));
      if (last_.confirmed()) mark_teardown_complete();
      return last_;
    }
    last_ = cell_.try_close_current([&] { return observe(); });
    has_close_report_ = true;
    return last_;
  }

  // ProviderOwnerRetentionCell protocol: complete real inventory, never caller evidence.
  ProviderOwnerSchema inventory_schema() const {
    auto schema = cuda_execution::CudaResources::inventory_schema();
    const auto extra = extension_schema();
    schema.slots.insert(schema.slots.begin() + 1, extra.slots.begin(), extra.slots.end());
    return schema;
  }
  ProviderOwnerInventory inventory() const {
    ProviderOwnerInventory state;
    if (bucket_)
      state = bucket_->resources.inventory();
    else
      for (const auto& slot : cuda_execution::CudaResources::inventory_schema().slots)
        state.resources.push_back({slot.slot, slot.kind, 0});
    const auto extra = extension_inventory();
    state.resources.insert(state.resources.begin() + 1, extra.resources.begin(),
                           extra.resources.end());
    return state;
  }
  bool unbound_host_only_inventory() const {
    for (const auto& resource : inventory().resources)
      if (resource.token && resource.kind != ProviderOwnedResourceKind::host_buffer &&
          resource.kind != ProviderOwnedResourceKind::host_object)
        return false;
    return true;
  }
  void enter_explicit_retirement_protocol() {
    bucket_->resources.enter_explicit_retirement_protocol();
    const auto weak = weak_from_this();
    bucket_->graphs.enter_provider_retirement(
        [weak] {
          auto self = weak.lock();
          if (!self) throw ProviderInitializationUnavailable("graph owner unavailable");
          self->require_current();
        },
        [weak](const std::string& op, std::optional<std::uint32_t> status) {
          if (auto self = weak.lock()) self->mark_uncertain(op, status);
        });
  }
  ProviderInitializationTeardownReport close_current_context() {
    if (bucket_) return bucket_->resources.close_current_context(this);
    ProviderInitializationTeardownReport report;
    for (const auto& operation : provider_teardown_plan(inventory()))
      report.steps.push_back({operation.operation, ProviderTeardownStepState::not_created, {}});
    return report;
  }
  void mark_teardown_complete() {
    if (bucket_) {
      bucket_->graphs.mark_provider_closed();
      bucket_->resources.mark_teardown_complete();
      if (cache_slot_ && *cache_slot_ == bucket_) *cache_slot_ = nullptr;
      auto* released = std::exchange(bucket_, nullptr);
      delete released;  // every physical member is already closed; no CUDA dtor calls
    }
    source_lease_.reset();
    pins_ = {};
    closed_ = true;
  }
  ProviderOwnerSchema extension_schema() const override {
    using K = ProviderOwnedResourceKind;
    ProviderOwnerSchema schema{{{"source_access_lease", K::host_object}}};
    for (std::size_t i = 0; i < event_slots_; ++i)
      schema.slots.push_back({"source_event_" + std::to_string(i), K::event});
    schema.slots.push_back({"source_use_storage", K::host_object});
    const auto graph = cuda_execution::RhfIterationGraphs::provider_schema();
    schema.slots.insert(schema.slots.end(), graph.slots.begin(), graph.slots.end());
    schema.slots.push_back({"bucket_host_payload", K::host_object});
    return schema;
  }
  ProviderOwnerInventory extension_inventory() const override {
    using K = ProviderOwnedResourceKind;
    ProviderOwnerInventory state{
        {{"source_access_lease", K::host_object,
          source_lease_active() ? reinterpret_cast<std::uintptr_t>(source_lease_.get()) : 0}}};
    for (std::size_t i = 0; i < event_slots_; ++i)
      state.resources.push_back(
          {"source_event_" + std::to_string(i), K::event,
           i < uses_.size() ? reinterpret_cast<std::uintptr_t>(uses_[i].event) : 0});
    state.resources.push_back(
        {"source_use_storage", K::host_object,
         uses_.capacity() ? reinterpret_cast<std::uintptr_t>(uses_.data()) : 0});
    if (bucket_) {
      const auto graph = bucket_->graphs.provider_inventory();
      state.resources.insert(state.resources.end(), graph.resources.begin(), graph.resources.end());
    } else
      for (const auto& slot : cuda_execution::RhfIterationGraphs::provider_schema().slots)
        state.resources.push_back({slot.slot, slot.kind, 0});
    state.resources.push_back(
        {"bucket_host_payload", K::host_object,
         host_payload_bytes() ? reinterpret_cast<std::uintptr_t>(bucket_) : 0});
    return state;
  }
  ProviderInitializationTeardownStep close_extension_operation(
      const std::string& operation) override {
    const auto slash = operation.find('/');
    const auto name = operation.substr(0, slash);
    if (name == "source_access_lease") {
      source_lease_->store(false, std::memory_order_release);
      return {operation, ProviderTeardownStepState::completed_without_status, {}};
    }
    if (name.starts_with("source_event_")) {
      const auto index = static_cast<std::size_t>(std::stoull(name.substr(13)));
      auto& use = uses_.at(index);
      cudaError_t status;
      if (operation.substr(slash + 1) == "drain") {
        status = use.pending && !use.recorded
                     ? cudaStreamSynchronize(static_cast<cudaStream_t>(use.stream))
                     : cudaEventSynchronize(static_cast<cudaEvent_t>(use.event));
        if (status == cudaSuccess) use.pending = false;
      } else {
        status = cudaEventDestroy(static_cast<cudaEvent_t>(use.event));
        if (status == cudaSuccess) use.event = nullptr;
      }
      return {operation, ProviderTeardownStepState::returned, static_cast<std::uint32_t>(status)};
    }
    if (name == "source_use_storage") {
      for (const auto& use : uses_)
        if (use.event) return {operation, ProviderTeardownStepState::unavailable, {}};
      std::vector<ProviderReferenceStreamUse>{}.swap(uses_);
      return {operation, ProviderTeardownStepState::completed_without_status, {}};
    }
    if (name == "bucket_host_payload") {
      release_host_payload();
      return {operation, ProviderTeardownStepState::completed_without_status, {}};
    }
    if (bucket_) return bucket_->graphs.close_provider_operation(operation);
    return {operation, ProviderTeardownStepState::unavailable, {}};
  }

 private:
  runtime::CudaProviderIdentityObservation observe() const {
    return runtime::observe_cuda_provider_identity(
        providers_.mask(), {reinterpret_cast<const void*>(&cudaRuntimeGetVersion),
                            reinterpret_cast<const void*>(&cublasCreate),
                            reinterpret_cast<const void*>(&cusolverDnCreate)});
  }
  std::size_t host_payload_bytes() const {
    if (!bucket_) return 0;
    const auto binding = bucket_->resources.matrix_binding_
                             ? tensor::CudaMatrixProductBinding::kPreparationHostBytes +
                                   tensor::CudaStructuredProductBinding::kPreparationHostBytes
                             : 0;
    return hf_cuda_retained_host_numeric_bytes(*bucket_) - binding;
  }
  void release_host_payload() {
    auto& p = *bucket_;
    p.topology = {};
    const auto release = [](auto&... vectors) {
      (std::decay_t<decltype(vectors)>{}.swap(vectors), ...);
    };
    release(p.cached_positions, p.resident_warm_positions, p.resident_warm_density,
            p.resident_previous_energy, p.frozen_warm_positions, p.frozen_warm_density,
            p.frozen_previous_energy, p.bounded_direct_shell_pair_order,
            p.bounded_stream_shell_pair_order, p.bounded_stream_pair_class_offsets,
            p.mixed_precision_system_census);
    p.last_ppps_queue_profile.reset();
    p.last_inactive_eigensolver_profile.reset();
  }
  const ProviderInitializationIntent intent_;
  const int device_;
  const ProviderOwnerOriginalBudget budget_;
  mutable std::recursive_mutex mutex_;
  ProviderOwnerRetentionCell<NativePendingReferenceOwner> cell_;
  Generation generation_;
  CudaRhfBucketPlan* bucket_{};
  CudaRhfBucketPlan** cache_slot_{};
  ReferenceProviders providers_{};
  std::optional<ProviderInitializationEigenFamily> family_;
  std::array<std::shared_ptr<void>, 4> pins_;
  std::vector<ProviderReferenceStreamUse> uses_;
  std::size_t event_slots_{};
  std::shared_ptr<std::atomic<bool>> source_lease_;
  bool retained_{}, closed_{}, stale_{}, uncertain_{}, has_close_report_{};
  ProviderInitializationTeardownReport last_, uncertainty_;
};
}  // namespace

std::shared_ptr<PendingReferenceProviderOwner> make_pending_reference_provider_owner(
    ProviderInitializationIntent intent, int device, std::size_t reference_budget,
    std::size_t endpoint_budget) {
  return std::make_shared<NativePendingReferenceOwner>(intent, device, reference_budget,
                                                       endpoint_budget);
}
}  // namespace generativeqc::scf
