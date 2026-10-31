#include "scf/cuda/resources.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <type_traits>

#include "runtime/allocation_measurement.hpp"
#include "runtime/residency_cuda.cuh"
#include "runtime/resource_cuda.cuh"

namespace generativeqc::scf::cuda_execution {

CudaResources::~CudaResources() {
  if (explicit_retirement_managed_) {
    if (!explicit_retirement_closed_) std::terminate();
    return;  // Owner deletion was admitted only after verified full close.
  }
  std::lock_guard<std::mutex> allocation_lock(runtime::allocation_measurement_mutex);
  const runtime::ResidencyExecution source_execution(runtime::ResidencyOwner::hf_bucket_resources);
  if (device_id_ >= 0) (void)cudaSetDevice(device_id_);
  eigen_handles_.reset();
  if (blas_ != nullptr) (void)cublasDestroy(blas_);
  if (stream_ != nullptr) {
    // Numeric allocations come from CUDA's stream-ordered device pool. Queue
    // their release on the owning bucket stream so destroying one plan does
    // not impose a device-wide synchronization on unrelated workloads.
    if (reference_eri_ != nullptr) {
      (void)runtime::resource_cuda_free_async(reference_eri_, stream_);
    }
    if (reference_fock_correction_ != nullptr)
      (void)runtime::resource_cuda_free_async(reference_fock_correction_, stream_);
    if (solver_workspace_ != nullptr) {
      (void)runtime::resource_cuda_free_async(solver_workspace_, stream_);
    }
    if (direct_tile_validation_ != nullptr) {
      (void)runtime::resource_cuda_free_async(direct_tile_validation_, stream_);
    }
    if (arena_ != nullptr) (void)runtime::resource_cuda_free_async(arena_, stream_);
    // Resource release can run inside an endpoint, not just during final close.
    // Declare its lifetime boundary without turning it into publication/setup.
    (void)runtime::residency_stream_synchronize(source_execution, runtime::ResidencyRole::lifetime,
                                                runtime::ResidencySite::hf_bucket_release_fence,
                                                stream_);
    (void)cudaStreamDestroy(stream_);
  }
  std::free(solver_host_workspace_);
}

ProviderOwnerSchema CudaResources::inventory_schema() {
  using K = ProviderOwnedResourceKind;
  return {{{"stream", K::stream},
           {"jacobi", K::status_resource},
           {"parameters", K::status_resource},
           {"solver", K::status_resource},
           {"blas", K::status_resource},
           {"matrix_binding", K::host_object},
           {"structured_binding", K::host_object},
           {"reference_eri", K::device_buffer},
           {"fock_correction", K::device_buffer},
           {"solver_workspace", K::device_buffer},
           {"validation", K::device_buffer},
           {"arena", K::device_buffer},
           {"host_workspace", K::host_buffer}}};
}

ProviderOwnerInventory CudaResources::inventory() const {
  using K = ProviderOwnedResourceKind;
  const auto eigen = eigen_handles_.view();
  const auto token = [](auto pointer) { return reinterpret_cast<std::uintptr_t>(pointer); };
  return {{{"stream", K::stream, token(stream_)},
           {"jacobi", K::status_resource, token(eigen.jacobi)},
           {"parameters", K::status_resource, token(eigen.parameters)},
           {"solver", K::status_resource, token(eigen.solver)},
           {"blas", K::status_resource, token(blas_)},
           {"matrix_binding", K::host_object, matrix_binding_ ? token(&*matrix_binding_) : 0},
           {"structured_binding", K::host_object,
            structured_binding_ ? token(&*structured_binding_) : 0},
           {"reference_eri", K::device_buffer, token(reference_eri_)},
           {"fock_correction", K::device_buffer, token(reference_fock_correction_)},
           {"solver_workspace", K::device_buffer, token(solver_workspace_)},
           {"validation", K::device_buffer, token(direct_tile_validation_)},
           {"arena", K::device_buffer, token(arena_)},
           {"host_workspace", K::host_buffer, token(solver_host_workspace_)}}};
}

ProviderInitializationTeardownReport CudaResources::close_current_context(
    ProviderOwnerCloseExtension* extension) {
  // This entry owns the same measurement serialization as legacy destruction.
  // Its caller holds only the context/owner lock, never this nonrecursive mutex.
  std::lock_guard<std::mutex> allocation_lock(runtime::allocation_measurement_mutex);
  ProviderInitializationTeardownReport report;
  auto before = inventory();
  auto schema = inventory_schema();
  if (extension) {
    const auto extra = extension->extension_inventory();
    const auto extra_schema = extension->extension_schema();
    before.resources.insert(before.resources.begin() + 1, extra.resources.begin(),
                            extra.resources.end());
    schema.slots.insert(schema.slots.begin() + 1, extra_schema.slots.begin(),
                        extra_schema.slots.end());
  }
  if (!provider_inventory_matches_schema(schema, before))
    return {{}, "concrete extended resource inventory is unavailable"};
  const auto plan = provider_teardown_plan(before);
  bool stopped = false;
  const runtime::ResidencyExecution execution(runtime::ResidencyOwner::hf_bucket_resources);
  for (const auto& operation : plan) {
    if (!operation.created) {
      report.steps.push_back({operation.operation, ProviderTeardownStepState::not_created, {}});
      continue;
    }
    if (stopped) {
      report.steps.push_back({operation.operation, ProviderTeardownStepState::unavailable, {}});
      continue;
    }
    const auto name = operation.operation.substr(0, operation.operation.find('/'));
    if (extension) {
      const auto extra_schema = extension->extension_schema();
      if (std::any_of(extra_schema.slots.begin(), extra_schema.slots.end(),
                      [&](const auto& slot) { return slot.slot == name; })) {
        auto step = extension->close_extension_operation(operation.operation);
        const bool successful =
            operation.returns_status
                ? step.state == ProviderTeardownStepState::returned && step.status &&
                      *step.status == 0
                : step.state == ProviderTeardownStepState::completed_without_status && !step.status;
        if (step.operation != operation.operation || !successful) stopped = true;
        report.steps.push_back(std::move(step));
        continue;
      }
    }
    std::optional<std::uint32_t> status;
    bool statusless = false;
    if (operation.operation == "stream/drain")
      status = cudaStreamSynchronize(stream_);
    else if (operation.operation == "stream/release_fence")
      status = runtime::residency_stream_synchronize(
          execution, runtime::ResidencyRole::lifetime,
          runtime::ResidencySite::hf_bucket_release_fence, stream_);
    else if (operation.operation == "stream/release") {
      status = cudaStreamDestroy(stream_);
      if (*status == 0) stream_ = nullptr;
    } else if (name == "jacobi")
      status = eigen_handles_.retire_jacobi_current();
    else if (name == "parameters")
      status = eigen_handles_.retire_parameters_current();
    else if (name == "solver")
      status = eigen_handles_.retire_solver_current();
    else if (name == "blas") {
      status = cublasDestroy(blas_);
      if (*status == 0) blas_ = nullptr;
    } else if (name == "matrix_binding") {
      matrix_binding_.reset();
      statusless = true;
    } else if (name == "structured_binding") {
      structured_binding_.reset();
      statusless = true;
    } else if (name == "host_workspace") {
      std::free(solver_host_workspace_);
      solver_host_workspace_ = nullptr;
      solver_host_workspace_bytes_ = 0;
      statusless = true;
    } else {
      void* pointer = name == "reference_eri"      ? static_cast<void*>(reference_eri_)
                      : name == "fock_correction"  ? static_cast<void*>(reference_fock_correction_)
                      : name == "solver_workspace" ? solver_workspace_
                      : name == "validation"       ? static_cast<void*>(direct_tile_validation_)
                      : name == "arena"            ? arena_
                                                   : nullptr;
      if (pointer) {
        status = runtime::resource_cuda_free_async(pointer, stream_);
        if (*status == 0) {
          if (name == "reference_eri") {
            reference_eri_ = nullptr;
            reference_eri_bytes_ = 0;
          }
          if (name == "fock_correction") {
            reference_fock_correction_ = nullptr;
            reference_fock_correction_bytes_ = 0;
          }
          if (name == "solver_workspace") {
            solver_workspace_ = nullptr;
            solver_workspace_bytes_ = 0;
          }
          if (name == "validation") direct_tile_validation_ = nullptr;
          if (name == "arena") arena_ = nullptr;
        }
      }
    }
    if (statusless)
      report.steps.push_back(
          {operation.operation, ProviderTeardownStepState::completed_without_status, {}});
    else if (status) {
      report.steps.push_back({operation.operation, ProviderTeardownStepState::returned, status});
      if (*status != 0) stopped = true;
    } else {
      report.steps.push_back({operation.operation, ProviderTeardownStepState::unavailable, {}});
      stopped = true;
    }
  }
  return report;
}

void CudaResources::mark_teardown_complete() {
  const auto state = inventory();
  if (!explicit_retirement_managed_ ||
      std::any_of(state.resources.begin(), state.resources.end(),
                  [](const auto& value) { return value.token != 0; }))
    throw std::logic_error("cannot delete a live explicitly managed CUDA resource owner");
  explicit_retirement_closed_ = true;
}

EigensolverResources CudaResources::eigensolver_view() const {
  return {stream_,
          static_cast<cusolverDnHandle_t>(eigen_handles_.view().solver),
          static_cast<cusolverDnParams_t>(eigen_handles_.view().parameters),
          static_cast<syevjInfo_t>(eigen_handles_.view().jacobi),
          solver_workspace_,
          solver_workspace_bytes_,
          solver_host_workspace_,
          solver_host_workspace_bytes_};
}

}  // namespace generativeqc::scf::cuda_execution
