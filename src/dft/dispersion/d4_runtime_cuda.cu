#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "dft/dispersion/d4_cuda.hpp"
#include "dft/dispersion/d4_runtime.hpp"
#include "generated_d4_derivative.hpp"
#include "runtime/resource_cuda.cuh"

#if defined(GENERATIVEQC_TEST_HOOKS)
namespace {
thread_local bool fail_next_d4_after_coordinate_upload = false;
thread_local bool fail_next_d4_after_allocations = false;
}  // namespace
extern "C" void d4_cuda_fail_after_allocations_for_test_v1() {
  fail_next_d4_after_allocations = true;
}
extern "C" void d4_cuda_fail_after_coordinate_upload_for_test_v1() {
  fail_next_d4_after_coordinate_upload = true;
}
#endif

namespace generativeqc::dft::dispersion {

struct D4CudaOwner {
  int device_id{-1};
  cudaStream_t stream{};
  std::uint32_t systems{};
  std::uint64_t atoms{};
  std::uint32_t workers{};
  std::size_t eeq_workspace_stride{};
  std::size_t dqdr_stride{};

  std::uint32_t* offsets{};
  std::int32_t* atomic_numbers{};
  double* total_charges{};
  double* coordinates{};
  std::uint8_t* active{};
  std::uint8_t* want_gradient{};
  std::uint8_t* fixed_active{};
  D4Status* statuses{};
  D4Status* fixed_statuses{};
  double* energies{};
  double* gradients{};
  double* charges{};
  double* dedq{};
  std::uint32_t* next_system{};
  double* eeq_workspace{};
  double* dqdr_workspace{};
  double* fixed_workspace{};
  data::D4ElementData* elements{};
  data::D4ReferenceData* references{};
  double* reference_c6{};
  D4EEQChargeElementData* charge_elements{};
};

namespace {

class DeviceScope {
 public:
  explicit DeviceScope(int requested) {
    error_ = cudaGetDevice(&previous_);
    if (error_ == cudaSuccess && previous_ != requested) {
      error_ = cudaSetDevice(requested);
      restore_ = error_ == cudaSuccess;
    }
  }
  ~DeviceScope() {
    if (restore_) cudaSetDevice(previous_);
  }
  [[nodiscard]] cudaError_t error() const noexcept { return error_; }

 private:
  int previous_{-1};
  bool restore_{false};
  cudaError_t error_{cudaSuccess};
};

generativeqc_status cuda_failure(cudaError_t error, const char* action, std::string& detail) {
  detail = std::string(action) + ": " + cudaGetErrorString(error);
  return error == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                            : GENERATIVEQC_STATUS_CUDA_ERROR;
}

template <typename T>
bool allocate(T*& pointer, std::size_t count, std::string& detail) {
  if (!count) {
    pointer = nullptr;
    return true;
  }
  // A prepared scope owns the actual bytes; without a scope this preserves
  // the standalone CUDA allocation contract.
  const auto error = runtime::resource_cuda_malloc(&pointer, count * sizeof(T));
  if (error == cudaSuccess) return true;
  detail = std::string("D4 CUDA allocation failed: ") + cudaGetErrorString(error);
  return false;
}

template <typename T>
bool upload(cudaStream_t stream, T* destination, const T* source, std::size_t count,
            std::string& detail) {
  if (!count) return true;
  const auto error =
      cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) return true;
  detail = std::string("D4 CUDA setup upload failed: ") + cudaGetErrorString(error);
  return false;
}

__device__ void zero_item(std::uint32_t system, const std::uint32_t* offsets, double* energies,
                          double* gradients, double* charges) {
  const std::size_t begin = offsets[system];
  const std::size_t end = offsets[system + 1];
  energies[2 * system] = 0.0;
  energies[2 * system + 1] = 0.0;
  for (std::size_t atom = begin; atom < end; ++atom) {
    charges[atom] = 0.0;
    gradients[3 * atom] = 0.0;
    gradients[3 * atom + 1] = 0.0;
    gradients[3 * atom + 2] = 0.0;
  }
}

// One CTA owns one EEQ system. Lane zero runs every order-dependent stage
// (validation, CN/matrix assembly, the charge solve and response assembly)
// through the same shared stage functions as the host provider. Only work
// whose stored values do not depend on traversal order is distributed: the
// row updates of one LU elimination step, independent response-column solves
// and plain stores. Every published value is therefore bitwise identical to
// the former single-thread provider.
constexpr unsigned kEEQThreads = 128;

struct EEQCooperativeShared {
  int pivots[kEEQMaximumAtoms + 1];
  D4Status status;
  int stop;
  std::uint32_t system;
};

// Read a lane-zero decision on every lane. The trailing barrier keeps lane
// zero from publishing its next decision before all lanes consumed this one.
__device__ bool cta_decision(const int& flag) {
  __syncthreads();
  const bool value = flag != 0;
  __syncthreads();
  return value;
}

__device__ D4Status cooperative_eeq2019(int n, const std::int32_t* z, const double* xyz,
                                        double total_charge, EEQTables t, double* workspace,
                                        std::size_t workspace_size, double* charges, double* dqdr,
                                        bool responses, EEQCooperativeShared& shared) {
  const unsigned lane = threadIdx.x, width = blockDim.x;
  if (lane == 0) {
    bool complete = false;
    shared.status = eeq_stage::validate(n, z, xyz, total_charge, t, workspace, workspace_size,
                                        charges, dqdr, &complete);
    shared.stop = complete ? 1 : 0;
  }
  // status is published before the decision and rewritten only after it.
  if (cta_decision(shared.stop)) return shared.status;
  const auto w = eeq_stage::layout(n, workspace);
  const int m = w.m;
  const std::size_t used = eeq2019_workspace_elements(n);
  for (std::size_t i = lane; i < used; i += width) workspace[i] = 0.0;
  __syncthreads();
  if (lane == 0) {
    shared.status = eeq_stage::assemble(n, z, xyz, total_charge, t, w);
    shared.stop = shared.status != D4Status::success;
  }
  if (cta_decision(shared.stop)) return shared.status;

  // Partial-pivot LU with the serial pivot choice; rows below the pivot are
  // independent within each step and use the provider's exact update form.
  double* a = w.lu;
  for (int k = 0; k < m; ++k) {
    if (lane == 0) {
      int pivot = k;
      double maximum = fabs(a[static_cast<std::size_t>(k) * m + k]);
      for (int i = k + 1; i < m; ++i) {
        const double value = fabs(a[static_cast<std::size_t>(i) * m + k]);
        if (value > maximum) {
          maximum = value;
          pivot = i;
        }
      }
      shared.stop = !(maximum > 1.0e-15) || !eeq_detail::finite(maximum);
      shared.pivots[k] = pivot;
    }
    if (cta_decision(shared.stop)) return D4Status::numerical_failure;
    const int pivot = shared.pivots[k];
    if (pivot != k)
      for (int j = static_cast<int>(lane); j < m; j += static_cast<int>(width)) {
        const auto p = static_cast<std::size_t>(pivot) * m + j;
        const auto q = static_cast<std::size_t>(k) * m + j;
        const double tmp = a[p];
        a[p] = a[q];
        a[q] = tmp;
      }
    __syncthreads();
    const double diagonal = a[static_cast<std::size_t>(k) * m + k];
    for (int i = k + 1 + static_cast<int>(lane); i < m; i += static_cast<int>(width))
      a[static_cast<std::size_t>(i) * m + k] /= diagonal;
    __syncthreads();
    const std::size_t extent = static_cast<std::size_t>(m - k - 1);
    for (std::size_t e = lane; e < extent * extent; e += width) {
      const int i = k + 1 + static_cast<int>(e / extent);
      const int j = k + 1 + static_cast<int>(e % extent);
      const double factor = a[static_cast<std::size_t>(i) * m + k];
      a[static_cast<std::size_t>(i) * m + j] -= factor * a[static_cast<std::size_t>(k) * m + j];
    }
    __syncthreads();
  }
  if (lane == 0) {
    shared.stop = !eeq_detail::lu_solve(w.lu, m, shared.pivots, w.solution);
    if (!shared.stop && responses) eeq_stage::response_rhs(n, z, xyz, t, w);
  }
  if (cta_decision(shared.stop)) return D4Status::numerical_failure;
  if (responses) {
    bool response_failed = false;
    for (int coordinate = static_cast<int>(lane); coordinate < 3 * n;
         coordinate += static_cast<int>(width))
      if (!eeq_stage::solve_response(n, w, shared.pivots, coordinate)) response_failed = true;
    if (__syncthreads_or(response_failed)) return D4Status::numerical_failure;
  }
  // Transactional publication after every requested solve has succeeded.
  for (int i = static_cast<int>(lane); i < n; i += static_cast<int>(width))
    charges[i] = w.solution[i];
  if (responses)
    for (std::size_t e = lane; e < 3 * static_cast<std::size_t>(n) * n; e += width) {
      const auto coordinate = e / static_cast<std::size_t>(n), i = e % static_cast<std::size_t>(n);
      dqdr[e] = w.responses[coordinate * m + i];
    }
  __syncthreads();
  return D4Status::success;
}

__device__ bool next_eeq_system(std::uint32_t systems, std::uint32_t* next_system,
                                EEQCooperativeShared& shared) {
  __syncthreads();
  if (threadIdx.x == 0) shared.system = atomicAdd(next_system, 1u);
  __syncthreads();
  return shared.system < systems;
}

__global__ void __launch_bounds__(kEEQThreads)
    eeq_prepare_kernel(std::uint32_t systems, const std::uint32_t* offsets,
                       const std::int32_t* atomic_numbers, const double* total_charges,
                       const double* coordinates, const std::uint8_t* active, EEQTables eeq_tables,
                       std::uint32_t* next_system, double* eeq_workspace,
                       std::size_t eeq_workspace_stride, double* dqdr_workspace,
                       std::size_t dqdr_stride, D4Status* statuses, std::uint8_t* fixed_active,
                       double* charges, double* energies, double* gradients) {
  __shared__ EEQCooperativeShared shared;
  const auto slot = static_cast<std::size_t>(blockIdx.x);
  double* workspace = eeq_workspace + slot * eeq_workspace_stride;
  double* dqdr = dqdr_workspace + slot * dqdr_stride;
  while (next_eeq_system(systems, next_system, shared)) {
    const std::uint32_t system = shared.system;
    if (!active[system]) {
      if (threadIdx.x == 0) {
        statuses[system] = D4Status::success;
        fixed_active[system] = 0;
        zero_item(system, offsets, energies, gradients, charges);
      }
      continue;
    }
    const std::size_t begin = offsets[system];
    const int n = static_cast<int>(offsets[system + 1] - offsets[system]);
    // Fixed-charge D4 consumes only q; the compose pass forms dq/dR itself.
    const auto status = cooperative_eeq2019(
        n, atomic_numbers + begin, coordinates + 3 * begin, total_charges[system], eeq_tables,
        workspace, eeq2019_workspace_elements(n), charges + begin, dqdr, false, shared);
    if (threadIdx.x == 0) {
      statuses[system] = status;
      fixed_active[system] = status == D4Status::success ? 1 : 0;
      if (status != D4Status::success) zero_item(system, offsets, energies, gradients, charges);
    }
  }
}

__global__ void __launch_bounds__(kEEQThreads)
    eeq_compose_kernel(std::uint32_t systems, const std::uint32_t* offsets,
                       const std::int32_t* atomic_numbers, const double* total_charges,
                       const double* coordinates, const std::uint8_t* active,
                       const std::uint8_t* want_gradient, const D4Status* fixed_statuses,
                       EEQTables eeq_tables, std::uint32_t* next_system, double* eeq_workspace,
                       std::size_t eeq_workspace_stride, double* dqdr_workspace,
                       std::size_t dqdr_stride, D4Status* statuses, double* energies,
                       double* gradients, double* charges, const double* dedq) {
  __shared__ EEQCooperativeShared shared;
  const auto slot = static_cast<std::size_t>(blockIdx.x);
  double* workspace = eeq_workspace + slot * eeq_workspace_stride;
  double* dqdr = dqdr_workspace + slot * dqdr_stride;
  while (next_eeq_system(systems, next_system, shared)) {
    const std::uint32_t system = shared.system;
    if (!active[system]) {
      if (threadIdx.x == 0) statuses[system] = D4Status::success;
      continue;
    }
    if (statuses[system] != D4Status::success) {
      if (threadIdx.x == 0) zero_item(system, offsets, energies, gradients, charges);
      continue;
    }
    if (fixed_statuses[system] != D4Status::success) {
      if (threadIdx.x == 0) {
        statuses[system] = fixed_statuses[system];
        zero_item(system, offsets, energies, gradients, charges);
      }
      continue;
    }
    const std::size_t begin = offsets[system];
    const int n = static_cast<int>(offsets[system + 1] - offsets[system]);
    if (!want_gradient[system]) {
      for (int coordinate = static_cast<int>(threadIdx.x); coordinate < 3 * n;
           coordinate += static_cast<int>(blockDim.x))
        gradients[3 * begin + coordinate] = 0.0;
      continue;
    }
    auto status = cooperative_eeq2019(
        n, atomic_numbers + begin, coordinates + 3 * begin, total_charges[system], eeq_tables,
        workspace, eeq2019_workspace_elements(n), charges + begin, dqdr, true, shared);
    if (threadIdx.x == 0) {
      if (status == D4Status::success)
        status = ::generativeqc::generated::d4::compose_eeq_gradient(
            n, dqdr, dedq + begin, gradients + 3 * begin, gradients + 3 * begin);
      statuses[system] = status;
      if (status != D4Status::success) zero_item(system, offsets, energies, gradients, charges);
    }
  }
}

}  // namespace

D4CudaOwner* create_d4_cuda_owner(int device_id, std::span<const std::uint32_t> offsets,
                                  std::span<const std::int32_t> atomic_numbers,
                                  std::span<const double> total_charges,
                                  std::span<const double> default_coordinates, D4EEQProfile profile,
                                  const D4ResourceUsage& resources, std::string& detail,
                                  generativeqc_status& status) {
  status = GENERATIVEQC_STATUS_CUDA_ERROR;
  DeviceScope scope(device_id);
  if (scope.error() != cudaSuccess) {
    status = cuda_failure(scope.error(), "select D4 CUDA device", detail);
    return nullptr;
  }

  // Keep every partial stream/allocation owned even if constructing an error
  // detail throws on the host. The same deleter retires ledger charges after
  // the submitting scope has ended.
  std::unique_ptr<D4CudaOwner, decltype(&destroy_d4_cuda_owner)> owner(new D4CudaOwner,
                                                                       destroy_d4_cuda_owner);
  owner->device_id = device_id;
  owner->systems = static_cast<std::uint32_t>(offsets.size() - 1);
  owner->atoms = atomic_numbers.size();
  owner->workers = resources.worker_blocks;
  owner->eeq_workspace_stride =
      eeq2019_workspace_elements(static_cast<int>(resources.maximum_atoms));
  owner->dqdr_stride = 3u * resources.maximum_atoms * resources.maximum_atoms;
  if (!owner->workers || !owner->eeq_workspace_stride || !owner->dqdr_stride) {
    detail = "invalid D4 CUDA EEQ worker/workspace schedule";
    status = GENERATIVEQC_STATUS_INTERNAL_ERROR;
    return nullptr;
  }

  auto error = cudaStreamCreateWithFlags(&owner->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    status = cuda_failure(error, "create D4 nonblocking stream", detail);
    return nullptr;
  }

  const auto atoms = static_cast<std::size_t>(owner->atoms);
  const auto systems = static_cast<std::size_t>(owner->systems);
  const auto host_eeq_tables = eeq2019_host_tables();
  const auto host_d4_tables = eeq_d4_host_tables(profile);
  const auto c6_count = host_d4_tables.reference_c6_count;
  if (!allocate(owner->offsets, offsets.size(), detail) ||
      !allocate(owner->atomic_numbers, atoms, detail) ||
      !allocate(owner->total_charges, systems, detail) ||
      !allocate(owner->coordinates, 3 * atoms, detail) ||
      !allocate(owner->active, systems, detail) ||
      !allocate(owner->want_gradient, systems, detail) ||
      !allocate(owner->fixed_active, systems, detail) ||
      !allocate(owner->statuses, systems, detail) ||
      !allocate(owner->fixed_statuses, systems, detail) ||
      !allocate(owner->energies, 2 * systems, detail) ||
      !allocate(owner->gradients, 3 * atoms, detail) || !allocate(owner->charges, atoms, detail) ||
      !allocate(owner->dedq, atoms, detail) || !allocate(owner->next_system, 1, detail) ||
      !allocate(owner->eeq_workspace,
                static_cast<std::size_t>(owner->workers) * owner->eeq_workspace_stride, detail) ||
      !allocate(owner->dqdr_workspace,
                static_cast<std::size_t>(owner->workers) * owner->dqdr_stride, detail) ||
      !allocate(owner->fixed_workspace, d4_cuda_workspace_elements(atoms), detail) ||
      !allocate(owner->elements, host_d4_tables.element_count, detail) ||
      !allocate(owner->references, host_d4_tables.reference_count, detail) ||
      !allocate(owner->reference_c6, c6_count, detail) ||
      !allocate(owner->charge_elements, host_eeq_tables.element_count, detail)) {
    status = GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    return nullptr;
  }

#if defined(GENERATIVEQC_TEST_HOOKS)
  if (std::exchange(fail_next_d4_after_allocations, false)) throw std::bad_alloc();
#endif

  if (!upload(owner->stream, owner->offsets, offsets.data(), offsets.size(), detail) ||
      !upload(owner->stream, owner->atomic_numbers, atomic_numbers.data(), atoms, detail) ||
      !upload(owner->stream, owner->total_charges, total_charges.data(), systems, detail) ||
      !upload(owner->stream, owner->coordinates, default_coordinates.data(), 3 * atoms, detail) ||
      !upload(owner->stream, owner->elements, host_d4_tables.elements, host_d4_tables.element_count,
              detail) ||
      !upload(owner->stream, owner->references, host_d4_tables.references,
              host_d4_tables.reference_count, detail) ||
      !upload(owner->stream, owner->reference_c6, host_d4_tables.reference_c6, c6_count, detail) ||
      !upload(owner->stream, owner->charge_elements, host_eeq_tables.charge_elements,
              host_eeq_tables.element_count, detail)) {
    return nullptr;
  }
  error = cudaStreamSynchronize(owner->stream);
  if (error != cudaSuccess) {
    status = cuda_failure(error, "synchronize D4 setup", detail);
    return nullptr;
  }
  status = GENERATIVEQC_STATUS_SUCCESS;
  detail.clear();
  return owner.release();
}

void destroy_d4_cuda_owner(D4CudaOwner* owner) noexcept {
  if (!owner) return;
  DeviceScope scope(owner->device_id);
  if (scope.error() == cudaSuccess) {
    if (owner->stream) cudaStreamSynchronize(owner->stream);
    runtime::resource_cuda_free(owner->charge_elements);
    runtime::resource_cuda_free(owner->reference_c6);
    runtime::resource_cuda_free(owner->references);
    runtime::resource_cuda_free(owner->elements);
    runtime::resource_cuda_free(owner->fixed_workspace);
    runtime::resource_cuda_free(owner->dqdr_workspace);
    runtime::resource_cuda_free(owner->eeq_workspace);
    runtime::resource_cuda_free(owner->next_system);
    runtime::resource_cuda_free(owner->dedq);
    runtime::resource_cuda_free(owner->charges);
    runtime::resource_cuda_free(owner->gradients);
    runtime::resource_cuda_free(owner->energies);
    runtime::resource_cuda_free(owner->fixed_statuses);
    runtime::resource_cuda_free(owner->statuses);
    runtime::resource_cuda_free(owner->fixed_active);
    runtime::resource_cuda_free(owner->want_gradient);
    runtime::resource_cuda_free(owner->active);
    runtime::resource_cuda_free(owner->coordinates);
    runtime::resource_cuda_free(owner->total_charges);
    runtime::resource_cuda_free(owner->atomic_numbers);
    runtime::resource_cuda_free(owner->offsets);
    if (owner->stream) cudaStreamDestroy(owner->stream);
  }
  delete owner;
}

generativeqc_status execute_d4_cuda(D4CudaOwner* owner, const D4Parameters& parameters,
                                    D4EEQProfile profile, std::span<const double> coordinates,
                                    bool coordinates_changed, std::span<const std::uint8_t> active,
                                    std::span<const std::uint8_t> want_gradient,
                                    std::vector<D4Status>& statuses,
                                    std::vector<double>& energy_components,
                                    std::vector<double>& gradients, std::vector<double>& charges,
                                    D4RuntimeCounters& counters, std::string& detail) {
  if (!owner || coordinates.size() != 3 * owner->atoms || active.size() != owner->systems ||
      want_gradient.size() != owner->systems || statuses.size() != owner->systems ||
      energy_components.size() != 2 * owner->systems || gradients.size() != 3 * owner->atoms ||
      charges.size() != owner->atoms) {
    detail = "invalid D4 CUDA replay shape";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  DeviceScope scope(owner->device_id);
  if (scope.error() != cudaSuccess)
    return cuda_failure(scope.error(), "select D4 CUDA replay device", detail);

  // Queued copies borrow caller storage until the stream completes.
  struct FailureDrain {
    cudaStream_t stream;
    bool armed{true};
    ~FailureDrain() {
      if (armed) (void)cudaStreamSynchronize(stream);
    }
  } failure_drain{owner->stream};

  auto copy_h2d = [&](void* destination, const void* source, std::size_t bytes,
                      const char* action) -> generativeqc_status {
    if (!bytes) return GENERATIVEQC_STATUS_SUCCESS;
    const auto error =
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, owner->stream);
    return error == cudaSuccess ? GENERATIVEQC_STATUS_SUCCESS : cuda_failure(error, action, detail);
  };
  generativeqc_status status = GENERATIVEQC_STATUS_SUCCESS;
  if (coordinates_changed) {
    status = copy_h2d(owner->coordinates, coordinates.data(), coordinates.size_bytes(),
                      "upload D4 changed coordinates");
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    counters.coordinate_h2d_bytes += coordinates.size_bytes();
#if defined(GENERATIVEQC_TEST_HOOKS)
    if (fail_next_d4_after_coordinate_upload) {
      fail_next_d4_after_coordinate_upload = false;
      detail = "injected D4 CUDA failure after coordinate upload";
      return GENERATIVEQC_STATUS_CUDA_ERROR;
    }
#endif
  }
  status = copy_h2d(owner->active, active.data(), active.size_bytes(), "upload D4 active mask");
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  status = copy_h2d(owner->want_gradient, want_gradient.data(), want_gradient.size_bytes(),
                    "upload D4 gradient mask");
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;

  for (const auto& clear :
       {std::pair<void*, std::size_t>{owner->energies, energy_components.size() * sizeof(double)},
        {owner->gradients, gradients.size() * sizeof(double)},
        {owner->charges, charges.size() * sizeof(double)},
        {owner->dedq, charges.size() * sizeof(double)},
        {owner->next_system, sizeof(std::uint32_t)}}) {
    const auto error = cudaMemsetAsync(clear.first, 0, clear.second, owner->stream);
    if (error != cudaSuccess) return cuda_failure(error, "clear D4 CUDA publication", detail);
  }

  const EEQTables eeq_tables{owner->elements, owner->charge_elements, kD4TableElementCount};
  eeq_prepare_kernel<<<owner->workers, kEEQThreads, 0, owner->stream>>>(
      owner->systems, owner->offsets, owner->atomic_numbers, owner->total_charges,
      owner->coordinates, owner->active, eeq_tables, owner->next_system, owner->eeq_workspace,
      owner->eeq_workspace_stride, owner->dqdr_workspace, owner->dqdr_stride, owner->statuses,
      owner->fixed_active, owner->charges, owner->energies, owner->gradients);
  ++counters.kernel_launches;
  auto error = cudaPeekAtLastError();
  if (error != cudaSuccess) return cuda_failure(error, "launch D4 EEQ prepare kernel", detail);

  const bool r2scan = profile == D4EEQProfile::r2scan3c;
  const D4Tables d4_tables{D4ReferenceModel::eeq,     owner->elements,      owner->references,
                           owner->reference_c6,       kD4TableElementCount, kD4TableReferenceCount,
                           kD4PackedReferenceC6Count, r2scan ? 2.0 : 3.0,   r2scan ? 1.0 : 2.0};
  const D4CudaBatch batch{owner->systems,     static_cast<std::uint32_t>(owner->atoms),
                          owner->offsets,     owner->atomic_numbers,
                          owner->coordinates, owner->charges,
                          owner->fixed_active};
  const D4CudaResult result{owner->fixed_statuses, owner->energies, owner->gradients, owner->dedq};
  error = launch_d4_fixed_charge_batched_cuda(
      batch, parameters, d4_tables, owner->fixed_workspace,
      d4_cuda_workspace_elements(static_cast<std::size_t>(owner->atoms)), result, owner->stream);
  counters.kernel_launches += parameters.s9 != 0.0 ? 7u : 5u;
  if (error != cudaSuccess)
    return cuda_failure(error, "launch cooperative fixed-charge D4 scheduler", detail);

  const bool gradients_requested = std::any_of(want_gradient.begin(), want_gradient.end(),
                                               [](std::uint8_t value) { return value != 0; });
  error = cudaMemsetAsync(owner->next_system, 0, sizeof(std::uint32_t), owner->stream);
  if (error != cudaSuccess) return cuda_failure(error, "reset D4 EEQ compose queue", detail);
  eeq_compose_kernel<<<owner->workers, kEEQThreads, 0, owner->stream>>>(
      owner->systems, owner->offsets, owner->atomic_numbers, owner->total_charges,
      owner->coordinates, owner->active, owner->want_gradient, owner->fixed_statuses, eeq_tables,
      owner->next_system, owner->eeq_workspace, owner->eeq_workspace_stride, owner->dqdr_workspace,
      owner->dqdr_stride, owner->statuses, owner->energies, owner->gradients, owner->charges,
      owner->dedq);
  ++counters.kernel_launches;
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return cuda_failure(error, "launch D4 EEQ VJP compose kernel", detail);
  (void)gradients_requested;

  auto copy_d2h = [&](void* destination, const void* source, std::size_t bytes,
                      const char* action) -> generativeqc_status {
    if (!bytes) return GENERATIVEQC_STATUS_SUCCESS;
    const auto copy_error =
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, owner->stream);
    return copy_error == cudaSuccess ? GENERATIVEQC_STATUS_SUCCESS
                                     : cuda_failure(copy_error, action, detail);
  };
  status = copy_d2h(statuses.data(), owner->statuses, statuses.size() * sizeof(D4Status),
                    "download D4 statuses");
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  status = copy_d2h(energy_components.data(), owner->energies,
                    energy_components.size() * sizeof(double), "download D4 energy components");
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  status = copy_d2h(charges.data(), owner->charges, charges.size() * sizeof(double),
                    "download D4 EEQ charges");
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  if (gradients_requested) {
    status = copy_d2h(gradients.data(), owner->gradients, gradients.size() * sizeof(double),
                      "download D4 gradients");
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  }
  error = cudaStreamSynchronize(owner->stream);
  if (error != cudaSuccess) return cuda_failure(error, "synchronize D4 CUDA replay", detail);
  failure_drain.armed = false;
  detail.clear();
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // namespace generativeqc::dft::dispersion
