#include <array>
#include <charconv>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>

#include "runtime/resource_cuda.cuh"
#include "scf/cuda/df_md_j.cuh"
#include "scf/cuda/df_md_layout.hpp"
#include "scf/cuda/df_md_owner.hpp"
#include "scf/df_md_schedule_policy.hpp"

namespace generativeqc::scf::cuda_execution {

struct CudaDfMdOwner {
  int device{-1};
  cudaStream_t stream{};  // borrowed; enclosing plan destroys this after the owner
  DfMdView pair{}, auxiliary{};
  std::array<std::size_t, 8> pair_offsets{}, auxiliary_offsets{};
  std::size_t pair_hermites{}, auxiliary_hermites{}, pair_responses{}, auxiliary_responses{};
  unsigned charge_splits{}, architecture{};
  std::array<void*, 14> allocations{};
  std::size_t allocation_count{};
  CudaDfMdResources resources{};

  ~CudaDfMdOwner() {
    int previous = -1;
    (void)cudaGetDevice(&previous);
    if (device >= 0 && cudaSetDevice(device) == cudaSuccess) {
      // This also protects pageable setup uploads on every failure path.
      (void)cudaStreamSynchronize(stream);
      for (std::size_t i = allocation_count; i; --i)
        (void)runtime::resource_cuda_free(allocations[i - 1]);
    }
    if (previous >= 0 && previous != device) (void)cudaSetDevice(previous);
  }

  template <class T>
  cudaError_t allocate(std::size_t count, T*& output, const T* host = nullptr) {
    output = nullptr;
    if (!count || count > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
        allocation_count == allocations.size())
      return cudaErrorInvalidValue;
    void* pointer = nullptr;
    auto status = runtime::resource_cuda_malloc(&pointer, count * sizeof(T));
    if (status != cudaSuccess) return status;
    allocations[allocation_count++] = pointer;
    output = static_cast<T*>(pointer);
    if (host)
      status = cudaMemcpyAsync(output, host, count * sizeof(T), cudaMemcpyHostToDevice, stream);
    return status;
  }
};

namespace {
/** Freeze the override at owner creation, including graph replay. An absent
 * override permits the bounded per-class work/capacity policy below. */
unsigned requested_charge_splits() {
  const char* control = std::getenv("GENERATIVEQC_DF_MD_CHARGE_SPLITS");
  if (!control) return 0;
  const std::string_view value(control);
  unsigned splits{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), splits);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !splits ||
      splits > 64)
    throw std::invalid_argument("GENERATIVEQC_DF_MD_CHARGE_SPLITS must be an integer in 1..64");
  return splits;
}

generativeqc_status fail(cudaError_t error, std::string& detail) {
  detail = std::string("DF Hermite owner CUDA operation failed: ") + cudaGetErrorString(error);
  return error == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                            : GENERATIVEQC_STATUS_INTERNAL_ERROR;
}

cudaError_t prepare_view(CudaDfMdOwner& owner, const DfMdLayout& layout, DfMdView& view) {
  view.count = layout.primitives.size();
  view.nbf = layout.aos.size();
  view.auxiliary = layout.auxiliary;
  auto status = owner.allocate(view.count, view.primitives, layout.primitives.data());
  if (status != cudaSuccess) return status;
  DfMdAo* aos = nullptr;
  status = owner.allocate(layout.aos.size(), aos, layout.aos.data());
  if (status != cudaSuccess) return status;
  view.first_aos = view.second_aos = aos;
  if (layout.auxiliary) {
    DfMdAo unit;
    unit.count = 1;
    unit.coefficients[0] = 1;
    status = owner.allocate(1, aos, &unit);
    if (status != cudaSuccess) return status;
    view.first_aos = aos;
    // The one-record local source may die before the later setup drain.
    status = cudaStreamSynchronize(owner.stream);
    if (status != cudaSuccess) return status;
  }
  status = owner.allocate(layout.transform_elements, view.transforms);
  if (status != cudaSuccess) return status;
  status = owner.allocate(layout.hermite_elements, view.density);
  if (status != cudaSuccess) return status;
  status = owner.allocate(layout.hermite_elements, view.potential);
  if (status != cudaSuccess) return status;
  status = owner.allocate(layout.response_elements, view.response_potential);
  if (status != cudaSuccess) return status;
  df_md_prepare_transforms<<<view.count, 128, 0, owner.stream>>>(view);
  return cudaPeekAtLastError();
}

/** No device switch is allowed during replay/capture. A missing or wrong-device
 * owner refuses before submitting a memset or kernel to the borrowed stream. */
cudaError_t validate_replay(const CudaDfMdOwner* owner, const double* input, double* output) {
  if (!owner || !input || !output || input == output) return cudaErrorInvalidValue;
  int device = -1;
  const auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  return device == owner->device ? cudaSuccess : cudaErrorInvalidDevice;
}

template <unsigned P, unsigned A, bool Charge, bool Response>
cudaError_t contract_class(const CudaDfMdOwner& owner) {
  const auto p = owner.pair_offsets[P], np = owner.pair_offsets[P + 1] - p;
  const auto a = owner.auxiliary_offsets[A], na = owner.auxiliary_offsets[A + 1] - a;
  if (!np || !na) return cudaSuccess;
  if constexpr (Charge) {
    constexpr auto components = df_md_count(A + Response);
    // Charge never consumes orbital primal potentials. Projection refreshes
    // them before use, while force keeps its raised pair potentials separate.
    // Reuse this existing allocation only when complete partials fit; there
    // is no added arena, capacity overcommit or allocation during replay.
    const auto capacity = owner.pair_hermites / na / components;
    const auto splits = df_md_charge_splits(owner.charge_splits, owner.architecture, np, capacity);
    if (splits > 1) {
      df_md_contract<P, A, true, Response><<<dim3(na, splits), 128, 0, owner.stream>>>(
          owner.pair, p, np, owner.auxiliary, a, na, owner.pair.potential);
      auto status = cudaPeekAtLastError();
      if (status != cudaSuccess) return status;
      df_md_reduce_charge<A, Response><<<(na * components + 127) / 128, 128, 0, owner.stream>>>(
          owner.auxiliary, a, na, owner.pair.potential, splits);
      return cudaPeekAtLastError();
    }
  }
  df_md_contract<P, A, Charge, Response>
      <<<Charge ? na : np, 128, 0, owner.stream>>>(owner.pair, p, np, owner.auxiliary, a, na);
  return cudaPeekAtLastError();
}

template <bool Charge, bool Response = false>
cudaError_t contract(const CudaDfMdOwner& owner) {
  // Class launches are ordered on one stream; each target Hermite interval has
  // one writer per class, and no class can observe a partially accumulated sum.
#define CLASS(P, A)                                                    \
  {                                                                    \
    const auto status = contract_class<P, A, Charge, Response>(owner); \
    if (status != cudaSuccess) return status;                          \
  }
#define ROW(P) \
  CLASS(P, 0); \
  CLASS(P, 1); \
  CLASS(P, 2); \
  CLASS(P, 3); \
  CLASS(P, 4)
  ROW(0);
  ROW(1);
  ROW(2);
  ROW(3);
  ROW(4);
  ROW(5);
  ROW(6);
#undef ROW
#undef CLASS
  return cudaSuccess;
}
}  // namespace

generativeqc_status create_cuda_df_md_owner(int device, cudaStream_t stream,
                                            const core::System& orbital,
                                            const core::System& auxiliary,
                                            std::size_t device_budget, std::size_t host_budget,
                                            CudaDfMdOwner** owner, CudaDfMdResources& resources,
                                            std::string& detail) {
  resources = {};
  detail.clear();
  if (!owner) {
    detail = "DF Hermite owner output handle is null";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  *owner = nullptr;
  int current = -1;
  auto error = cudaGetDevice(&current);
  if (error != cudaSuccess) return fail(error, detail);
  if (device < 0 || current != device) {
    detail = "DF Hermite owner requires the caller's current device";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  cudaStreamCaptureStatus capture;
  error = cudaStreamIsCapturing(stream, &capture);
  if (error != cudaSuccess) return fail(error, detail);
  if (capture != cudaStreamCaptureStatusNone) {
    detail = "DF Hermite owner preparation is forbidden during stream capture";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    if (orbital.atoms.size() != auxiliary.atoms.size())
      throw std::invalid_argument("DF Hermite orbital and auxiliary geometries differ");
    for (std::size_t i = 0; i < orbital.atoms.size(); ++i)
      if (orbital.atoms[i].atomic_number != auxiliary.atoms[i].atomic_number ||
          orbital.atoms[i].position != auxiliary.atoms[i].position)
        throw std::invalid_argument("DF Hermite orbital and auxiliary geometries differ");
    if (host_budget < sizeof(CudaDfMdOwner))
      throw std::length_error("DF Hermite owner exceeds the host budget");
    const auto pair_layout =
        make_df_md_layout(orbital, false, device_budget, host_budget - sizeof(CudaDfMdOwner));
    // Reserve both complete layouts before their GPU uploads. The auxiliary
    // preflight receives only the unspent host/device capacity; two separately
    // admitted layouts must never transiently exceed the caller's total cap.
    const auto aux_layout =
        make_df_md_layout(auxiliary, true, device_budget - pair_layout.device_bytes,
                          host_budget - sizeof(CudaDfMdOwner) - pair_layout.host_peak_bytes);
    using df_md_layout_detail::add;
    const auto device_bytes = add(pair_layout.device_bytes, aux_layout.device_bytes);
    const auto host_peak =
        add(sizeof(CudaDfMdOwner), add(pair_layout.host_peak_bytes, aux_layout.host_peak_bytes));
    if (device_bytes > device_budget || host_peak > host_budget)
      throw std::length_error("combined DF Hermite owner exceeds the optional budget");
    (void)df_md_layout_detail::multiply(
        df_md_layout_detail::multiply(pair_layout.aos.size(), pair_layout.aos.size()),
        sizeof(double));
    // Layouts outlive the candidate, so failure destruction drains their uploads
    // before any pageable host metadata can be released by stack unwinding.
    auto candidate = std::make_unique<CudaDfMdOwner>();
    candidate->charge_splits = requested_charge_splits();
    candidate->device = device;
    candidate->stream = stream;
    // Device policy is immutable for this owner's lifetime. Never query the
    // environment or device properties while capturing/replaying its graph.
    int major = 0, minor = 0;
    error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (error == cudaSuccess)
      error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (error != cudaSuccess) return fail(error, detail);
    candidate->architecture = static_cast<unsigned>(10 * major + minor);
    candidate->pair_offsets = pair_layout.offsets;
    candidate->auxiliary_offsets = aux_layout.offsets;
    candidate->pair_hermites = pair_layout.hermite_elements;
    candidate->auxiliary_hermites = aux_layout.hermite_elements;
    candidate->pair_responses = pair_layout.response_elements;
    candidate->auxiliary_responses = aux_layout.response_elements;
    error = prepare_view(*candidate, pair_layout, candidate->pair);
    if (error == cudaSuccess) error = prepare_view(*candidate, aux_layout, candidate->auxiliary);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return fail(error, detail);
    candidate->resources = {device_bytes,
                            sizeof(CudaDfMdOwner),
                            host_peak,
                            pair_layout.primitives.size(),
                            aux_layout.primitives.size(),
                            pair_layout.aos.size(),
                            aux_layout.aos.size()};
    resources = candidate->resources;
    *owner = candidate.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for DF Hermite owner";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

void destroy_cuda_df_md_owner(CudaDfMdOwner* owner) noexcept { delete owner; }

cudaError_t cuda_df_md_charge(CudaDfMdOwner* owner, const double* density, double* charge) {
  auto status = validate_replay(owner, density, charge);
  if (status != cudaSuccess) return status;
  df_md_precontract<<<owner->pair.count, 128, 0, owner->stream>>>(owner->pair, density);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(owner->auxiliary.potential, 0,
                           owner->auxiliary_hermites * sizeof(double), owner->stream);
  if (status != cudaSuccess) return status;
  status = contract<true>(*owner);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(charge, 0, owner->resources.naux * sizeof(double), owner->stream);
  if (status != cudaSuccess) return status;
  df_md_project<<<owner->auxiliary.count, 128, 0, owner->stream>>>(owner->auxiliary, charge);
  return cudaPeekAtLastError();
}

cudaError_t cuda_df_md_project(CudaDfMdOwner* owner, const double* potential, double* coulomb) {
  auto status = validate_replay(owner, potential, coulomb);
  if (status != cudaSuccess) return status;
  df_md_precontract<<<owner->auxiliary.count, 128, 0, owner->stream>>>(owner->auxiliary, potential);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(owner->pair.potential, 0, owner->pair_hermites * sizeof(double),
                           owner->stream);
  if (status != cudaSuccess) return status;
  status = contract<false>(*owner);
  if (status != cudaSuccess) return status;
  const auto n = owner->resources.nbf;
  status = cudaMemsetAsync(coulomb, 0, n * n * sizeof(double), owner->stream);
  if (status != cudaSuccess) return status;
  df_md_project<<<owner->pair.count, 128, 0, owner->stream>>>(owner->pair, coulomb);
  return cudaPeekAtLastError();
}

cudaError_t cuda_df_md_three_center_gradient(CudaDfMdOwner* owner, const double* density,
                                             const double* potential, double coefficient,
                                             double* gradient) {
  auto status = validate_replay(owner, density, gradient);
  if (status != cudaSuccess) return status;
  status = validate_replay(owner, potential, gradient);
  if (status != cudaSuccess) return status;
  if (!std::isfinite(coefficient)) return cudaErrorInvalidValue;
  if (coefficient == 0) return cudaSuccess;
  // Both inputs are refreshed: force response must not depend on a prior J or
  // charge submission, even after zero-density or captured-state replay.
  df_md_precontract<<<owner->pair.count, 128, 0, owner->stream>>>(owner->pair, density);
  df_md_precontract<<<owner->auxiliary.count, 128, 0, owner->stream>>>(owner->auxiliary, potential);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(owner->pair.response_potential, 0,
                           owner->pair_responses * sizeof(double), owner->stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(owner->auxiliary.response_potential, 0,
                           owner->auxiliary_responses * sizeof(double), owner->stream);
  if (status != cudaSuccess) return status;
  status = contract<false, true>(*owner);
  if (status != cudaSuccess) return status;
  status = contract<true, true>(*owner);
  if (status != cudaSuccess) return status;
  df_md_project_derivative<true>
      <<<owner->pair.count, 128, 0, owner->stream>>>(owner->pair, density, coefficient, gradient);
  df_md_project_derivative<false>
      <<<owner->pair.count, 128, 0, owner->stream>>>(owner->pair, density, coefficient, gradient);
  df_md_project_derivative<false><<<owner->auxiliary.count, 128, 0, owner->stream>>>(
      owner->auxiliary, potential, coefficient, gradient);
  return cudaPeekAtLastError();
}

}  // namespace generativeqc::scf::cuda_execution
