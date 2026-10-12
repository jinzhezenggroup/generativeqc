#pragma once

// Included after an isolated compiler-emitted profile wrapper. This adapter
// owns only transport, complete-source admission and borrowed-stream lifetime;
// XC/AO/Becke, nuclear derivatives and the terminal sum remain emitted
// programs.
#include "cuda_stationary_binding.hpp"

namespace generativeqc_stationary_cuda {
struct NativeOwner {
  Owner* scientific{};
  cudaStream_t integral_stream{};
  bool integral_pending{}, nuclear_ready{};
  uint64_t integral_mask{}, grid_points{};
};

size_t native_numeric_bytes(size_t atoms, size_t aos, size_t primitives,
                            size_t points, size_t tasks, size_t spins,
                            size_t lanes, bool center_cache) {
  return allocation(atoms, aos, primitives, points, tasks, spins, lanes,
                    center_cache);
}

/** Disjoint source slots may be populated on the existing provider stream.
 * A pending stream remains retained until publication, reset or destruction,
 * even if the provider callback subsequently fails. */
void native_drain_integrals(NativeOwner& owner) {
  if (!owner.integral_pending) return;
  auto stream = owner.integral_stream;
  finished(*owner.scientific, stream);
  owner.integral_pending = false;
  owner.integral_stream = nullptr;
}

__global__ void native_source_transport(const double* input, size_t coordinates,
                                        size_t channels, double sign,
                                        double* first, double* second,
                                        int* error) {
  const auto coordinate = blockIdx.x * blockDim.x + threadIdx.x;
  if (coordinate >= coordinates) return;
  first[coordinate] = finite(sign * input[coordinate], error, 0);
  if (channels == 2) {
    const auto value = finite(sign * input[coordinates + coordinate], error, 0);
    if (second)
      second[coordinate] = value;
    else if (value != 0.0)
      atomicExch(error, 1);
  }
}

/** Re-enter the outer guard without flattening typed inner executor failures.
 * A driver fault must never become a seed-related numerical retry. */
void native_require_status(int status, const char* error, size_t error_size,
                           const char* fallback) {
  const auto* detail = error && error_size ? error : fallback;
  if (status == 2) throw DeviceAllocationError(detail);
  if (status == 3) throw std::bad_alloc();
  if (status == 4) throw DeviceRuntimeError(detail);
  if (status) throw std::runtime_error(detail);
}

int native_create(int device, int major, int minor, size_t atoms, size_t aos,
                  size_t primitives, size_t points, size_t tasks,
                  size_t spin_blocks, size_t primitive_work, size_t budget,
                  size_t lanes, size_t threads, void** output, char* error,
                  size_t error_size) {
  if (output) *output = nullptr;
  return guarded(nullptr, error, error_size, [&] {
    if (!output || spin_blocks != stationary_spin_blocks)
      throw std::invalid_argument("native stationary profile spin mismatch");
    auto owner = std::make_unique<NativeOwner>();
    void* scientific = nullptr;
    const auto status =
        stationary_create(device, major, minor, atoms, aos, primitives, points,
                          tasks, spin_blocks, primitive_work, budget, lanes,
                          threads, &scientific, error, error_size);
    native_require_status(status, error, error_size,
                          "stationary owner creation failed");
    owner->scientific = static_cast<Owner*>(scientific);
    *output = owner.release();
  });
}

int native_topology(void* pointer, const double* primitives,
                    const int64_t* ranges, const double* norms,
                    const int64_t* ao_atoms, char* error, size_t error_size) {
  auto* owner = static_cast<NativeOwner*>(pointer);
  if (!owner)
    return guarded(nullptr, error, error_size, [] {
      throw std::invalid_argument("null native stationary owner");
    });
  return stationary_topology(owner->scientific, primitives, ranges, norms,
                             ao_atoms, error, error_size);
}

int native_reset_geometry(void* pointer, const double* centers,
                          double tolerance, char* error, size_t error_size) {
  auto* owner = static_cast<NativeOwner*>(pointer);
  return guarded(owner ? owner->scientific : nullptr, error, error_size, [&] {
    if (!owner) throw std::invalid_argument("null native stationary owner");
    owner->scientific->context.check_device();
    // Reset discards failed sources, but must still fence their storage users.
    // Do not revalidate the discarded sticky numeric status before clearing it.
    if (owner->integral_pending) {
      cuda_check(cudaStreamSynchronize(owner->integral_stream));
      ++owner->scientific->synchronizations;
      owner->integral_pending = false;
      owner->integral_stream = nullptr;
    }
    native_require_status(
        stationary_geometry_reset(owner->scientific, centers, tolerance, error,
                                  error_size),
        error, error_size, "stationary geometry reset failed");
    owner->integral_mask = 0;
    owner->grid_points = 0;
    owner->nuclear_ready = false;
  });
}

int native_consume_integrals(
    void* pointer, const generativeqc::scf::CudaDerivativeSourceView* view,
    char* error, size_t error_size) {
  using namespace generativeqc::scf;
  auto* owner = static_cast<NativeOwner*>(pointer);
  return guarded(owner ? owner->scientific : nullptr, error, error_size, [&] {
    if (!owner || !view)
      throw std::invalid_argument("null native stationary source lease");
    auto& scientific = *owner->scientific;
    check(scientific);
    if (!view->values || view->device != scientific.context.device ||
        view->coordinates != 3 * scientific.atoms || !view->producer_stream)
      throw std::invalid_argument(
          "native stationary source device/extent mismatch");
    const auto stream = reinterpret_cast<cudaStream_t>(view->producer_stream);
    cudaStreamCaptureStatus capture{};
    cuda_check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument(
          "native stationary sources require a noncapturing stream");
    if (owner->integral_pending && owner->integral_stream != stream)
      throw std::invalid_argument(
          "native stationary integral producer changed before drain");
    size_t slot = 0;
    uint64_t mask = 0;
    double* second = nullptr;
    if (view->kind == CudaDerivativeSourceKind::OneElectron) {
      slot = native_one_electron_slot;
    } else if (view->kind == CudaDerivativeSourceKind::OverlapPulay) {
      slot = native_overlap_slot;
    } else if (view->kind == CudaDerivativeSourceKind::FullRangeJk) {
      slot = native_coulomb_slot;
      if (view->channels != 2 ||
          view->convention != CudaDerivativeConvention::Force)
        throw std::invalid_argument(
            "native stationary J/K source convention mismatch");
      if constexpr (native_exchange_slot >= 0) {
        mask |= native_exchange_mask;
        second = scientific.sources +
                 native_exchange_storage_slot * view->coordinates;
      }
    } else {
      throw std::invalid_argument("unknown native stationary source kind");
    }
    if (view->kind != CudaDerivativeSourceKind::FullRangeJk &&
        (view->channels != 1 ||
         view->convention != CudaDerivativeConvention::EnergyGradient))
      throw std::invalid_argument(
          "native stationary one-electron source convention mismatch");
    mask |= uint64_t{1} << slot;
    if (owner->integral_mask & mask)
      throw std::invalid_argument("duplicate native stationary source");
    owner->integral_stream = stream;
    owner->integral_pending = true;
    const auto sign =
        view->convention == CudaDerivativeConvention::Force ? -1.0 : 1.0;
    native_source_transport<<<blocks(view->coordinates, 64), 64, 0, stream>>>(
        view->values, view->coordinates, view->channels, sign,
        scientific.sources + slot * view->coordinates, second,
        scientific.context.error);
    cuda_check(cudaGetLastError());
    ++scientific.launches;
    owner->integral_mask |= mask;
  });
}

int native_nuclear(void* pointer, const double* charges, size_t count,
                   char* error, size_t error_size) {
  auto* owner = static_cast<NativeOwner*>(pointer);
  return guarded(owner ? owner->scientific : nullptr, error, error_size, [&] {
    if (!owner || owner->nuclear_ready)
      throw std::invalid_argument(
          "null owner or duplicate native nuclear source");
    owner->scientific->context.check_device();
    native_drain_integrals(*owner);
    native_require_status(
        stationary_nuclear_all(owner->scientific, native_nuclear_kind, charges,
                               count, error, error_size),
        error, error_size, "stationary nuclear source failed");
    owner->nuclear_ready = true;
  });
}

int native_geometry(void* pointer, const generativeqc::dft::GridTaskView* view,
                    const double* work, size_t owner_offset,
                    size_t points_per_atom, const double* weights,
                    const double* raw, uint64_t generation,
                    uint64_t binding_flags, char* error, size_t error_size) {
  auto* owner = static_cast<NativeOwner*>(pointer);
  return guarded(owner ? owner->scientific : nullptr, error, error_size, [&] {
    if (!owner || !view || owner_offset != owner->grid_points ||
        !view->npoint ||
        view->npoint >
            std::numeric_limits<uint64_t>::max() - owner->grid_points)
      throw std::invalid_argument(
          "native stationary grid tiles are not an ordered complete span");
    native_require_status(
        stationary_geometry_molecular_resident_weights_enqueue_v2(
            owner->scientific, view, work, owner_offset, points_per_atom,
            weights, raw, generation, binding_flags, error, error_size),
        error, error_size, "stationary geometry source failed");
    owner->grid_points += view->npoint;
  });
}

int native_finish_gradient(void* pointer, size_t expected_points,
                           double* output, size_t count, char* error,
                           size_t error_size) {
  auto* owner = static_cast<NativeOwner*>(pointer);
  return guarded(owner ? owner->scientific : nullptr, error, error_size, [&] {
    if (!owner || !expected_points || owner->grid_points != expected_points ||
        owner->integral_mask != native_required_integral_mask ||
        !owner->nuclear_ready)
      throw std::invalid_argument(
          "native stationary force source inventory is incomplete");
    owner->scientific->context.check_device();
    native_drain_integrals(*owner);
    native_require_status(stationary_finish_reduced(owner->scientific, output,
                                                    count, error, error_size),
                          error, error_size, "stationary reduction failed");
  });
}

/** Diagnostic spans do not authorize complete-force publication. */
int native_source_span(void* pointer, size_t begin, size_t sources,
                       double* output, size_t count, char* error,
                       size_t error_size) {
  auto* owner = static_cast<NativeOwner*>(pointer);
  return guarded(owner ? owner->scientific : nullptr, error, error_size, [&] {
    if (!owner) throw std::invalid_argument("null native stationary owner");
    owner->scientific->context.check_device();
    native_drain_integrals(*owner);
    native_require_status(
        stationary_finish_span(owner->scientific, begin, sources, output, count,
                               error, error_size),
        error, error_size, "stationary source span failed");
  });
}

int native_metrics(void* pointer, uint64_t* output, size_t count) {
  auto* owner = static_cast<NativeOwner*>(pointer);
  return owner ? stationary_metrics(owner->scientific, output, count) : 1;
}

void native_destroy(void* pointer) {
  auto* owner = static_cast<NativeOwner*>(pointer);
  if (!owner) return;
  int previous = 0;
  const auto have_device = cudaGetDevice(&previous) == cudaSuccess;
  if (cudaSetDevice(owner->scientific->context.device) == cudaSuccess &&
      owner->integral_pending)
    (void)cudaStreamSynchronize(owner->integral_stream);
  stationary_destroy(owner->scientific);
  if (have_device) (void)cudaSetDevice(previous);
  delete owner;
}
}  // namespace generativeqc_stationary_cuda
