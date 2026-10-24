#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <limits>
#include <new>

#include "dft/cuda_wb97mv_df_response.hpp"
#include "response/wb97mv_df_response.cuh"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_wb97mv_df_maps;

bool add(std::size_t& total, std::size_t increment) {
  if (increment > SIZE_MAX - total) return false;
  total += increment;
  return true;
}
bool multiply(std::size_t first, std::size_t second, std::size_t& result) {
  if (second && first > SIZE_MAX / second) return false;
  result = first * second;
  return true;
}
struct Plan {
  std::size_t bytes{}, panels{}, raw{}, metric{}, map_bytes{}, map_offset{}, inverse{},
      inverse_direction{}, seed{}, status{}, reference_terms{}, action_terms{};
  std::array<std::size_t, 7> offsets{}, counts{};
  bool reserve(std::size_t count, std::size_t& offset) {
    offset = bytes;
    std::size_t increment{};
    if (!multiply(count, sizeof(double), increment) || !add(bytes, increment) || !add(bytes, 255))
      return false;
    bytes = bytes / 256 * 256;
    return true;
  }
};
cudaError_t plan(const CudaWb97mvDfShape& shape, const CudaWb97mvDfOptions& options, Plan& result) {
  result = {};
  if (!shape.nao || !shape.naux || (shape.spins != 1 && shape.spins != 2) ||
      !options.max_workspace_bytes || !options.max_affine_term_visits ||
      !std::isfinite(options.symmetry_absolute_tolerance) ||
      options.symmetry_absolute_tolerance < 0 ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_relative_tolerance < 0)
    return cudaErrorInvalidValue;
  std::size_t matrix{};
  if (!multiply(shape.nao, shape.nao, matrix) || !multiply(matrix, shape.spins, result.panels) ||
      !multiply(matrix, shape.naux, result.raw) || !multiply(shape.naux, shape.naux, result.metric))
    return cudaErrorInvalidValue;
  // Emission and admission consume the same compiler-reassociated graphs.
  // Materializing an unplanned multioperand graph would bound storage, not work.
  const maps::Extents extents{shape.nao, shape.naux, shape.spins};
  const auto admit = [&](auto workspace, auto work, std::size_t& total) {
    std::size_t bytes{}, terms{};
    if (workspace(extents, bytes) != cudaSuccess || !work(extents, terms) || !add(total, terms) ||
        total > options.max_affine_term_visits)
      return false;
    result.map_bytes = std::max(result.map_bytes, bytes);
    return true;
  };
  if (!admit(maps::workspace_inverse, maps::work_inverse, result.reference_terms) ||
      !admit(maps::workspace_inverse_direction, maps::work_inverse_direction, result.action_terms))
    return cudaErrorInvalidValue;
  if (shape.spins == 1) {
    if (!admit(maps::workspace_rks_reference, maps::work_rks_reference, result.reference_terms) ||
        !admit(maps::workspace_rks_response, maps::work_rks_response, result.action_terms))
      return cudaErrorInvalidValue;
  } else if (!admit(maps::workspace_uks_reference, maps::work_uks_reference,
                    result.reference_terms) ||
             !admit(maps::workspace_uks_response, maps::work_uks_response, result.action_terms))
    return cudaErrorInvalidValue;
  result.counts = {1,          result.panels, result.panels, result.raw, result.metric,
                   result.raw, result.metric};
  for (std::size_t index = 0; index < result.counts.size(); ++index)
    if (!result.reserve(result.counts[index], result.offsets[index])) return cudaErrorInvalidValue;
  if (!result.reserve(result.map_bytes / sizeof(double), result.map_offset) ||
      !result.reserve(result.metric, result.inverse) ||
      !result.reserve(result.metric, result.inverse_direction) || !result.reserve(1, result.seed) ||
      !result.reserve(1, result.status) || result.bytes > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}
bool disjoint(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left >= first_bytes : left - right >= second_bytes;
}
cudaError_t device_buffer(const void* pointer, int device) {
  if (!pointer || reinterpret_cast<std::uintptr_t>(pointer) % alignof(double))
    return cudaErrorInvalidValue;
  cudaPointerAttributes attributes{};
  const auto error = cudaPointerGetAttributes(&attributes, pointer);
  if (error != cudaSuccess) return error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == device
             ? cudaSuccess
             : cudaErrorInvalidDevicePointer;
}
cudaError_t drain(cudaError_t error, cudaStream_t stream) {
  const auto completed = cudaStreamSynchronize(stream);
  return error == cudaSuccess ? completed : error;
}
cudaError_t gate(int* status, cudaStream_t stream) {
  int failed{};
  auto error = cudaMemcpyAsync(&failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
  error = drain(error, stream);
  return error == cudaSuccess && failed ? cudaErrorInvalidValue : error;
}
__global__ void symmetric(const double* values, std::size_t extent, std::size_t panels,
                          std::size_t channels, double absolute, double relative, int* status) {
  if (atomicAdd(status, 0)) return;
  // Raw A has auxiliary channels inside each AO pair; D has outer spin panels.
  const auto matrix = extent * extent;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < panels * matrix * channels;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto pair = index / channels;
    const auto transposed =
        (pair / matrix * matrix + pair % extent * extent + pair % matrix / extent) * channels +
        index % channels;
    const auto first = values[index], second = values[transposed];
    if (!isfinite(first) || !isfinite(second) ||
        fabs(first - second) > absolute + relative * fmax(fabs(first), fabs(second)))
      atomicExch(status, 1);
  }
}
__global__ void metric_domain(const double* eigenvalues, std::size_t naux, double threshold,
                              int* status) {
  if (atomicAdd(status, 0)) return;
  const auto largest = eigenvalues[naux - 1];
  const auto boundary = fmax(1.1 * threshold, 64 * DBL_EPSILON);
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < naux;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto value = eigenvalues[index];
    if (!isfinite(largest) || largest <= 0 || !isfinite(value) || value <= boundary * largest ||
        (index && value < eigenvalues[index - 1]))
      atomicExch(status, 1);
  }
}
}  // namespace

struct CudaWb97mvDfResponse::Impl {
  CudaWb97mvDfShape shape{};
  CudaWb97mvDfOptions options{};
  CudaWb97mvDfReference source{};
  CudaWb97mvDfWork receipt{};
  Plan layout{};
  void* arena{};
  bool ready{};
  double* at(std::size_t offset) const {
    return reinterpret_cast<double*>(static_cast<char*>(arena) + offset);
  }
  int* status() const { return reinterpret_cast<int*>(at(layout.status)); }
  maps::Extents extents() const { return {shape.nao, shape.naux, shape.spins}; }
  cudaError_t authenticate(cudaStream_t stream) const {
    if (!ready || stream != source.stream) return cudaErrorInvalidResourceHandle;
    int device{-1};
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    if (device != source.device_id) return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture{};
    if ((error = cudaStreamIsCapturing(stream, &capture)) != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    return source.authenticate(source.owner);
  }
  cudaError_t audit(const double* pointer, std::size_t extent, std::size_t panels = 1,
                    std::size_t channels = 1) const {
    symmetric<<<maps::blocks(extent * extent * panels * channels), 128, 0, source.stream>>>(
        pointer, extent, panels, channels, options.symmetry_absolute_tolerance,
        options.symmetry_relative_tolerance, status());
    return cudaGetLastError();
  }
  cudaError_t audit_outputs(bool reference) const {
    auto error = audit(at(layout.offsets[reference ? 1 : 2]), shape.nao, shape.spins);
    if (error == cudaSuccess)
      error = audit(at(layout.offsets[reference ? 3 : 5]), shape.nao, 1, shape.naux);
    if (error == cudaSuccess) error = audit(at(layout.offsets[reference ? 4 : 6]), shape.naux);
    return error == cudaSuccess ? gate(status(), source.stream) : drain(error, source.stream);
  }
};

cudaError_t cuda_wb97mv_df_workspace(const CudaWb97mvDfShape& shape,
                                     const CudaWb97mvDfOptions& options,
                                     std::size_t& bytes) noexcept {
  bytes = 0;
  Plan layout;
  const auto error = plan(shape, options, layout);
  if (error == cudaSuccess) bytes = layout.bytes;
  return error;
}
CudaWb97mvDfResponse::CudaWb97mvDfResponse() = default;
CudaWb97mvDfResponse::~CudaWb97mvDfResponse() = default;

cudaError_t CudaWb97mvDfResponse::prepare(const CudaWb97mvDfShape& shape,
                                          const CudaWb97mvDfOptions& options,
                                          const CudaWb97mvDfReference& reference, void* workspace,
                                          std::size_t workspace_bytes) noexcept {
  impl_.reset();
  try {
    auto state = std::make_unique<Impl>();
    auto error = plan(shape, options, state->layout);
    if (error != cudaSuccess) return error;
    state->shape = shape;
    state->options = options;
    state->source = reference;
    state->arena = workspace;
    const auto& layout = state->layout;
    if (!workspace || workspace_bytes < layout.bytes || reference.device_id < 0 ||
        !reference.stream || !reference.authenticate || !reference.owner ||
        !std::isfinite(reference.metric_relative_threshold) ||
        reference.metric_relative_threshold < 0 || reference.metric_relative_threshold >= 1)
      return cudaErrorInvalidValue;
    state->ready = true;
    if ((error = state->authenticate(reference.stream)) != cudaSuccess) return error;
    const double* inputs[] = {reference.density, reference.three_center, reference.whitening,
                              reference.metric_eigenvalues};
    const std::size_t counts[] = {layout.panels, layout.raw, layout.metric, shape.naux};
    if ((error = device_buffer(workspace, reference.device_id)) != cudaSuccess) return error;
    for (unsigned index = 0; index < 4; ++index) {
      if (!disjoint(workspace, layout.bytes, inputs[index], counts[index] * sizeof(double)))
        return cudaErrorInvalidValue;
      if ((error = device_buffer(inputs[index], reference.device_id)) != cudaSuccess) return error;
    }
    auto* status = state->status();
    error = cudaMemsetAsync(status, 0, sizeof(int), reference.stream);
    if (error == cudaSuccess) {
      metric_domain<<<maps::blocks(shape.naux), 128, 0, reference.stream>>>(
          reference.metric_eigenvalues, shape.naux, reference.metric_relative_threshold, status);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = state->audit(reference.density, shape.nao, shape.spins);
    if (error == cudaSuccess)
      error = state->audit(reference.three_center, shape.nao, 1, shape.naux);
    if (error == cudaSuccess) error = gate(status, reference.stream);
    const double seed = 1;
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(state->at(layout.seed), &seed, sizeof(seed), cudaMemcpyHostToDevice,
                              reference.stream);
    if (error == cudaSuccess)
      error = maps::launch_inverse(state->extents(), reference.whitening, state->at(layout.inverse),
                                   state->at(layout.map_offset), layout.map_bytes, status,
                                   reference.stream);
    // A failed map deliberately leaves its staged output unpublished. Do not
    // let the next map's input validation read such an unwritten inverse.
    if (error == cudaSuccess) error = gate(status, reference.stream);
    const auto launch = shape.spins == 1 ? maps::launch_rks_reference : maps::launch_uks_reference;
    if (error == cudaSuccess)
      error =
          launch(state->extents(), state->at(layout.seed), state->at(layout.seed),
                 state->at(layout.seed), reference.density, state->at(layout.inverse),
                 reference.three_center, state->at(layout.offsets[0]), state->at(layout.offsets[1]),
                 state->at(layout.offsets[4]), state->at(layout.offsets[3]),
                 state->at(layout.map_offset), layout.map_bytes, status, reference.stream);
    if (error == cudaSuccess) error = state->audit_outputs(true);
    error = drain(error, reference.stream);
    if (error != cudaSuccess) return error;
    if ((error = state->authenticate(reference.stream)) != cudaSuccess) return error;
    state->receipt = {layout.bytes,          layout.reference_terms, layout.action_terms, 0, 0,
                      layout.reference_terms};
    impl_ = std::move(state);
    return cudaSuccess;
  } catch (const std::bad_alloc&) {
    return cudaErrorMemoryAllocation;
  } catch (...) {
    return cudaErrorUnknown;
  }
}

cudaError_t CudaWb97mvDfResponse::apply(const double* delta_density,
                                        const double* delta_three_center,
                                        const double* delta_metric,
                                        const CudaWb97mvDfOutput& output,
                                        cudaStream_t stream) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  auto& state = *impl_;
  auto& receipt = state.receipt;
  if (receipt.attempted_actions == UINT64_MAX) return cudaErrorInvalidValue;
  ++receipt.attempted_actions;
  auto error = state.authenticate(stream);
  if (error != cudaSuccess) return error;
  const auto& layout = state.layout;
  const auto& source = state.source;
  const double* inputs[] = {
      source.density, source.three_center, source.whitening, source.metric_eigenvalues,
      delta_density,  delta_three_center,  delta_metric};
  const std::size_t counts[] = {layout.panels, layout.raw, layout.metric, state.shape.naux,
                                layout.panels, layout.raw, layout.metric};
  double* outputs[] = {output.energy,
                       output.fock,
                       output.fock_direction,
                       output.weight_three_center,
                       output.weight_metric,
                       output.weight_direction_three_center,
                       output.weight_direction_metric};
  for (unsigned index = 0; index < 7; ++index) {
    if (!disjoint(state.arena, layout.bytes, inputs[index], counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(inputs[index], source.device_id)) != cudaSuccess) return error;
    if (!disjoint(state.arena, layout.bytes, outputs[index], layout.counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(outputs[index], source.device_id)) != cudaSuccess) return error;
    for (unsigned input = 0; input < 7; ++input)
      if (!disjoint(outputs[index], layout.counts[index] * sizeof(double), inputs[input],
                    counts[input] * sizeof(double)))
        return cudaErrorInvalidValue;
    for (unsigned previous = 0; previous < index; ++previous)
      if (!disjoint(outputs[index], layout.counts[index] * sizeof(double), outputs[previous],
                    layout.counts[previous] * sizeof(double)))
        return cudaErrorInvalidValue;
  }
  if (receipt.affine_terms > UINT64_MAX - layout.action_terms) return cudaErrorInvalidValue;
  auto* status = state.status();
  error = cudaMemsetAsync(status, 0, sizeof(int), stream);
  if (error == cudaSuccess) error = state.audit(delta_density, state.shape.nao, state.shape.spins);
  if (error == cudaSuccess)
    error = state.audit(delta_three_center, state.shape.nao, 1, state.shape.naux);
  if (error == cudaSuccess) error = state.audit(delta_metric, state.shape.naux);
  if (error == cudaSuccess) error = gate(status, stream);
  if (error == cudaSuccess) {
    receipt.affine_terms += layout.action_terms;
    error = maps::launch_inverse_direction(
        state.extents(), state.at(layout.inverse), delta_metric, state.at(layout.inverse_direction),
        state.at(layout.map_offset), layout.map_bytes, status, stream);
  }
  if (error == cudaSuccess) error = gate(status, stream);
  const auto launch =
      state.shape.spins == 1 ? maps::launch_rks_response : maps::launch_uks_response;
  if (error == cudaSuccess)
    error = launch(state.extents(), state.at(layout.seed), state.at(layout.seed),
                   state.at(layout.seed), delta_density, state.at(layout.inverse_direction),
                   delta_three_center, source.density, state.at(layout.inverse),
                   source.three_center, state.at(layout.offsets[2]), state.at(layout.offsets[6]),
                   state.at(layout.offsets[5]), state.at(layout.map_offset), layout.map_bytes,
                   status, stream);
  if (error == cudaSuccess) error = state.audit_outputs(false);
  if (error != cudaSuccess) return drain(error, stream);
  if ((error = state.authenticate(stream)) != cudaSuccess) return error;
  for (unsigned index = 0; index < 7 && error == cudaSuccess; ++index)
    error =
        cudaMemcpyAsync(outputs[index], state.at(layout.offsets[index]),
                        layout.counts[index] * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  error = drain(error, stream);
  if (error == cudaSuccess) ++receipt.completed_actions;
  return error;
}
const CudaWb97mvDfWork& CudaWb97mvDfResponse::work() const noexcept {
  static const CudaWb97mvDfWork empty;
  return impl_ ? impl_->receipt : empty;
}
}  // namespace generativeqc::dft
