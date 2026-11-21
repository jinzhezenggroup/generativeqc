#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <limits>
#include <new>

#include "dft/cuda_wb97mv_df_response.hpp"
#include "generated_symmetric_matrix_function.cuh"
#include "response/wb97mv_df_response.cuh"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_wb97mv_df_maps;
// Select generated maps and admission for the accepted inverse or metric branch.
#define METRIC_LEAF(options, name) ((options).spectral_metric ? maps::name##_spectral : maps::name)

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
  std::size_t bytes{}, panels{}, raw{}, metric{}, map_bytes{}, map_offset{}, inverse{}, fit{},
      dressed{}, inverse_direction{}, seed{}, exchange_fraction{}, status{}, reference_terms{},
      action_terms{}, spectral_scratch{}, inverse_seed{}, charge_eigenbasis{}, raw_charge{},
      rotated_charge_motion{}, inverse_motion_charge{}, matrix_reference_terms{},
      matrix_action_terms{}, pairs{}, packed{};
  std::array<std::size_t, 7> offsets{}, counts{};
  bool fit_charge{};
  bool reserve(std::size_t count, std::size_t& offset) {
    offset = bytes;
    std::size_t increment{};
    if (!multiply(count, sizeof(double), increment) || !add(bytes, increment) || !add(bytes, 255))
      return false;
    bytes = bytes / 256 * 256;
    return true;
  }
};
cudaError_t plan(const CudaWb97mvDfShape& shape, const CudaWb97mvDfOptions& options, Plan& result,
                 const char** detail) {
  result = {};
  if (detail) *detail = "invalid fitted response shape or options";
  if (!shape.nao || !shape.naux || (shape.spins != 1 && shape.spins != 2) ||
      !options.max_workspace_bytes || !options.max_affine_term_visits ||
      !std::isfinite(options.symmetry_absolute_tolerance) ||
      options.symmetry_absolute_tolerance < 0 ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_relative_tolerance < 0 ||
      !(options.exchange_fraction >= (options.coulomb ? 0 : -1) &&
        options.exchange_fraction <= 1) ||
      (!options.coulomb && options.exchange_fraction == 0))
    return cudaErrorInvalidValue;
  std::size_t matrix{};
  if (!multiply(shape.nao, shape.nao, matrix) || !multiply(matrix, shape.spins, result.panels) ||
      !multiply(matrix, shape.naux, result.raw) || !multiply(shape.naux, shape.naux, result.metric))
    return cudaErrorInvalidValue;
  // Emission and admission consume the same compiler-reassociated graphs.
  // Materializing an unplanned multioperand graph would bound storage, not work.
  result.pairs = (matrix - shape.nao) / 2 + shape.nao;
  const maps::Extents extents{shape.nao, shape.naux, result.pairs, shape.spins};
  if (detail) *detail = "fitted response affine schedule exceeds its explicit term-visit bound";
  const auto admit = [&](auto workspace, auto work, std::size_t& total) {
    auto& matrix_total = &total == &result.reference_terms ? result.matrix_reference_terms
                                                           : result.matrix_action_terms;
    std::size_t bytes{}, terms{}, matrix_terms{};
    if (workspace(extents, bytes) != cudaSuccess || !work(extents, terms, matrix_terms) ||
        !add(total, terms) || total > options.max_affine_term_visits)
      return false;
    if (!add(matrix_total, matrix_terms) || matrix_total > options.max_matrix_terms) {
      if (detail)
        *detail = "fitted response tiled matrix schedule exceeds its explicit product bound";
      return false;
    }
    result.map_bytes = std::max(result.map_bytes, bytes);
    return true;
  };
  const bool hybrid = options.exchange_fraction != 0;
  result.fit_charge = !hybrid;
  if (!result.fit_charge &&
      !admit(maps::workspace_inverse, maps::work_inverse, result.reference_terms))
    return cudaErrorInvalidValue;
  if (options.spectral_metric && hybrid) {
    // Reference: enter and leave once. Action: enter dM and dG, leave
    // the inverse JVP and the summed moving pullback. Rotations use the
    // compiler work split; budget two divided-difference terms per cubic visit.
    const auto rotation = [&](std::size_t& terms) {
      return admit(maps::workspace_spectral_enter, maps::work_spectral_enter, terms) &&
             admit(maps::workspace_spectral_leave, maps::work_spectral_leave, terms);
    };
    std::size_t cubic{}, mixed{}, pointwise{}, pack_visits{}, pack_stage_bytes{};
    if (!multiply(result.metric, shape.naux, cubic) || !multiply(cubic, 2, mixed) ||
        !multiply(result.metric, 4, pointwise) || !add(mixed, pointwise) ||
        !add(result.action_terms, mixed) || !add(result.reference_terms, result.metric) ||
        !multiply(result.raw, 3, pack_visits) || !add(result.action_terms, pack_visits) ||
        !rotation(result.reference_terms) || !rotation(result.action_terms) ||
        !rotation(result.action_terms) ||
        !admit(maps::workspace_packed_spectral_fit_motion, maps::work_packed_spectral_fit_motion,
               result.action_terms) ||
        !multiply(result.pairs, shape.naux, result.packed) ||
        !multiply(result.packed, 3 * sizeof(double), pack_stage_bytes))
      return cudaErrorInvalidValue;
    std::size_t packed_map_bytes{};
    if (maps::workspace_packed_spectral_fit_motion(extents, packed_map_bytes) != cudaSuccess ||
        !add(pack_stage_bytes, packed_map_bytes))
      return cudaErrorInvalidValue;
    result.map_bytes = std::max(result.map_bytes, pack_stage_bytes);
  } else if (options.spectral_metric) {
    // The charge rule rotates dM once, contracts one ordered bilinear leg,
    // and rotates the summed metric weight back. Charge vector contractions
    // and pointwise difference/seed work separately from the cubic terms.
    std::size_t cubic{}, spectral_reference{}, quadratic{};
    if (!multiply(result.metric, shape.naux, cubic) || !multiply(cubic, 2, spectral_reference) ||
        !multiply(cubic, 7, result.action_terms) || !multiply(result.metric, 8, quadratic) ||
        !add(spectral_reference, quadratic) || !multiply(result.metric, 16, quadratic) ||
        !add(result.action_terms, quadratic) || !add(result.reference_terms, spectral_reference) ||
        result.reference_terms > options.max_affine_term_visits ||
        result.action_terms > options.max_affine_term_visits)
      return cudaErrorInvalidValue;
  }
  const auto selected = [&](auto reference_ws, auto reference_work, auto response_ws,
                            auto response_work) {
    return admit(reference_ws, reference_work, result.reference_terms) &&
           admit(response_ws, response_work, result.action_terms);
  };
  if (result.fit_charge) {
    if (!admit(shape.spins == 1 ? METRIC_LEAF(options, workspace_rks_fit_charge)
                                : METRIC_LEAF(options, workspace_uks_fit_charge),
               shape.spins == 1 ? METRIC_LEAF(options, work_rks_fit_charge)
                                : METRIC_LEAF(options, work_uks_fit_charge),
               result.reference_terms))
      return cudaErrorInvalidValue;
  } else if (!admit(shape.spins == 1 ? maps::workspace_rks_fit_exchange
                                     : maps::workspace_uks_fit_exchange,
                    shape.spins == 1 ? maps::work_rks_fit_exchange : maps::work_uks_fit_exchange,
                    result.reference_terms))
    return cudaErrorInvalidValue;
  const bool supported =
      !options.coulomb
          ? (shape.spins == 1 ? selected(METRIC_LEAF(options, workspace_rks_reference_exchange),
                                         METRIC_LEAF(options, work_rks_reference_exchange),
                                         METRIC_LEAF(options, workspace_rks_response_exchange),
                                         METRIC_LEAF(options, work_rks_response_exchange))
                              : selected(METRIC_LEAF(options, workspace_uks_reference_exchange),
                                         METRIC_LEAF(options, work_uks_reference_exchange),
                                         METRIC_LEAF(options, workspace_uks_response_exchange),
                                         METRIC_LEAF(options, work_uks_response_exchange)))
      : hybrid ? (shape.spins == 1 ? selected(METRIC_LEAF(options, workspace_rks_reference),
                                              METRIC_LEAF(options, work_rks_reference),
                                              METRIC_LEAF(options, workspace_rks_response),
                                              METRIC_LEAF(options, work_rks_response))
                                   : selected(METRIC_LEAF(options, workspace_uks_reference),
                                              METRIC_LEAF(options, work_uks_reference),
                                              METRIC_LEAF(options, workspace_uks_response),
                                              METRIC_LEAF(options, work_uks_response)))
               : (shape.spins == 1 ? selected(METRIC_LEAF(options, workspace_rks_reference_coulomb),
                                              METRIC_LEAF(options, work_rks_reference_coulomb),
                                              METRIC_LEAF(options, workspace_rks_response_coulomb),
                                              METRIC_LEAF(options, work_rks_response_coulomb))
                                   : selected(METRIC_LEAF(options, workspace_uks_reference_coulomb),
                                              METRIC_LEAF(options, work_uks_reference_coulomb),
                                              METRIC_LEAF(options, workspace_uks_response_coulomb),
                                              METRIC_LEAF(options, work_uks_response_coulomb)));
  if (!supported) return cudaErrorInvalidValue;
  if (detail) *detail = "fitted response workspace exceeds its explicit device bound";
  result.counts = {1,          result.panels, result.panels, result.raw, result.metric,
                   result.raw, result.metric};
  for (std::size_t index = 0; index < result.counts.size(); ++index)
    if (!result.reserve(result.counts[index], result.offsets[index])) return cudaErrorInvalidValue;
  std::size_t spectral_scratch{};
  if (options.spectral_metric) {
    if (!multiply(result.metric, 3, spectral_scratch) ||
        !result.reserve(spectral_scratch, result.spectral_scratch))
      return cudaErrorInvalidValue;
    if (hybrid) {
      if (!result.reserve(result.metric, result.inverse_seed)) return cudaErrorInvalidValue;
      // The dense inverse motion is dead after the packed solve. The later
      // inverse cotangent JVP can publish to the same slot.
      result.inverse_direction = result.inverse_seed;
    } else if (!result.reserve(shape.naux, result.charge_eigenbasis) ||
               !result.reserve(shape.naux, result.raw_charge) ||
               !result.reserve(shape.naux, result.rotated_charge_motion) ||
               !result.reserve(shape.naux, result.inverse_motion_charge)) {
      return cudaErrorInvalidValue;
    }
  }
  std::size_t dressed{};
  if ((hybrid && !multiply(result.raw, shape.spins, dressed)) ||
      !result.reserve(result.fit_charge ? shape.naux : result.raw, result.fit) ||
      !result.reserve(dressed, result.dressed) ||
      !result.reserve(result.map_bytes / sizeof(double), result.map_offset) ||
      !result.reserve(result.fit_charge ? 0 : result.metric, result.inverse) ||
      !result.reserve(1, result.seed) ||
      !result.reserve(hybrid ? 1 : 0, result.exchange_fraction) ||
      !result.reserve(1, result.status) || result.bytes > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  if (detail) *detail = nullptr;
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
// The emitted maps use 1. Preserve the first failing check and never inspect
// a staged output that a failed map may have left unpublished.
enum class Check : int {
  spectral_domain = 2,
  density,
  three_center,
  metric,
  fock_output,
  three_center_output,
  metric_output,
  density_direction,
  three_center_direction,
};
cudaError_t gate(int* status, cudaStream_t stream, const char** detail = nullptr) {
  int failed{};
  auto error = cudaMemcpyAsync(&failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
  error = drain(error, stream);
  if (error != cudaSuccess || !failed) return error;
  if (detail) {
    switch (static_cast<Check>(failed)) {
      case Check::spectral_domain:
        *detail = "invalid fitted metric spectrum or insufficient metric cutoff separation";
        break;
      case Check::density:
        *detail = "nonfinite or nonsymmetric fitted reference density";
        break;
      case Check::three_center:
        *detail = "nonfinite or nonsymmetric raw three-center source";
        break;
      case Check::fock_output:
        *detail = "nonfinite or nonsymmetric fitted Fock output";
        break;
      case Check::three_center_output:
        *detail = "nonfinite or nonsymmetric fitted three-center cotangent";
        break;
      case Check::metric_output:
        *detail = "nonfinite or nonsymmetric fitted metric cotangent";
        break;
      case Check::density_direction:
        *detail = "nonfinite or nonsymmetric fitted density direction";
        break;
      case Check::three_center_direction:
        *detail = "nonfinite or nonsymmetric fitted raw three-center direction";
        break;
      case Check::metric:
        *detail = "nonfinite or nonsymmetric fitted metric direction";
        break;
      default:
        break;  // Map failures retain their caller's stage description.
    }
  }
  return cudaErrorInvalidValue;
}
__global__ void symmetric(const double* values, std::size_t extent, std::size_t panels,
                          std::size_t channels, double absolute, double relative, Check check,
                          int* status) {
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
      atomicCAS(status, 0, static_cast<int>(check));
  }
}
__global__ void metric_domain(const double* eigenvalues, std::size_t naux, double threshold,
                              bool spectral, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto largest = eigenvalues[naux - 1];
  const auto boundary = fmax(1.1 * threshold, 64 * DBL_EPSILON);
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < naux;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto value = eigenvalues[index];
    if (!isfinite(largest) || largest <= 0 || !isfinite(value) ||
        (index && value < eigenvalues[index - 1])) {
      atomicCAS(status, 0, static_cast<int>(Check::spectral_domain));
    } else if (spectral) {
      const double fraction = value / largest;
      if (fraction < -128 * DBL_EPSILON ||
          fabs(fraction - threshold) <= fmax(0.1 * threshold, 128 * DBL_EPSILON) ||
          eigenvalues[0] > threshold * largest)
        atomicCAS(status, 0, static_cast<int>(Check::spectral_domain));
    } else if (value <= boundary * largest) {
      atomicCAS(status, 0, static_cast<int>(Check::spectral_domain));
    }
  }
}
// Pack off-diagonal pairs by averaging both directions before the inverse
// acts, then restore both copies from the same fitted tangent.
__global__ void pack_ao_pair_motion(std::size_t nao, std::size_t naux, std::size_t count,
                                    const double* raw, const double* motion, double* packed,
                                    double* packed_motion) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto row = index / naux / nao, column = index / naux % nao;
    if (column > row) continue;
    const auto reflected = (column * nao + row) * naux + index % naux;
    const auto pair = (row * (row + 1) / 2 + column) * naux + index % naux;
    packed[pair] = .5 * (raw[index] + raw[reflected]);
    packed_motion[pair] = .5 * (motion[index] + motion[reflected]);
  }
}

__global__ void expand_ao_pair_motion(std::size_t nao, std::size_t naux, std::size_t count,
                                      const double* packed, double* expanded, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto row = index / naux / nao, column = index / naux % nao;
    const auto upper = row > column ? row : column, lower = row > column ? column : row;
    const auto pair = (upper * (upper + 1) / 2 + lower) * naux + index % naux;
    expanded[index] = packed[pair];
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
  maps::Extents extents() const { return {shape.nao, shape.naux, layout.pairs, shape.spins}; }
  enum class SpectralMap { reference, inverse_motion, weight_motion };
  cudaError_t spectral_dense(SpectralMap kind, const double* input, double* output) const {
    auto* scratch = at(layout.spectral_scratch);
    auto* metric_frame = scratch + layout.metric;
    // The ordered leg lives between the two rotation maps, which already
    // reserve more than one metric panel in their shared scratch.
    auto* ordered = at(layout.map_offset);
    auto* reference_frame = scratch + 2 * layout.metric;
    // Only the reference seed persists between actions. The rotated dM is
    // shared by the frozen inverse motion and the later moving pullback.
    auto* entered = kind == SpectralMap::reference        ? reference_frame
                    : kind == SpectralMap::inverse_motion ? metric_frame
                                                          : scratch;
    auto error = maps::launch_spectral_enter(extents(), source.metric_eigenvectors, input, entered,
                                             at(layout.map_offset), layout.map_bytes, status(),
                                             source.stream);
    if (error == cudaSuccess) error = gate(status(), source.stream);
    const bool moving = kind == SpectralMap::weight_motion;
    if (error == cudaSuccess)
      error = tensor::launch_pseudoinverse_eigenbasis_response(
          shape.naux, source.metric_eigenvalues, source.metric_relative_threshold, entered,
          moving ? metric_frame : nullptr, moving ? reference_frame : nullptr, ordered, scratch,
          source.stream);
    if (error == cudaSuccess)
      error = maps::launch_spectral_leave(extents(), source.metric_eigenvectors, scratch, output,
                                          at(layout.map_offset), layout.map_bytes, status(),
                                          source.stream);
    return error;
  }
  cudaError_t spectral_fit_motion(const double* delta_three_center) const {
    auto* packed = at(layout.map_offset);
    auto* packed_motion = packed + layout.packed;
    auto* solved = packed_motion + layout.packed;
    auto* workspace = solved + layout.packed;
    const auto blocks = maps::blocks(layout.raw);
    pack_ao_pair_motion<<<blocks, 128, 0, source.stream>>>(shape.nao, shape.naux, layout.raw,
                                                           source.three_center, delta_three_center,
                                                           packed, packed_motion);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    error = maps::launch_packed_spectral_fit_motion(
        extents(), at(layout.inverse), at(layout.inverse_direction), packed, packed_motion, solved,
        workspace, layout.map_bytes - 3 * layout.packed * sizeof(double), status(), source.stream);
    if (error != cudaSuccess) return error;
    // The generated directional map reads dB to completion before publishing
    // dW_A, so its staged three-center output can borrow this buffer as input.
    expand_ao_pair_motion<<<blocks, 128, 0, source.stream>>>(
        shape.nao, shape.naux, layout.raw, solved, at(layout.offsets[5]), status());
    return cudaGetLastError();
  }
  cudaError_t spectral_charge_pullback(bool moving, double* output) const {
    auto* scratch = at(layout.spectral_scratch);
    if (moving) {
      tensor::launch_matrix_function_rotate_vector(shape.naux, source.metric_eigenvectors, true,
                                                   at(layout.raw_charge),
                                                   at(layout.rotated_charge_motion), source.stream);
      const auto error = cudaGetLastError();
      if (error != cudaSuccess) return error;
    }
    tensor::launch_pseudoinverse_outer_pullback(
        shape.naux, source.metric_eigenvectors, source.metric_eigenvalues,
        source.metric_relative_threshold, at(layout.charge_eigenbasis),
        moving ? at(layout.rotated_charge_motion) : nullptr, scratch + layout.metric, scratch,
        scratch + 2 * layout.metric, output, source.stream);
    return cudaGetLastError();
  }
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
  cudaError_t audit(const double* pointer, Check check, std::size_t extent, std::size_t panels = 1,
                    std::size_t channels = 1) const {
    symmetric<<<maps::blocks(extent * extent * panels * channels), 128, 0, source.stream>>>(
        pointer, extent, panels, channels, options.symmetry_absolute_tolerance,
        options.symmetry_relative_tolerance, check, status());
    return cudaGetLastError();
  }
  cudaError_t audit_outputs(bool reference, const char** detail = nullptr) const {
    auto error =
        audit(at(layout.offsets[reference ? 1 : 2]), Check::fock_output, shape.nao, shape.spins);
    if (error == cudaSuccess)
      error = audit(at(layout.offsets[reference ? 3 : 5]), Check::three_center_output, shape.nao, 1,
                    shape.naux);
    if (error == cudaSuccess)
      error = audit(at(layout.offsets[reference ? 4 : 6]), Check::metric_output, shape.naux);
    return error == cudaSuccess ? gate(status(), source.stream, detail)
                                : drain(error, source.stream);
  }
};

cudaError_t cuda_wb97mv_df_workspace(const CudaWb97mvDfShape& shape,
                                     const CudaWb97mvDfOptions& options, std::size_t& bytes,
                                     const char** detail) noexcept {
  bytes = 0;
  Plan layout;
  const auto error = plan(shape, options, layout, detail);
  if (error == cudaSuccess) bytes = layout.bytes;
  return error;
}
CudaWb97mvDfResponse::CudaWb97mvDfResponse() = default;
CudaWb97mvDfResponse::~CudaWb97mvDfResponse() = default;

cudaError_t CudaWb97mvDfResponse::prepare(const CudaWb97mvDfShape& shape,
                                          const CudaWb97mvDfOptions& options,
                                          const CudaWb97mvDfReference& reference, void* workspace,
                                          std::size_t workspace_bytes,
                                          const char** detail) noexcept {
  impl_.reset();
  try {
    auto state = std::make_unique<Impl>();
    auto error = plan(shape, options, state->layout, detail);
    if (error != cudaSuccess) return error;
    if (detail) *detail = "invalid fitted reference buffers or workspace";
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
                              reference.metric_eigenvalues, reference.metric_eigenvectors};
    const std::size_t counts[] = {layout.panels, layout.raw, layout.metric, shape.naux,
                                  layout.metric};
    const auto input_count = options.spectral_metric ? 5u : 4u;
    if ((error = device_buffer(workspace, reference.device_id)) != cudaSuccess) return error;
    for (unsigned index = 0; index < input_count; ++index) {
      if (!disjoint(workspace, layout.bytes, inputs[index], counts[index] * sizeof(double)))
        return cudaErrorInvalidValue;
      if ((error = device_buffer(inputs[index], reference.device_id)) != cudaSuccess) return error;
    }
    auto* status = state->status();
    error = cudaMemsetAsync(status, 0, sizeof(int), reference.stream);
    if (error == cudaSuccess) {
      metric_domain<<<maps::blocks(shape.naux), 128, 0, reference.stream>>>(
          reference.metric_eigenvalues, shape.naux, reference.metric_relative_threshold,
          options.spectral_metric, status);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess)
      error = state->audit(reference.density, Check::density, shape.nao, shape.spins);
    if (error == cudaSuccess)
      error = state->audit(reference.three_center, Check::three_center, shape.nao, 1, shape.naux);
    if (error == cudaSuccess) error = gate(status, reference.stream, detail);
    if (error != cudaSuccess) return drain(error, reference.stream);
    if (detail) *detail = "failed to construct the fitted inverse from SCF whitening";
    const double seed = 1;
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(state->at(layout.seed), &seed, sizeof(seed), cudaMemcpyHostToDevice,
                              reference.stream);
    if (error == cudaSuccess && options.exchange_fraction)
      error =
          cudaMemcpyAsync(state->at(layout.exchange_fraction), &state->options.exchange_fraction,
                          sizeof(double), cudaMemcpyHostToDevice, reference.stream);
    if (error == cudaSuccess && !layout.fit_charge) {
      error = maps::launch_inverse(state->extents(), reference.whitening, state->at(layout.inverse),
                                   state->at(layout.map_offset), layout.map_bytes, status,
                                   reference.stream);
      // A failed map leaves its staged output unpublished.
      if (error == cudaSuccess) error = gate(status, reference.stream);
    }
    if (error == cudaSuccess) {
      if (detail) *detail = "failed to prepare the retained fitted exchange or charge legs";
      const auto* inverse = state->at(layout.inverse);
      auto* fit = state->at(layout.fit);
      auto* scratch = state->at(layout.map_offset);
      // The accepted geometry and reference density remain immutable for the
      // lease. Each action differentiates these legs without refitting them.
      if (layout.fit_charge && options.spectral_metric) {
        const auto prepare = shape.spins == 1 ? maps::launch_rks_fit_charge_spectral
                                              : maps::launch_uks_fit_charge_spectral;
        error = prepare(state->extents(), reference.density, reference.three_center,
                        reference.whitening, fit, state->at(layout.raw_charge), scratch,
                        layout.map_bytes, status, reference.stream);
        if (error == cudaSuccess) error = gate(status, reference.stream, detail);
        if (error == cudaSuccess) {
          tensor::launch_matrix_function_rotate_vector(
              shape.naux, reference.metric_eigenvectors, true, state->at(layout.raw_charge),
              state->at(layout.charge_eigenbasis), reference.stream);
          error = cudaGetLastError();
        }
      } else if (layout.fit_charge) {
        const auto prepare =
            shape.spins == 1 ? maps::launch_rks_fit_charge : maps::launch_uks_fit_charge;
        error =
            prepare(state->extents(), reference.density, reference.three_center,
                    reference.whitening, fit, scratch, layout.map_bytes, status, reference.stream);
      } else {
        const auto prepare =
            shape.spins == 1 ? maps::launch_rks_fit_exchange : maps::launch_uks_fit_exchange;
        error =
            prepare(state->extents(), reference.density, inverse, reference.three_center, fit,
                    state->at(layout.dressed), scratch, layout.map_bytes, status, reference.stream);
      }
      if (error == cudaSuccess) error = gate(status, reference.stream, detail);
    }
    if (error == cudaSuccess) {
      if (detail) *detail = "failed to evaluate fitted reference Fock and source cotangents";
      auto output = [&](auto launch, auto... inputs) {
        const auto metric_weight =
            options.spectral_metric ? layout.inverse_seed : layout.offsets[4];
        return launch(state->extents(), inputs..., reference.three_center,
                      state->at(layout.offsets[0]), state->at(layout.offsets[1]),
                      state->at(metric_weight), state->at(layout.offsets[3]),
                      state->at(layout.map_offset), layout.map_bytes, status, reference.stream);
      };
      const auto* seed_value = state->at(layout.seed);
      if (!options.coulomb)
        error = output(shape.spins == 1 ? METRIC_LEAF(options, launch_rks_reference_exchange)
                                        : METRIC_LEAF(options, launch_uks_reference_exchange),
                       state->at(layout.dressed), seed_value, seed_value, reference.density,
                       state->at(layout.exchange_fraction));
      else if (options.exchange_fraction)
        error = output(shape.spins == 1 ? METRIC_LEAF(options, launch_rks_reference)
                                        : METRIC_LEAF(options, launch_uks_reference),
                       state->at(layout.dressed), state->at(layout.fit), seed_value, seed_value,
                       seed_value, reference.density, state->at(layout.exchange_fraction));
      else if (options.spectral_metric) {
        const auto launch = shape.spins == 1 ? maps::launch_rks_reference_coulomb_spectral
                                             : maps::launch_uks_reference_coulomb_spectral;
        error = launch(state->extents(), state->at(layout.fit), seed_value, seed_value,
                       reference.density, reference.three_center, state->at(layout.offsets[0]),
                       state->at(layout.offsets[1]), state->at(layout.offsets[3]),
                       state->at(layout.map_offset), layout.map_bytes, status, reference.stream);
      } else
        error = output(shape.spins == 1 ? maps::launch_rks_reference_coulomb
                                        : maps::launch_uks_reference_coulomb,
                       state->at(layout.fit), seed_value, seed_value, reference.density);
    }
    if (error == cudaSuccess && options.spectral_metric) {
      error = gate(status, reference.stream);
      if (error == cudaSuccess)
        error = layout.fit_charge
                    ? state->spectral_charge_pullback(false, state->at(layout.offsets[4]))
                    : state->spectral_dense(Impl::SpectralMap::reference,
                                            state->at(layout.inverse_seed),
                                            state->at(layout.offsets[4]));
    }
    if (error == cudaSuccess) error = state->audit_outputs(true, detail);
    error = drain(error, reference.stream);
    if (error != cudaSuccess) return error;
    if ((error = state->authenticate(reference.stream)) != cudaSuccess) return error;
    state->receipt.workspace_bytes = layout.bytes;
    state->receipt.reference_terms = state->receipt.affine_terms = layout.reference_terms;
    state->receipt.action_terms = layout.action_terms;
    state->receipt.matrix_reference_terms = state->receipt.matrix_terms =
        layout.matrix_reference_terms;
    state->receipt.matrix_action_terms = layout.matrix_action_terms;
    impl_ = std::move(state);
    if (detail) *detail = nullptr;
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
                                        const CudaWb97mvDfOutput& output, cudaStream_t stream,
                                        const char** detail) noexcept {
  if (detail) *detail = "invalid fitted action buffers or owner";
  if (!impl_) return cudaErrorInvalidResourceHandle;
  auto& state = *impl_;
  auto& receipt = state.receipt;
  if (receipt.attempted_actions == UINT64_MAX) return cudaErrorInvalidValue;
  ++receipt.attempted_actions;
  auto error = state.authenticate(stream);
  if (error != cudaSuccess) return error;
  const auto& layout = state.layout;
  const auto& source = state.source;
  const auto& options = state.options;
  const double* inputs[] = {
      source.density, source.three_center, source.whitening, source.metric_eigenvalues,
      delta_density,  delta_three_center,  delta_metric,     source.metric_eigenvectors};
  const std::size_t counts[] = {layout.panels, layout.raw, layout.metric, state.shape.naux,
                                layout.panels, layout.raw, layout.metric, layout.metric};
  const auto input_count = options.spectral_metric ? 8u : 7u;
  double* outputs[] = {output.energy,
                       output.fock,
                       output.fock_direction,
                       output.weight_three_center,
                       output.weight_metric,
                       output.weight_direction_three_center,
                       output.weight_direction_metric};
  for (unsigned index = 0; index < input_count; ++index) {
    if (!disjoint(state.arena, layout.bytes, inputs[index], counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(inputs[index], source.device_id)) != cudaSuccess) return error;
  }
  for (unsigned index = 0; index < 7; ++index) {
    if (!disjoint(state.arena, layout.bytes, outputs[index], layout.counts[index] * sizeof(double)))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(outputs[index], source.device_id)) != cudaSuccess) return error;
    for (unsigned input = 0; input < input_count; ++input)
      if (!disjoint(outputs[index], layout.counts[index] * sizeof(double), inputs[input],
                    counts[input] * sizeof(double)))
        return cudaErrorInvalidValue;
    for (unsigned previous = 0; previous < index; ++previous)
      if (!disjoint(outputs[index], layout.counts[index] * sizeof(double), outputs[previous],
                    layout.counts[previous] * sizeof(double)))
        return cudaErrorInvalidValue;
  }
  if (receipt.affine_terms > UINT64_MAX - layout.action_terms ||
      receipt.matrix_terms > UINT64_MAX - layout.matrix_action_terms)
    return cudaErrorInvalidValue;
  auto* status = state.status();
  error = cudaMemsetAsync(status, 0, sizeof(int), stream);
  if (error == cudaSuccess)
    error =
        state.audit(delta_density, Check::density_direction, state.shape.nao, state.shape.spins);
  if (error == cudaSuccess)
    error = state.audit(delta_three_center, Check::three_center_direction, state.shape.nao, 1,
                        state.shape.naux);
  if (error == cudaSuccess) error = state.audit(delta_metric, Check::metric, state.shape.naux);
  if (error == cudaSuccess) error = gate(status, stream, detail);
  if (error == cudaSuccess) {
    if (detail) *detail = "failed to evaluate fitted inverse direction";
    receipt.affine_terms += layout.action_terms;
    receipt.matrix_terms += layout.matrix_action_terms;
    if (options.spectral_metric && layout.fit_charge) {
      auto* scratch = state.at(layout.spectral_scratch);
      tensor::launch_pseudoinverse_vector_jvp(
          state.shape.naux, source.metric_eigenvectors, source.metric_eigenvalues,
          source.metric_relative_threshold, delta_metric, state.at(layout.charge_eigenbasis),
          scratch, scratch + layout.metric, state.at(layout.rotated_charge_motion),
          state.at(layout.inverse_motion_charge), stream);
      error = cudaGetLastError();
    } else if (options.spectral_metric) {
      error = state.spectral_dense(Impl::SpectralMap::inverse_motion, delta_metric,
                                   state.at(layout.inverse_direction));
      if (error == cudaSuccess)
        error = state.audit(state.at(layout.inverse_direction), Check::metric, state.shape.naux);
    }
  }
  if (error == cudaSuccess && options.spectral_metric) error = gate(status, stream);
  if (error == cudaSuccess && options.spectral_metric && !layout.fit_charge) {
    if (detail) *detail = "failed to solve the packed spectral fitting direction";
    error = state.spectral_fit_motion(delta_three_center);
    if (error == cudaSuccess) error = gate(status, stream);
  }
  if (error == cudaSuccess) {
    if (detail) *detail = "failed to evaluate fitted Fock and cotangent directions";
    const auto metric_weight = options.spectral_metric ? layout.inverse_seed : layout.offsets[6];
    auto output = [&](auto launch, auto... inputs) {
      return launch(state.extents(), inputs..., state.at(layout.offsets[2]),
                    state.at(metric_weight), state.at(layout.offsets[5]),
                    state.at(layout.map_offset), layout.map_bytes, status, stream);
    };
    const bool rks = state.shape.spins == 1;
    const auto* fit = state.at(layout.fit);
    const auto* dressed = state.at(layout.dressed);
    const auto* seed = state.at(layout.seed);
    const auto* inverse = state.at(layout.inverse);
    if (!options.coulomb) {
      const auto* fraction = state.at(layout.exchange_fraction);
      if (options.spectral_metric)
        error = output(rks ? maps::launch_rks_response_exchange_spectral
                           : maps::launch_uks_response_exchange_spectral,
                       dressed, fit, seed, seed, delta_density, delta_three_center, source.density,
                       fraction, state.at(layout.offsets[5]), source.three_center);
      else
        error =
            output(rks ? maps::launch_rks_response_exchange : maps::launch_uks_response_exchange,
                   dressed, fit, seed, seed, delta_density, delta_metric, delta_three_center,
                   source.density, fraction, inverse);
    } else if (options.exchange_fraction && options.spectral_metric) {
      error = output(rks ? maps::launch_rks_response_spectral : maps::launch_uks_response_spectral,
                     dressed, fit, seed, seed, seed, delta_density, delta_three_center,
                     source.density, state.at(layout.exchange_fraction),
                     state.at(layout.offsets[5]), source.three_center);
    } else if (options.exchange_fraction)
      error =
          output(rks ? maps::launch_rks_response : maps::launch_uks_response, dressed, fit, seed,
                 seed, seed, delta_density, delta_metric, delta_three_center, source.density,
                 state.at(layout.exchange_fraction), inverse, source.three_center);
    else if (!options.spectral_metric)
      error = output(rks ? maps::launch_rks_response_coulomb : maps::launch_uks_response_coulomb,
                     fit, seed, seed, delta_density, delta_metric, delta_three_center,
                     source.density, source.three_center, source.whitening);
    else {
      const auto launch = rks ? maps::launch_rks_response_coulomb_spectral
                              : maps::launch_uks_response_coulomb_spectral;
      error = launch(state.extents(), fit, seed, seed, delta_density, delta_three_center,
                     source.density, state.at(layout.inverse_motion_charge), source.three_center,
                     source.whitening, state.at(layout.offsets[2]), state.at(layout.raw_charge),
                     state.at(layout.offsets[5]), state.at(layout.map_offset), layout.map_bytes,
                     status, stream);
    }
  }
  if (error == cudaSuccess && options.spectral_metric && layout.fit_charge) {
    error = gate(status, stream);
    if (error == cudaSuccess)
      error = state.spectral_charge_pullback(true, state.at(layout.offsets[6]));
  } else if (error == cudaSuccess && options.spectral_metric) {
    // The ambient reference seed was consumed at preparation. Its slot now
    // holds dG; combine the first and second Fréchet terms in the cached frame.
    error = gate(status, stream);
    if (error == cudaSuccess)
      error = state.spectral_dense(Impl::SpectralMap::weight_motion, state.at(layout.inverse_seed),
                                   state.at(layout.offsets[6]));
  }
  if (error == cudaSuccess) error = state.audit_outputs(false, detail);
  if (error != cudaSuccess) return drain(error, stream);
  if ((error = state.authenticate(stream)) != cudaSuccess) return error;
  for (unsigned index = 0; index < 7 && error == cudaSuccess; ++index)
    error =
        cudaMemcpyAsync(outputs[index], state.at(layout.offsets[index]),
                        layout.counts[index] * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  error = drain(error, stream);
  if (error == cudaSuccess) {
    if (detail) *detail = nullptr;
    ++receipt.completed_actions;
  }
  return error;
}
const CudaWb97mvDfWork& CudaWb97mvDfResponse::work() const noexcept {
  static const CudaWb97mvDfWork empty;
  return impl_ ? impl_->receipt : empty;
}
#undef METRIC_LEAF
}  // namespace generativeqc::dft
