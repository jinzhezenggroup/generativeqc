#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "response/cuda_cpks.hpp"
#include "response/ks_cpks_maps.cuh"

namespace generativeqc::response {
namespace {
namespace maps = generativeqc_ks_cpks_maps;

struct MapDispatch {
  decltype(&maps::launch_rks_metric) metric;
  decltype(&maps::launch_rks_rotation) rotation;
  decltype(&maps::launch_rks_action) action;
  decltype(&maps::launch_rks_rhs) rhs;
  decltype(&maps::launch_rks_density) density;
  decltype(&maps::launch_rks_weights) weights;
  std::array<cudaError_t (*)(maps::Extents, std::size_t&), 6> workspaces;
};

const MapDispatch& dispatch(unsigned spins) noexcept {
  static const MapDispatch rks{
      maps::launch_rks_metric,
      maps::launch_rks_rotation,
      maps::launch_rks_action,
      maps::launch_rks_rhs,
      maps::launch_rks_density,
      maps::launch_rks_weights,
      {maps::workspace_rks_metric, maps::workspace_rks_rotation, maps::workspace_rks_action,
       maps::workspace_rks_rhs, maps::workspace_rks_density, maps::workspace_rks_weights}};
  static const MapDispatch uks{
      maps::launch_uks_metric,
      maps::launch_uks_rotation,
      maps::launch_uks_action,
      maps::launch_uks_rhs,
      maps::launch_uks_density,
      maps::launch_uks_weights,
      {maps::workspace_uks_metric, maps::workspace_uks_rotation, maps::workspace_uks_action,
       maps::workspace_uks_rhs, maps::workspace_uks_density, maps::workspace_uks_weights}};
  return spins == 1 ? rks : uks;
}

bool multiply_size(std::size_t left, std::size_t right, std::size_t& product) noexcept {
  if (right && left > std::numeric_limits<std::size_t>::max() / right) return false;
  product = left * right;
  return true;
}

struct SpinPlan {
  maps::Extents extents{};
  std::size_t rotation_offset{};
  std::size_t coefficients{}, occupied{}, virtuals{}, occupied_energies{}, virtual_energies{};
  std::size_t energy_matrix{}, metric_motion{}, orbital_motion{};
};

struct Plan {
  std::size_t matrix{}, matrices{}, dimension{}, bytes{}, tensor_bytes{}, gmres_bytes{};
  std::size_t status{}, densities{}, induced{}, rhs{}, diagonal{}, solution{}, weights{};
  std::size_t tensor{}, gmres{};
  SpinPlan spin[2]{};
};

bool plan_workspace(const CudaCpksShape& shape, const CudaCpksOptions& options,
                    Plan& plan) noexcept {
  if (!shape.nao || (shape.spins != 1 && shape.spins != 2) ||
      (shape.spins == 1 && shape.occupied[1]) ||
      !std::isfinite(options.symmetry_absolute_tolerance) ||
      options.symmetry_absolute_tolerance < 0 ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_relative_tolerance < 0 || options.symmetry_relative_tolerance >= 1 ||
      !multiply_size(shape.nao, shape.nao, plan.matrix) ||
      !multiply_size(plan.matrix, shape.spins, plan.matrices))
    return false;
  std::size_t elements{};
  const auto reserve = [&](std::size_t count, std::size_t& offset) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double) - elements) return false;
    offset = elements;
    elements += count;
    return true;
  };
  if (!reserve(1, plan.status)) return false;
  const auto& selected = dispatch(shape.spins);
  for (unsigned spin = 0; spin < shape.spins; ++spin) {
    const auto occupied = shape.occupied[spin];
    if (occupied > shape.nao) return false;
    const auto virtuals = shape.nao - occupied;
    auto& block = plan.spin[spin];
    block.extents = {shape.nao, occupied, virtuals};
    block.rotation_offset = plan.dimension;
    std::size_t rotation{}, occupied_panel{}, virtual_panel{}, occupied_matrix{};
    if (!multiply_size(occupied, virtuals, rotation) ||
        !multiply_size(shape.nao, occupied, occupied_panel) ||
        !multiply_size(shape.nao, virtuals, virtual_panel) ||
        !multiply_size(occupied, occupied, occupied_matrix) ||
        rotation > std::numeric_limits<std::size_t>::max() - plan.dimension ||
        !reserve(plan.matrix, block.coefficients) || !reserve(occupied_panel, block.occupied) ||
        !reserve(virtual_panel, block.virtuals) || !reserve(occupied, block.occupied_energies) ||
        !reserve(virtuals, block.virtual_energies) ||
        !reserve(occupied_matrix, block.energy_matrix) ||
        !reserve(occupied_panel, block.metric_motion) ||
        !reserve(occupied_panel, block.orbital_motion))
      return false;
    plan.dimension += rotation;
    for (auto query : selected.workspaces) {
      std::size_t required{};
      if (query(block.extents, required) != cudaSuccess) return false;
      plan.tensor_bytes = std::max(plan.tensor_bytes, required);
    }
  }
  if (cuda_gmres_workspace(plan.dimension, options.gmres, plan.gmres_bytes) != cudaSuccess ||
      !reserve(plan.matrices, plan.densities) || !reserve(plan.matrices, plan.induced) ||
      !reserve(plan.matrices, plan.weights) || !reserve(plan.dimension, plan.rhs) ||
      !reserve(plan.dimension, plan.diagonal) || !reserve(plan.dimension, plan.solution) ||
      !reserve(plan.tensor_bytes / sizeof(double), plan.tensor) ||
      plan.gmres_bytes > std::numeric_limits<std::size_t>::max() - sizeof(double) + 1 ||
      !reserve((plan.gmres_bytes + sizeof(double) - 1) / sizeof(double), plan.gmres))
    return false;
  plan.bytes = elements * sizeof(double);
  return plan.bytes <= options.max_workspace_bytes;
}

bool interval(const void* pointer, std::size_t bytes, std::uintptr_t& end) noexcept {
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if ((bytes && (!pointer || begin % alignof(double))) ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
    return false;
  end = begin + bytes;
  return true;
}

bool overlaps(const void* left, std::size_t left_bytes, const void* right,
              std::size_t right_bytes) noexcept {
  return left_bytes && right_bytes &&
         reinterpret_cast<std::uintptr_t>(left) <
             reinterpret_cast<std::uintptr_t>(right) + right_bytes &&
         reinterpret_cast<std::uintptr_t>(right) <
             reinterpret_cast<std::uintptr_t>(left) + left_bytes;
}

__global__ void pack_frame(std::size_t nao, std::size_t nocc, bool column_major,
                           const double* frame, const double* energies, double* coefficients,
                           double* occupied, double* virtuals, double* occupied_energies,
                           double* virtual_energies, double* energy_matrix, int* status) {
  for (std::size_t position = blockIdx.x * blockDim.x + threadIdx.x; position < nao * nao;
       position += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto row = position / nao;
    const auto column = position % nao;
    const double value = frame[column_major ? column * nao + row : position];
    if (!isfinite(value)) atomicExch(status, 1);
    coefficients[position] = value;
    if (column < nocc)
      occupied[row * nocc + column] = value;
    else
      virtuals[row * (nao - nocc) + column - nocc] = value;
    if (row < nocc && column < nocc)
      energy_matrix[row * nocc + column] = row == column ? energies[row] : 0.0;
  }
  for (std::size_t position = blockIdx.x * blockDim.x + threadIdx.x; position < nao;
       position += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const double value = energies[position];
    if (!isfinite(value)) atomicExch(status, 1);
    if (position < nocc)
      occupied_energies[position] = value;
    else
      virtual_energies[position - nocc] = value;
  }
}

__global__ void check_symmetric(const double* matrix, std::size_t nao, std::size_t size,
                                double absolute, double relative, int* status) {
  for (std::size_t position = blockIdx.x * blockDim.x + threadIdx.x; position < size;
       position += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto local = position % (nao * nao);
    const auto transposed = position - local + (local % nao) * nao + local / nao;
    const double left = matrix[position], right = matrix[transposed];
    // Scaling before subtraction avoids overflow in an otherwise valid check.
    const double magnitude = fmax(fabs(left), fabs(right));
    const double scale = fmax(1.0, magnitude);
    if (!isfinite(left) || !isfinite(right) ||
        fabs(left / scale - right / scale) > absolute / scale + relative * (magnitude / scale))
      atomicExch(status, 1);
  }
}

__global__ void poison_failed_action(double* output, std::size_t size, int* status) {
  if (!atomicAdd(status, 0)) return;
  for (std::size_t position = blockIdx.x * blockDim.x + threadIdx.x; position < size;
       position += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output[position] = nan("");
}

struct Execution {
  const CudaCpksShape& shape;
  const CudaCpksOptions& options;
  const Plan& plan;
  const MapDispatch& maps;
  double* base;
  CudaCpksInducedFock provider;
  void* context;
  CudaCpksResult& result;

  int* status() const noexcept { return reinterpret_cast<int*>(base + plan.status); }
  void* scratch() const noexcept { return base + plan.tensor; }
  cudaError_t check(const double* matrices, std::size_t size, cudaStream_t stream) noexcept {
    check_symmetric<<<maps::blocks(size), 128, 0, stream>>>(
        matrices, shape.nao, size, options.symmetry_absolute_tolerance,
        options.symmetry_relative_tolerance, status());
    return cudaGetLastError();
  }
  cudaError_t induced(cudaStream_t stream) noexcept {
    ++result.induced_fock_actions;
    // A provider that omits even one output must fail, not reuse a previous
    // finite Fock element and accidentally pass a false residual gate.
    auto error = cudaMemsetAsync(base + plan.induced, 0xff, plan.matrices * sizeof(double), stream);
    if (error != cudaSuccess) return error;
    error = provider(context, base + plan.densities, base + plan.induced, shape, stream);
    if (error != cudaSuccess) return error;
    return check(base + plan.induced, plan.matrices, stream);
  }
};

cudaError_t apply(void* context, const double* rotation, double* output, std::size_t dimension,
                  cudaStream_t stream) noexcept {
  auto& execution = *static_cast<Execution*>(context);
  const auto& plan = execution.plan;
  if (dimension != plan.dimension) return cudaErrorInvalidValue;
  auto* base = execution.base;
  for (unsigned spin = 0; spin < execution.shape.spins; ++spin) {
    const auto& block = plan.spin[spin];
    auto error = execution.maps.rotation(
        block.extents, base + block.occupied, rotation + block.rotation_offset,
        base + block.virtuals, base + plan.densities + spin * plan.matrix, execution.scratch(),
        plan.tensor_bytes, execution.status(), stream);
    if (error != cudaSuccess) return error;
  }
  auto error = execution.induced(stream);
  if (error != cudaSuccess) return error;
  for (unsigned spin = 0; spin < execution.shape.spins; ++spin) {
    const auto& block = plan.spin[spin];
    error = execution.maps.action(
        block.extents, base + plan.induced + spin * plan.matrix, base + block.occupied,
        base + block.occupied_energies, rotation + block.rotation_offset,
        base + block.virtual_energies, base + block.virtuals, output + block.rotation_offset,
        execution.scratch(), plan.tensor_bytes, execution.status(), stream);
    if (error != cudaSuccess) return error;
  }
  poison_failed_action<<<maps::blocks(dimension), 128, 0, stream>>>(output, dimension,
                                                                    execution.status());
  return cudaGetLastError();
}

cudaError_t checkpoint(Execution& execution, cudaStream_t stream, bool& failed) noexcept {
  int status{};
  auto error =
      cudaMemcpyAsync(&status, execution.status(), sizeof(status), cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess) return error;
  error = cudaStreamSynchronize(stream);
  failed = status != 0;
  return error;
}

struct DrainOnFailure {
  cudaStream_t stream;
  bool active{true};
  ~DrainOnFailure() {
    // A provider may fail after earlier maps were queued. Never return borrowed
    // workspace to its caller while those maps can still reference it.
    if (active) (void)cudaStreamSynchronize(stream);
  }
};
}  // namespace

cudaError_t cuda_cpks_workspace(const CudaCpksShape& shape, const CudaCpksOptions& options,
                                std::size_t& bytes) noexcept {
  bytes = 0;
  Plan plan;
  if (!plan_workspace(shape, options, plan)) return cudaErrorInvalidValue;
  bytes = plan.bytes;
  return cudaSuccess;
}

cudaError_t solve_cuda_cpks(const CudaCpksShape& shape, const CudaCpksOptions& options,
                            const CudaCpksReference& reference, const double* frozen_fock,
                            const double* overlap_direction, CudaCpksInducedFock induced_fock,
                            void* context, double* density_direction,
                            double* weighted_density_direction, void* workspace,
                            std::size_t workspace_bytes, cudaStream_t stream,
                            CudaCpksResult& result) noexcept {
  result = {};
  Plan plan;
  if (!plan_workspace(shape, options, plan) || workspace_bytes < plan.bytes ||
      !reference.authenticate || !induced_fock || reference.stream != stream ||
      (reference.coefficient_layout != CudaCpksLayout::column_major &&
       reference.coefficient_layout != CudaCpksLayout::row_major))
    return cudaErrorInvalidValue;
  std::size_t matrix_bytes{}, matrices_bytes{}, energy_bytes{}, end{};
  if (!multiply_size(plan.matrix, sizeof(double), matrix_bytes) ||
      !multiply_size(plan.matrices, sizeof(double), matrices_bytes) ||
      !multiply_size(shape.nao, shape.spins * sizeof(double), energy_bytes))
    return cudaErrorInvalidValue;
  const std::array<std::pair<const void*, std::size_t>, 4> inputs{
      {{reference.coefficients, matrices_bytes},
       {reference.orbital_energies, energy_bytes},
       {frozen_fock, matrices_bytes},
       {overlap_direction, matrix_bytes}}};
  const std::array<std::pair<const void*, std::size_t>, 3> writes{
      {{workspace, workspace_bytes},
       {density_direction, matrices_bytes},
       {weighted_density_direction, matrices_bytes}}};
  for (auto [pointer, bytes] : inputs)
    if (!interval(pointer, bytes, end)) return cudaErrorInvalidValue;
  for (std::size_t index = 0; index < writes.size(); ++index) {
    const auto [pointer, bytes] = writes[index];
    if (!interval(pointer, bytes, end)) return cudaErrorInvalidValue;
    for (auto [input, input_bytes] : inputs)
      if (overlaps(pointer, bytes, input, input_bytes)) return cudaErrorInvalidValue;
    for (std::size_t other = 0; other < index; ++other)
      if (overlaps(pointer, bytes, writes[other].first, writes[other].second))
        return cudaErrorInvalidValue;
  }
  result.workspace_bytes = plan.bytes;
  auto error = reference.authenticate(reference.owner);
  if (error != cudaSuccess) return error;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  error = cudaStreamIsCapturing(stream, &capture);
  if (error != cudaSuccess) return error;
  if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
  DrainOnFailure drain{stream};
  error = cudaMemsetAsync(workspace, 0, plan.bytes, stream);
  if (error != cudaSuccess) return error;
  auto* base = static_cast<double*>(workspace);
  Execution execution{shape, options,      plan,    dispatch(shape.spins),
                      base,  induced_fock, context, result};
  error = execution.check(frozen_fock, plan.matrices, stream);
  if (error != cudaSuccess) return error;
  error = execution.check(overlap_direction, plan.matrix, stream);
  if (error != cudaSuccess) return error;
  for (unsigned spin = 0; spin < shape.spins; ++spin) {
    const auto& block = plan.spin[spin];
    pack_frame<<<maps::blocks(plan.matrix), 128, 0, stream>>>(
        shape.nao, shape.occupied[spin],
        reference.coefficient_layout == CudaCpksLayout::column_major,
        reference.coefficients + spin * plan.matrix, reference.orbital_energies + spin * shape.nao,
        base + block.coefficients, base + block.occupied, base + block.virtuals,
        base + block.occupied_energies, base + block.virtual_energies, base + block.energy_matrix,
        execution.status());
    if ((error = cudaGetLastError()) != cudaSuccess) return error;
    error = execution.maps.metric(
        block.extents, base + block.coefficients, base + block.occupied,
        base + block.occupied_energies, overlap_direction, base + block.virtual_energies,
        base + plan.densities + spin * plan.matrix, base + plan.diagonal + block.rotation_offset,
        base + block.metric_motion, execution.scratch(), plan.tensor_bytes, execution.status(),
        stream);
    if (error != cudaSuccess) return error;
  }
  error = execution.induced(stream);
  if (error != cudaSuccess) return error;
  for (unsigned spin = 0; spin < shape.spins; ++spin) {
    const auto& block = plan.spin[spin];
    error = execution.maps.rhs(block.extents, frozen_fock + spin * plan.matrix,
                               base + plan.induced + spin * plan.matrix, base + block.occupied,
                               base + block.occupied_energies, overlap_direction,
                               base + block.virtual_energies, base + block.virtuals,
                               base + plan.rhs + block.rotation_offset, execution.scratch(),
                               plan.tensor_bytes, execution.status(), stream);
    if (error != cudaSuccess) return error;
  }
  bool failed{};
  error = checkpoint(execution, stream, failed);
  if (error != cudaSuccess) return error;
  if (failed) {
    result.status = CudaCpksStatus::numerical_failure;
    return cudaSuccess;
  }
  error = solve_cuda_gmres(plan.dimension, options.gmres, apply, &execution, base + plan.rhs,
                           base + plan.diagonal, base + plan.solution, base + plan.gmres,
                           plan.gmres_bytes, stream, result.gmres);
  if (error != cudaSuccess) return error;
  if (!result.gmres.converged()) {
    result.status = CudaCpksStatus::linear_solve_failed;
    return cudaSuccess;
  }
  for (unsigned spin = 0; spin < shape.spins; ++spin) {
    const auto& block = plan.spin[spin];
    error = execution.maps.density(
        block.extents, base + block.metric_motion, base + block.occupied,
        base + plan.solution + block.rotation_offset, base + block.virtuals,
        base + plan.densities + spin * plan.matrix, base + block.orbital_motion,
        execution.scratch(), plan.tensor_bytes, execution.status(), stream);
    if (error != cudaSuccess) return error;
  }
  error = execution.induced(stream);
  if (error != cudaSuccess) return error;
  for (unsigned spin = 0; spin < shape.spins; ++spin) {
    const auto& block = plan.spin[spin];
    error = execution.maps.weights(
        block.extents, frozen_fock + spin * plan.matrix, base + plan.induced + spin * plan.matrix,
        base + block.occupied, base + block.occupied_energies, base + block.energy_matrix,
        base + block.orbital_motion, overlap_direction, base + plan.weights + spin * plan.matrix,
        execution.scratch(), plan.tensor_bytes, execution.status(), stream);
    if (error != cudaSuccess) return error;
  }
  error = execution.check(base + plan.densities, plan.matrices, stream);
  if (error != cudaSuccess) return error;
  error = execution.check(base + plan.weights, plan.matrices, stream);
  if (error != cudaSuccess) return error;
  error = checkpoint(execution, stream, failed);
  if (error != cudaSuccess) return error;
  if (failed) {
    result.status = CudaCpksStatus::numerical_failure;
    return cudaSuccess;
  }
  error = reference.authenticate(reference.owner);
  if (error != cudaSuccess) return error;
  error = cudaMemcpyAsync(density_direction, base + plan.densities, matrices_bytes,
                          cudaMemcpyDeviceToDevice, stream);
  if (error != cudaSuccess) return error;
  error = cudaMemcpyAsync(weighted_density_direction, base + plan.weights, matrices_bytes,
                          cudaMemcpyDeviceToDevice, stream);
  if (error != cudaSuccess) return error;
  error = cudaStreamSynchronize(stream);
  drain.active = false;
  if (error == cudaSuccess) result.status = CudaCpksStatus::converged;
  return error;
}
}  // namespace generativeqc::response
