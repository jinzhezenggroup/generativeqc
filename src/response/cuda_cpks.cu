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
  decltype(&maps::launch_rks_reference_weights) reference_weights;
  decltype(&maps::launch_rks_metric) metric;
  decltype(&maps::launch_rks_rotation) rotation;
  decltype(&maps::launch_rks_action) action;
  decltype(&maps::launch_rks_rhs) rhs;
  decltype(&maps::launch_rks_density) density;
  decltype(&maps::launch_rks_weights) weights;
  std::array<cudaError_t (*)(maps::Extents, std::size_t&), 7> workspaces;
};

const MapDispatch& dispatch(unsigned spins) noexcept {
  static const MapDispatch rks{
      maps::launch_rks_reference_weights,
      maps::launch_rks_metric,
      maps::launch_rks_rotation,
      maps::launch_rks_action,
      maps::launch_rks_rhs,
      maps::launch_rks_density,
      maps::launch_rks_weights,
      {maps::workspace_rks_metric, maps::workspace_rks_rotation, maps::workspace_rks_action,
       maps::workspace_rks_rhs, maps::workspace_rks_density, maps::workspace_rks_weights,
       maps::workspace_rks_reference_weights}};
  static const MapDispatch uks{
      maps::launch_uks_reference_weights,
      maps::launch_uks_metric,
      maps::launch_uks_rotation,
      maps::launch_uks_action,
      maps::launch_uks_rhs,
      maps::launch_uks_density,
      maps::launch_uks_weights,
      {maps::workspace_uks_metric, maps::workspace_uks_rotation, maps::workspace_uks_action,
       maps::workspace_uks_rhs, maps::workspace_uks_density, maps::workspace_uks_weights,
       maps::workspace_uks_reference_weights}};
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

/** Keep the physical check's failure category even if reconstruction later
 * updates the sticky status. Both belong to the same publication transaction. */
struct SolveAudit {
  double physical_residual_norm;
  int status;
  int physical_failure;
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
      (options.resident_precision != CudaCpksResidentPrecision::fp64 &&
       options.resident_precision != CudaCpksResidentPrecision::fp32_proposal) ||
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
  static_assert(sizeof(SolveAudit) % sizeof(double) == 0);
  if (!reserve(sizeof(SolveAudit) / sizeof(double), plan.status)) return false;
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

  SolveAudit* audit() const noexcept { return reinterpret_cast<SolveAudit*>(base + plan.status); }
  int* status() const noexcept { return &audit()->status; }
  void* scratch() const noexcept { return base + plan.tensor; }
  cudaError_t check(const double* matrices, std::size_t size, cudaStream_t stream) noexcept {
    check_symmetric<<<maps::blocks(size), 128, 0, stream>>>(
        matrices, shape.nao, size, options.symmetry_absolute_tolerance,
        options.symmetry_relative_tolerance, status());
    return cudaGetLastError();
  }
  cudaError_t induced(cudaStream_t stream, bool eager = true) noexcept {
    if (eager) ++result.induced_fock_actions;
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
  auto error = execution.induced(stream, false);
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

__global__ void unit_rotation(double* output, std::size_t dimension, std::size_t column) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < dimension;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output[index] = index == column ? 1.0 : 0.0;
}
template <typename Storage>
__global__ void resident_action(const Storage* matrix, const double* input, double* output,
                                std::size_t dimension) {
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < dimension;
       row += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double value{};
    for (std::size_t column = 0; column < dimension; ++column)
      value += matrix[column * dimension + row] * input[column];
    output[row] = value;
  }
}
__global__ void compress_column(const double* input, float* output, std::size_t dimension) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < dimension;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output[index] = static_cast<float>(input[index]);
}
struct ResidentExecution {
  const void* matrix;
  CudaCpksResidentPrecision precision;
};
cudaError_t apply_resident(void* context, const double* input, double* output,
                           std::size_t dimension, cudaStream_t stream) noexcept {
  auto& resident = *static_cast<ResidentExecution*>(context);
  if (resident.precision == CudaCpksResidentPrecision::fp32_proposal)
    resident_action<<<maps::blocks(dimension), 128, 0, stream>>>(
        static_cast<const float*>(resident.matrix), input, output, dimension);
  else
    resident_action<<<maps::blocks(dimension), 128, 0, stream>>>(
        static_cast<const double*>(resident.matrix), input, output, dimension);
  return cudaGetLastError();
}

std::size_t resident_scalar_bytes(CudaCpksResidentPrecision precision) noexcept {
  return precision == CudaCpksResidentPrecision::fp32_proposal ? sizeof(float) : sizeof(double);
}
__global__ void physical_residual(const double* rhs, const double* gaps, const double* rotation,
                                  std::size_t dimension, double tolerance,
                                  const CudaGmresDeviceReceipt* replay, SolveAudit* audit) {
  if (replay) {
    if (replay->code != static_cast<int>(CudaGmresStatus::converged)) return;
    tolerance = replay->tolerance;
  }
  double value{};
  for (std::size_t index = 0; index < dimension; ++index)
    value = hypot(value, rhs[index] - gaps[index] * rotation[index]);
  audit->physical_residual_norm = value;
  audit->physical_failure = audit->status || !isfinite(value) || value > tolerance;
  if (audit->physical_failure) audit->status = 1;
}

__global__ void propagate_failure(const CudaCpksDenseReceipt* aggregate, int* status) {
  if (aggregate->failure != CudaCpksStatus::running) *status = 1;
}

__global__ void finish_dense(const CudaGmresDeviceReceipt* solver, const SolveAudit* audit,
                             bool cached, CudaCpksDenseReceipt* aggregate, int* column_status,
                             const double* density, const double* weights, std::size_t count,
                             double* out_density, double* out_weights) {
  const bool solved = solver->code == static_cast<int>(CudaGmresStatus::converged);
  const bool accepted = solved && !audit->status;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    out_density[index] = accepted ? density[index] : 0.0;
    out_weights[index] = accepted ? weights[index] : 0.0;
  }
  if (blockIdx.x || threadIdx.x) return;
  aggregate->iterations += solver->iterations;
  if (cached) {
    aggregate->cached_actions += 2 * solver->iterations;
    aggregate->physical_checks += solved;
    if (audit->physical_failure) aggregate->revoke_cache = 1;
  }
  if (aggregate->failure == CudaCpksStatus::running) {
    aggregate->last = *solver;
    if (cached && accepted) aggregate->last.residual_norm = audit->physical_residual_norm;
    if (!accepted)
      aggregate->failure = !solved || audit->physical_failure ? CudaCpksStatus::linear_solve_failed
                                                              : CudaCpksStatus::numerical_failure;
  }
  if (!accepted) *column_status = 1;
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

cudaError_t cuda_cpks_operator_workspace(const CudaCpksShape& shape, const CudaCpksOptions& options,
                                         std::size_t& bytes) noexcept {
  bytes = 0;
  Plan plan;
  if (!plan_workspace(shape, options, plan) ||
      !multiply_size(plan.dimension, plan.dimension, bytes) ||
      !multiply_size(bytes, resident_scalar_bytes(options.resident_precision), bytes)) {
    bytes = 0;
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t cuda_cpks_reference_weights(const CudaCpksShape& shape, const CudaCpksOptions& options,
                                        const CudaCpksReference& reference, double* output,
                                        void* workspace, std::size_t workspace_bytes,
                                        cudaStream_t stream) noexcept {
  Plan plan;
  if (!plan_workspace(shape, options, plan) || workspace_bytes < plan.bytes ||
      !reference.authenticate || reference.stream != stream ||
      (reference.coefficient_layout != CudaCpksLayout::column_major &&
       reference.coefficient_layout != CudaCpksLayout::row_major))
    return cudaErrorInvalidValue;
  std::size_t matrices_bytes{}, energy_bytes{}, end{};
  if (!multiply_size(plan.matrices, sizeof(double), matrices_bytes) ||
      !multiply_size(shape.nao, shape.spins * sizeof(double), energy_bytes))
    return cudaErrorInvalidValue;
  const std::array<std::pair<const void*, std::size_t>, 2> inputs{
      {{reference.coefficients, matrices_bytes}, {reference.orbital_energies, energy_bytes}}};
  const std::array<std::pair<const void*, std::size_t>, 2> writes{
      {{workspace, workspace_bytes}, {output, matrices_bytes}}};
  for (const auto [pointer, bytes] : inputs)
    if (!interval(pointer, bytes, end)) return cudaErrorInvalidValue;
  for (std::size_t index = 0; index < writes.size(); ++index) {
    const auto [pointer, bytes] = writes[index];
    if (!interval(pointer, bytes, end)) return cudaErrorInvalidValue;
    for (const auto [input, input_bytes] : inputs)
      if (overlaps(pointer, bytes, input, input_bytes)) return cudaErrorInvalidValue;
    for (std::size_t other = 0; other < index; ++other)
      if (overlaps(pointer, bytes, writes[other].first, writes[other].second))
        return cudaErrorInvalidValue;
  }
  auto error = reference.authenticate(reference.owner);
  if (error != cudaSuccess) return error;
  cudaStreamCaptureStatus capture{};
  error = cudaStreamIsCapturing(stream, &capture);
  if (error != cudaSuccess) return error;
  if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
  DrainOnFailure drain{stream};
  error = cudaMemsetAsync(workspace, 0, plan.bytes, stream);
  if (error != cudaSuccess) return error;
  CudaCpksResult receipt;
  auto* base = static_cast<double*>(workspace);
  Execution execution{shape, options, plan, dispatch(shape.spins), base, nullptr, nullptr, receipt};
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
    error = execution.maps.reference_weights(
        block.extents, base + block.occupied, base + block.energy_matrix,
        base + plan.weights + spin * plan.matrix, execution.scratch(), plan.tensor_bytes,
        execution.status(), stream);
    if (error != cudaSuccess) return error;
  }
  error = execution.check(base + plan.weights, plan.matrices, stream);
  if (error != cudaSuccess) return error;
  bool failed{};
  error = checkpoint(execution, stream, failed);
  if (error != cudaSuccess) return error;
  if (failed) return cudaErrorInvalidValue;
  error = reference.authenticate(reference.owner);
  if (error != cudaSuccess) return error;
  error = cudaMemcpyAsync(output, base + plan.weights, matrices_bytes, cudaMemcpyDeviceToDevice,
                          stream);
  if (error != cudaSuccess) return error;
  error = cudaStreamSynchronize(stream);
  drain.active = false;
  return error;
}

cudaError_t solve_cuda_cpks(const CudaCpksShape& shape, const CudaCpksOptions& options,
                            const CudaCpksReference& reference, const double* frozen_fock,
                            const double* overlap_direction, CudaCpksInducedFock induced_fock,
                            void* context, double* density_direction,
                            double* weighted_density_direction, void* workspace,
                            std::size_t workspace_bytes, cudaStream_t stream,
                            CudaCpksResult& result, CudaCpksOperatorCache* operator_cache,
                            const int* upstream_status,
                            CudaCpksDenseReplay* dense_replay) noexcept {
  result = {};
  Plan plan;
  if (!plan_workspace(shape, options, plan) || workspace_bytes < plan.bytes ||
      !reference.authenticate || !induced_fock || reference.stream != stream ||
      (dense_replay && (!dense_replay->receipt || !dense_replay->column_status ||
                        options.resident_precision != CudaCpksResidentPrecision::fp64)) ||
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
  if (upstream_status) {
    const auto address = reinterpret_cast<std::uintptr_t>(upstream_status);
    if (address % alignof(int) || address > UINTPTR_MAX - sizeof(int)) return cudaErrorInvalidValue;
    for (auto [pointer, bytes] : inputs)
      if (overlaps(upstream_status, sizeof(int), pointer, bytes)) return cudaErrorInvalidValue;
  }
  for (std::size_t index = 0; index < writes.size(); ++index) {
    const auto [pointer, bytes] = writes[index];
    if (!interval(pointer, bytes, end)) return cudaErrorInvalidValue;
    if (overlaps(pointer, bytes, upstream_status, upstream_status ? sizeof(int) : 0))
      return cudaErrorInvalidValue;
    for (auto [input, input_bytes] : inputs)
      if (overlaps(pointer, bytes, input, input_bytes)) return cudaErrorInvalidValue;
    for (std::size_t other = 0; other < index; ++other)
      if (overlaps(pointer, bytes, writes[other].first, writes[other].second))
        return cudaErrorInvalidValue;
  }
  const bool cached = operator_cache && plan.dimension;
  const bool mixed =
      cached && options.resident_precision == CudaCpksResidentPrecision::fp32_proposal;
  if (cached) {
    auto& cache = *operator_cache;
    std::size_t required{};
    if (cache.precision_ != options.resident_precision ||
        !multiply_size(plan.dimension, plan.dimension, required) ||
        !multiply_size(required, resident_scalar_bytes(cache.precision_), required) ||
        cache.bytes_ < required || !interval(cache.values_, required, end) ||
        overlaps(cache.values_, required, upstream_status, upstream_status ? sizeof(int) : 0))
      return cudaErrorInvalidValue;
    for (auto [pointer, bytes] : inputs)
      if (overlaps(cache.values_, required, pointer, bytes)) return cudaErrorInvalidValue;
    for (auto [pointer, bytes] : writes)
      if (overlaps(cache.values_, required, pointer, bytes)) return cudaErrorInvalidValue;
    if (cache.ready_ &&
        (cache.dimension_ != plan.dimension || cache.shape_.nao != shape.nao ||
         cache.shape_.spins != shape.spins || cache.shape_.occupied[0] != shape.occupied[0] ||
         cache.shape_.occupied[1] != shape.occupied[1] || cache.provider_ != induced_fock ||
         cache.context_ != context || cache.reference_.coefficients != reference.coefficients ||
         cache.reference_.orbital_energies != reference.orbital_energies ||
         cache.reference_.coefficient_layout != reference.coefficient_layout ||
         cache.reference_.stream != reference.stream ||
         cache.reference_.authenticate != reference.authenticate ||
         cache.reference_.owner != reference.owner))
      return cudaErrorInvalidResourceHandle;
  }
  result.workspace_bytes = plan.bytes;
  auto error = reference.authenticate(reference.owner);
  if (error != cudaSuccess) return error;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  error = cudaStreamIsCapturing(stream, &capture);
  if (error != cudaSuccess) return error;
  if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
  if (upstream_status) {
    cudaPointerAttributes attributes{};
    int device{};
    if ((error = cudaPointerGetAttributes(&attributes, upstream_status)) != cudaSuccess ||
        (error = cudaGetDevice(&device)) != cudaSuccess)
      return error;
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
      return cudaErrorInvalidValue;
  }
  DrainOnFailure drain{stream};
  error = cudaMemsetAsync(workspace, 0, plan.bytes, stream);
  if (error != cudaSuccess) return error;
  auto* base = static_cast<double*>(workspace);
  Execution execution{shape, options,      plan,    dispatch(shape.spins),
                      base,  induced_fock, context, result};
  if (upstream_status) {
    // Retain the assembly audit until the existing CPKS RHS checkpoint, even
    // when the rejected producer left an otherwise finite private buffer.
    error = cudaMemcpyAsync(execution.status(), upstream_status, sizeof(int),
                            cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) return error;
  }
  if (dense_replay) {
    propagate_failure<<<1, 1, 0, stream>>>(dense_replay->receipt, execution.status());
    if ((error = cudaGetLastError()) != cudaSuccess) return error;
  }
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
  if (!dense_replay || (cached && !operator_cache->ready_)) {
    error = checkpoint(execution, stream, failed);
    if (error != cudaSuccess) return error;
    if (failed) {
      result.status = CudaCpksStatus::numerical_failure;
      return cudaSuccess;
    }
  }
  if (cached && !operator_cache->ready_) {
    auto& cache = *operator_cache;
    for (std::size_t column = 0; column < plan.dimension; ++column) {
      unit_rotation<<<maps::blocks(plan.dimension), 128, 0, stream>>>(base + plan.solution,
                                                                      plan.dimension, column);
      if ((error = cudaGetLastError()) != cudaSuccess) return error;
      // The weighted-density output panel is dead until final assembly and
      // can hold an FP64 column while we encode it into the FP32 lease.
      auto* column64 = mixed ? base + plan.weights
                             : static_cast<double*>(cache.values_) + column * plan.dimension;
      error = apply(&execution, base + plan.solution, column64, plan.dimension, stream);
      if (error != cudaSuccess) return error;
      ++result.induced_fock_actions;
      if (mixed) {
        compress_column<<<maps::blocks(plan.dimension), 128, 0, stream>>>(
            column64, static_cast<float*>(cache.values_) + column * plan.dimension, plan.dimension);
        if ((error = cudaGetLastError()) != cudaSuccess) return error;
      }
      ++cache.columns_;
    }
    error = checkpoint(execution, stream, failed);
    if (error != cudaSuccess) return error;
    if (failed) {
      result.status = CudaCpksStatus::numerical_failure;
      return cudaSuccess;
    }
    if ((error = reference.authenticate(reference.owner)) != cudaSuccess) return error;
    cache.dimension_ = plan.dimension;
    cache.shape_ = shape;
    cache.reference_ = reference;
    cache.provider_ = induced_fock;
    cache.context_ = context;
    cache.ready_ = true;
  }
  ResidentExecution resident{cached ? operator_cache->values_ : nullptr,
                             options.resident_precision};
  auto strict = options.gmres;
  const double* seed = nullptr;
  if (mixed && strict.max_iterations > 1) {
    auto proposal = strict;
    proposal.max_iterations = std::min<std::size_t>(12, strict.max_iterations / 2);
    // A proposal is not a convergence proof. Reserve the rest of the budget
    // for the original operator and measure both phases against the same RHS.
    proposal.relative_tolerance = std::max(proposal.relative_tolerance, 1e-6);
    CudaGmresResult suggested;
    error = solve_cuda_gmres(plan.dimension, proposal, apply_resident, &resident, base + plan.rhs,
                             base + plan.diagonal, base + plan.solution, base + plan.gmres,
                             plan.gmres_bytes, stream, suggested);
    if (error != cudaSuccess) return error;
    operator_cache->actions_ += suggested.operator_actions;
    result.proposal_iterations = suggested.iterations;
    result.proposal_operator_actions = suggested.operator_actions;
    strict.max_iterations -= suggested.iterations;
    if (suggested.converged() && suggested.iterations) {
      seed = base + plan.solution;
      result.proposal_seeded = true;
    }
  }
  const bool use_resident = cached && !mixed;
  const auto gmres_action = use_resident ? apply_resident : apply;
  void* gmres_context = use_resident ? static_cast<void*>(&resident) : &execution;
  const CudaGmresDeviceReceipt* solver_receipt{};
  if (dense_replay) {
    auto& solver = dense_replay->solver;
    if (!solver.receipt()) {
      error = solver.prepare(plan.dimension, strict, gmres_action, gmres_context, base + plan.rhs,
                             base + plan.diagonal, base + plan.solution, base + plan.gmres,
                             plan.gmres_bytes, stream);
      if (error != cudaSuccess) return error;
    }
    error = solver.enqueue(execution.status());
    if (error != cudaSuccess) return error;
    solver_receipt = solver.receipt();
    result.gmres.workspace_bytes = plan.gmres_bytes;
  } else {
    error = solve_cuda_gmres(plan.dimension, strict, gmres_action, gmres_context, base + plan.rhs,
                             base + plan.diagonal, base + plan.solution, base + plan.gmres,
                             plan.gmres_bytes, stream, result.gmres, seed);
    if (error != cudaSuccess) return error;
    if (use_resident)
      operator_cache->actions_ += result.gmres.operator_actions;
    else
      result.induced_fock_actions += result.gmres.operator_actions;
    if (!result.gmres.converged()) {
      result.status = CudaCpksStatus::linear_solve_failed;
      return cudaSuccess;
    }
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
  if (cached) {
    // The final physical Fock already contains both metric and solved rotation
    // motion. Re-evaluate the compiler's stationarity RHS with that Fock, then
    // subtract the gap term. This is the true residual, not a cached matvec,
    // and avoids a second physical Fock evaluation of the rotation alone.
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
    if (!dense_replay) ++operator_cache->checks_;
    physical_residual<<<1, 1, 0, stream>>>(
        base + plan.rhs, base + plan.diagonal, base + plan.solution, plan.dimension,
        result.gmres.tolerance, solver_receipt, execution.audit());
    if ((error = cudaGetLastError()) != cudaSuccess) return error;
  }
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
  if (dense_replay) {
    finish_dense<<<maps::blocks(plan.matrices), 128, 0, stream>>>(
        solver_receipt, execution.audit(), cached, dense_replay->receipt,
        dense_replay->column_status, base + plan.densities, base + plan.weights, plan.matrices,
        density_direction, weighted_density_direction);
    if ((error = cudaGetLastError()) != cudaSuccess) return error;
    drain.active = false;
    return cudaSuccess;
  }
  if (cached) {
    // Reconstruction uses private panels and cannot publish after a failed
    // physical check. Read both gates once, before either output is copied.
    SolveAudit audit{};
    error =
        cudaMemcpyAsync(&audit, execution.audit(), sizeof(audit), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return error;
    if ((error = cudaStreamSynchronize(stream)) != cudaSuccess) return error;
    result.gmres.checkpoint_bytes += sizeof(audit);
    if (audit.physical_failure) {
      operator_cache->ready_ = false;
      result.status = CudaCpksStatus::linear_solve_failed;
      return cudaSuccess;
    }
    result.gmres.residual_norm = audit.physical_residual_norm;
    failed = audit.status != 0;
  } else if ((error = checkpoint(execution, stream, failed)) != cudaSuccess) {
    return error;
  }
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
