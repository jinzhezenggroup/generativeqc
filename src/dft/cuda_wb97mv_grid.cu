#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "dft/cuda_wb97mv_grid.hpp"
#include "dft/hessian_aot.hpp"
#include "response/wb97mv_grid_response.cuh"

namespace generativeqc::dft {
namespace {
namespace maps = generativeqc_wb97mv_grid_maps;
namespace ad = generativeqc_wb97mv_grid_ad;

bool product(std::size_t left, std::size_t right, std::size_t& value) noexcept {
  if (right && left > std::numeric_limits<std::size_t>::max() / right) return false;
  value = left * right;
  return true;
}

struct Layout {
  std::size_t bytes{}, matrix{}, matrices{}, fields{}, basis_elements{}, ao_elements{};
  std::size_t status{}, count{}, active{}, ao{}, base{}, direction{}, coefficients{};
  std::size_t reference{}, candidate{}, pairs{}, tensor{}, tensor_bytes{};
  maps::Extents extents{};
};

bool layout(const CudaWb97mvGridShape& shape, const CudaWb97mvGridOptions& options,
            Layout& plan) noexcept {
  if (!shape.atoms || !shape.primitives || !shape.nao || !shape.points ||
      (shape.spins != 1 && shape.spins != 2) || !options.pair_tile_points ||
      options.pair_tile_points > 256 || !options.max_pair_evaluations ||
      !std::isfinite(options.symmetry_absolute_tolerance) ||
      options.symmetry_absolute_tolerance < 0 ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_relative_tolerance < 0 || options.symmetry_relative_tolerance >= 1 ||
      !product(shape.nao, shape.nao, plan.matrix) ||
      !product(plan.matrix, shape.spins, plan.matrices) ||
      !product(shape.points, 5 * shape.spins, plan.fields) ||
      !product(shape.points, shape.nao, plan.ao_elements) ||
      !product(plan.ao_elements, 4, plan.ao_elements))
    return false;
  std::size_t centers{}, primitives{}, records{}, elements{};
  if (!product(shape.atoms, 3, centers) || !product(shape.primitives, 2, primitives) ||
      !product(shape.nao, 16, records) || centers > SIZE_MAX - primitives ||
      centers + primitives > SIZE_MAX - records)
    return false;
  plan.basis_elements = centers + primitives + records;
  const auto reserve = [&](std::size_t count, std::size_t& offset) {
    if (count > SIZE_MAX / sizeof(double) - elements) return false;
    offset = elements;
    elements += count;
    return true;
  };
  static_assert(sizeof(std::size_t) <= sizeof(double));
  plan.extents = {shape.nao, shape.points};
  std::size_t features{}, potential{}, pair_elements{};
  if (maps::workspace_features(plan.extents, features) != cudaSuccess ||
      maps::workspace_potential(plan.extents, potential) != cudaSuccess ||
      !product(options.pair_tile_points, options.pair_tile_points, pair_elements) ||
      !product(pair_elements, 4, pair_elements))
    return false;
  plan.tensor_bytes = std::max(features, potential);
  if (!reserve(1, plan.status) || !reserve(1, plan.count) || !reserve(shape.points, plan.active) ||
      !reserve(plan.ao_elements, plan.ao) || !reserve(plan.fields, plan.base) ||
      !reserve(plan.fields, plan.direction) || !reserve(plan.fields, plan.coefficients) ||
      !reserve(plan.matrices, plan.reference) || !reserve(plan.matrices, plan.candidate) ||
      !reserve(pair_elements, plan.pairs) ||
      !reserve(plan.tensor_bytes / sizeof(double), plan.tensor))
    return false;
  plan.bytes = elements * sizeof(double);
  return plan.bytes <= options.max_workspace_bytes;
}

bool interval(const void* pointer, std::size_t elements, std::uintptr_t& end) noexcept {
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  std::size_t bytes{};
  if (!pointer || begin % alignof(double) || !product(elements, sizeof(double), bytes) ||
      bytes > UINTPTR_MAX - begin)
    return false;
  end = begin + bytes;
  return true;
}

bool disjoint(const void* left, std::size_t first, const void* right, std::size_t second) noexcept {
  std::uintptr_t left_end{}, right_end{};
  return interval(left, first, left_end) && interval(right, second, right_end) &&
         (left_end <= reinterpret_cast<std::uintptr_t>(right) ||
          right_end <= reinterpret_cast<std::uintptr_t>(left));
}

cudaError_t device_buffer(const void* pointer, int device) noexcept {
  cudaPointerAttributes attributes{};
  auto error = cudaPointerGetAttributes(&attributes, pointer);
  if (error != cudaSuccess) return error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == device
             ? cudaSuccess
             : cudaErrorInvalidValue;
}

cudaError_t execution(cudaStream_t stream, int device) noexcept {
  if (!stream || device < 0) return cudaErrorInvalidValue;
  int current = -1;
  auto error = cudaGetDevice(&current);
  if (error != cudaSuccess) return error;
  if (current != device) return cudaErrorInvalidDevice;
  cudaStreamCaptureStatus capture{};
  error = cudaStreamIsCapturing(stream, &capture);
  if (error != cudaSuccess) return error;
  return capture == cudaStreamCaptureStatusNone ? cudaSuccess : cudaErrorStreamCaptureUnsupported;
}

cudaError_t drain(cudaError_t error, cudaStream_t stream) noexcept {
  const auto fence = cudaStreamSynchronize(stream);
  return error == cudaSuccess ? fence : error;
}

cudaError_t numerical_status(int* status, cudaStream_t stream) noexcept {
  int host{};
  auto error = cudaMemcpyAsync(&host, status, sizeof(host), cudaMemcpyDeviceToHost, stream);
  error = drain(error, stream);
  return error != cudaSuccess ? error : host ? cudaErrorInvalidValue : cudaSuccess;
}

__global__ void check_matrix(const double* density, std::size_t nao, unsigned spins,
                             double absolute, double relative, int* status) {
  const auto matrix = nao * nao;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < matrix * spins;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto element = index % matrix;
    const auto other = index / matrix * matrix + element % nao * nao + element / nao;
    const auto value = density[index], transposed = density[other];
    if (!isfinite(value) || !isfinite(transposed) ||
        fabs(value - transposed) > absolute + relative * fmax(fabs(value), fabs(transposed)))
      atomicExch(status, 1);
  }
}

__global__ void discover(const double* fields, const double* weights, const double* points,
                         std::size_t count, unsigned spins, std::size_t* active,
                         std::size_t* selected, int* status) {
  if (blockIdx.x || threadIdx.x) return;
  std::size_t size{};
  for (std::size_t point = 0; point < count; ++point) {
    double total{};
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto rho = fields[5 * spin * count + point];
      const auto tau = fields[(5 * spin + 4) * count + point];
      if (!isfinite(rho) || !isfinite(tau) || rho < 0 || tau < 0) atomicExch(status, 1);
      total += rho;
      for (unsigned axis = 1; axis < 4; ++axis)
        if (!isfinite(fields[(5 * spin + axis) * count + point])) atomicExch(status, 1);
    }
    if (!isfinite(total) || !isfinite(weights[point])) atomicExch(status, 1);
    for (unsigned axis = 0; axis < 3; ++axis)
      if (!isfinite(points[3 * point + axis])) atomicExch(status, 1);
    if (total >= ad::vv10_density_threshold) active[size++] = point;
  }
  *selected = size;
}

__global__ void pair_reduce(const double* values, const std::size_t* active, std::size_t row_begin,
                            std::size_t rows, std::size_t columns, std::size_t points,
                            unsigned spins, double* coefficients, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 4 * rows;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto field = index / rows, row = index % rows;
    double value{};
    for (std::size_t column = 0; column < columns; ++column)
      value += values[(field * rows + row) * columns + column];
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto offset = (5 * spin + field) * points + active[row_begin + row];
      coefficients[offset] += value;
      if (!isfinite(coefficients[offset])) atomicExch(status, 1);
    }
  }
}

__global__ void local_beta(const double* weights, const std::size_t* active, std::size_t count,
                           std::size_t points, unsigned spins, double* coefficients, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto offset = 5 * spin * points + active[index];
      coefficients[offset] += ad::vv10_beta * weights[active[index]];
      if (!isfinite(coefficients[offset])) atomicExch(status, 1);
    }
}

cudaError_t features(const CudaWb97mvGridShape& shape, const Layout& plan, double* arena,
                     const double* density, double* fields, cudaStream_t stream) noexcept {
  const auto panel = shape.points * shape.nao;
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  for (unsigned spin = 0; spin < shape.spins; ++spin) {
    auto* field = fields + 5 * spin * shape.points;
    const auto error = maps::launch_features(
        plan.extents, arena + plan.ao, arena + plan.ao + panel, arena + plan.ao + 2 * panel,
        arena + plan.ao + 3 * panel, density + spin * plan.matrix, field + shape.points,
        field + 2 * shape.points, field + 3 * shape.points, field, field + 4 * shape.points,
        arena + plan.tensor, plan.tensor_bytes, status, stream);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}
}  // namespace

cudaError_t cuda_wb97mv_grid_workspace(const CudaWb97mvGridShape& shape,
                                       const CudaWb97mvGridOptions& options,
                                       std::size_t& bytes) noexcept {
  Layout plan;
  bytes = 0;
  if (!layout(shape, options, plan)) return cudaErrorInvalidValue;
  bytes = plan.bytes;
  return cudaSuccess;
}

cudaError_t CudaWb97mvGridResponse::authenticate(cudaStream_t stream) noexcept {
  if (!ready_ || stream != reference_.stream) return cudaErrorInvalidResourceHandle;
  auto error = execution(stream, reference_.device_id);
  if (error != cudaSuccess) return error;
  error = reference_.authenticate(reference_.owner);
  if (error != cudaSuccess) ready_ = false;
  return error;
}

cudaError_t CudaWb97mvGridResponse::prepare(const CudaWb97mvGridShape& shape,
                                            const CudaWb97mvGridOptions& options,
                                            const CudaWb97mvGridReference& reference,
                                            void* workspace, std::size_t workspace_bytes) noexcept {
  ready_ = false;
  work_ = {};
  Layout plan;
  if (!layout(shape, options, plan) || workspace_bytes < plan.bytes || !reference.authenticate ||
      !reference.owner)
    return cudaErrorInvalidValue;
  auto error = execution(reference.stream, reference.device_id);
  if (error != cudaSuccess) return error;
  if ((error = reference.authenticate(reference.owner)) != cudaSuccess) return error;
  const void* inputs[] = {reference.basis, reference.points, reference.weights, reference.density};
  const std::size_t counts[] = {plan.basis_elements, 3 * shape.points, shape.points, plan.matrices};
  for (unsigned input = 0; input < 4; ++input) {
    if (!disjoint(workspace, plan.bytes / sizeof(double), inputs[input], counts[input]))
      return cudaErrorInvalidValue;
    if ((error = device_buffer(inputs[input], reference.device_id)) != cudaSuccess) return error;
  }
  if ((error = device_buffer(workspace, reference.device_id)) != cudaSuccess) return error;
  shape_ = shape;
  options_ = options;
  reference_ = reference;
  workspace_ = workspace;
  work_.workspace_bytes = plan.bytes;
  auto* arena = static_cast<double*>(workspace);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  if ((error = cudaMemsetAsync(arena, 0, plan.bytes, reference.stream)) != cudaSuccess)
    return drain(error, reference.stream);
  check_matrix<<<maps::blocks(plan.matrices), 128, 0, reference.stream>>>(
      reference.density, shape.nao, shape.spins, options.symmetry_absolute_tolerance,
      options.symmetry_relative_tolerance, status);
  error = cudaGetLastError();
  if (error == cudaSuccess)
    error = hessian_aot::launch_ao(reference.basis, shape.atoms, shape.primitives, shape.nao,
                                   reference.points, shape.points, 4, arena + plan.ao, status,
                                   reference.stream);
  if (error == cudaSuccess)
    error = features(shape, plan, arena, reference.density, arena + plan.base, reference.stream);
  if (error != cudaSuccess) return drain(error, reference.stream);
  discover<<<1, 1, 0, reference.stream>>>(
      arena + plan.base, reference.weights, reference.points, shape.points, shape.spins,
      reinterpret_cast<std::size_t*>(arena + plan.active),
      reinterpret_cast<std::size_t*>(arena + plan.count), status);
  if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, reference.stream);
  if ((error = cudaMemcpyAsync(&work_.active_nonlocal_points, arena + plan.count,
                               sizeof(std::size_t), cudaMemcpyDeviceToHost, reference.stream)) !=
      cudaSuccess)
    return drain(error, reference.stream);
  if ((error = numerical_status(status, reference.stream)) != cudaSuccess) return error;
  const auto active = work_.active_nonlocal_points;
  if (active && active > options.max_pair_evaluations / active) return cudaErrorInvalidValue;
  ready_ = true;
  if ((error = evaluate(false)) == cudaSuccess) error = authenticate(reference.stream);
  if (error != cudaSuccess) ready_ = false;
  return error;
}

cudaError_t CudaWb97mvGridResponse::evaluate(bool response) noexcept {
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  auto* arena = static_cast<double*>(workspace_);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  auto* active = reinterpret_cast<std::size_t*>(arena + plan.active);
  const auto stream = reference_.stream;
  if (shape_.spins == 1)
    ad::point_1<<<maps::blocks(shape_.points), 128, 0, stream>>>(
        arena + plan.base, arena + plan.direction, reference_.weights, shape_.points, response,
        arena + plan.coefficients, status);
  else
    ad::point_2<<<maps::blocks(shape_.points), 128, 0, stream>>>(
        arena + plan.base, arena + plan.direction, reference_.weights, shape_.points, response,
        arena + plan.coefficients, status);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return drain(error, stream);
  const auto count = work_.active_nonlocal_points;
  auto& pairs = response ? work_.response_pairs : work_.reference_pairs;
  const auto work = static_cast<std::uint64_t>(count) * count;
  if (work > UINT64_MAX - pairs) return drain(cudaErrorInvalidValue, stream);
  for (std::size_t row = 0; row < count; row += options_.pair_tile_points)
    for (std::size_t column = 0; column < count; column += options_.pair_tile_points) {
      const auto rows = std::min(count - row, options_.pair_tile_points);
      const auto columns = std::min(count - column, options_.pair_tile_points);
      ad::pairs<<<maps::blocks(rows * columns), 128, 0, stream>>>(
          arena + plan.base, arena + plan.direction, reference_.weights, reference_.points,
          shape_.points, shape_.spins, active, row, column, rows, columns, response,
          arena + plan.pairs, status);
      error = cudaGetLastError();
      if (error != cudaSuccess) return drain(error, stream);
      pairs += rows * columns;
      pair_reduce<<<maps::blocks(4 * rows), 128, 0, stream>>>(arena + plan.pairs, active, row, rows,
                                                              columns, shape_.points, shape_.spins,
                                                              arena + plan.coefficients, status);
      if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
    }
  if (!response && count) {
    local_beta<<<maps::blocks(count), 128, 0, stream>>>(reference_.weights, active, count,
                                                        shape_.points, shape_.spins,
                                                        arena + plan.coefficients, status);
    if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  }
  const auto panel = shape_.points * shape_.nao;
  for (unsigned spin = 0; spin < shape_.spins; ++spin) {
    auto* coefficients = arena + plan.coefficients + 5 * spin * shape_.points;
    error = maps::launch_potential(
        plan.extents, arena + plan.ao, arena + plan.ao + panel, arena + plan.ao + 2 * panel,
        arena + plan.ao + 3 * panel, coefficients + shape_.points, coefficients + 2 * shape_.points,
        coefficients + 3 * shape_.points, coefficients, coefficients + 4 * shape_.points,
        arena + (response ? plan.candidate : plan.reference) + spin * plan.matrix,
        arena + plan.tensor, plan.tensor_bytes, status, stream);
    if (error != cudaSuccess) return drain(error, stream);
  }
  check_matrix<<<maps::blocks(plan.matrices), 128, 0, stream>>>(
      arena + (response ? plan.candidate : plan.reference), shape_.nao, shape_.spins,
      options_.symmetry_absolute_tolerance, options_.symmetry_relative_tolerance, status);
  if ((error = cudaGetLastError()) != cudaSuccess) return drain(error, stream);
  return numerical_status(status, stream);
}

cudaError_t CudaWb97mvGridResponse::apply(const double* direction, double* output,
                                          cudaStream_t stream) noexcept {
  auto error = authenticate(stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan) || work_.response_actions == UINT64_MAX)
    return cudaErrorInvalidValue;
  const void* inputs[] = {reference_.basis,   reference_.points, reference_.weights,
                          reference_.density, workspace_,        direction};
  const std::size_t counts[] = {plan.basis_elements, 3 * shape_.points,           shape_.points,
                                plan.matrices,       plan.bytes / sizeof(double), plan.matrices};
  for (unsigned input = 0; input < 6; ++input)
    if (!disjoint(output, plan.matrices, inputs[input], counts[input]))
      return cudaErrorInvalidValue;
  if (!disjoint(direction, plan.matrices, workspace_, plan.bytes / sizeof(double)))
    return cudaErrorInvalidValue;
  for (const auto* pointer : {direction, static_cast<const double*>(output)})
    if ((error = device_buffer(pointer, reference_.device_id)) != cudaSuccess) return error;
  auto* arena = static_cast<double*>(workspace_);
  auto* status = reinterpret_cast<int*>(arena + plan.status);
  if ((error = cudaMemsetAsync(status, 0, sizeof(int), stream)) != cudaSuccess)
    return drain(error, stream);
  ++work_.response_actions;
  check_matrix<<<maps::blocks(plan.matrices), 128, 0, stream>>>(
      direction, shape_.nao, shape_.spins, options_.symmetry_absolute_tolerance,
      options_.symmetry_relative_tolerance, status);
  error = cudaGetLastError();
  if (error == cudaSuccess)
    error = features(shape_, plan, arena, direction, arena + plan.direction, stream);
  if (error != cudaSuccess) return drain(error, stream);
  if ((error = numerical_status(status, stream)) != cudaSuccess) return error;
  if ((error = evaluate(true)) != cudaSuccess) return error;
  if ((error = authenticate(stream)) != cudaSuccess) return error;
  error = cudaMemcpyAsync(output, arena + plan.candidate, plan.matrices * sizeof(double),
                          cudaMemcpyDeviceToDevice, stream);
  return drain(error, stream);
}

cudaError_t CudaWb97mvGridResponse::reference_potential(const double*& potential) noexcept {
  potential = nullptr;
  auto error = authenticate(reference_.stream);
  if (error != cudaSuccess) return error;
  Layout plan;
  if (!layout(shape_, options_, plan)) return cudaErrorInvalidValue;
  potential = static_cast<double*>(workspace_) + plan.reference;
  return cudaSuccess;
}
}  // namespace generativeqc::dft
