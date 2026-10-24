#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <new>
#include <vector>

#include "dft/hessian_aot.hpp"
#include "integrals/cuda_ecp_nuclear_response.hpp"
#include "integrals/ecp.hpp"
#include "response/ecp_quadrature.hpp"
#include "response/ecp_response.cuh"
#include "response/nuclear_integral_response.cuh"

namespace generativeqc::integrals {
namespace {
namespace maps = generativeqc_ecp_response_maps;
namespace quadrature = generativeqc_ecp_quadrature;
namespace ad = generativeqc_nuclear_integral_ad;
namespace source = generativeqc::generated;
bool add(std::size_t& value, std::size_t increment) {
  if (increment > SIZE_MAX - value) return false;
  value += increment;
  return true;
}
bool multiply(std::size_t first, std::size_t second, std::size_t& result) {
  if (second && first > SIZE_MAX / second) return false;
  result = first * second;
  return true;
}
struct Grid {
  std::size_t radial{}, sphere{}, angular{}, projector{};
};
struct Layout {
  std::size_t bytes{}, basis_count{}, matrix{}, vector{}, candidates{}, map_bytes{}, centers{},
      points{}, jets{}, projected{}, motion{}, measure{}, map_output{}, map_arena{}, coarse{},
      fine{}, potential{}, seed{}, status{};
  std::array<Grid, 2> grids{};
  CudaEcpNuclearWork work{};
  bool reserve(std::size_t count, std::size_t size, std::size_t& offset) {
    offset = bytes;
    std::size_t increment{};
    if (!multiply(count, size, increment) || !add(bytes, increment) || !add(bytes, 255))
      return false;
    bytes = bytes / 256 * 256;
    return true;
  }
};
cudaError_t layout(const CudaEcpNuclearShape& shape, std::span<const unsigned> centers,
                   const CudaEcpNuclearOptions& options, Layout& plan) {
  plan = {};
  if (!shape.atoms || !shape.primitives || !shape.nao || !shape.terms || centers.empty() ||
      shape.atoms > INT32_MAX || shape.primitives > INT32_MAX || shape.nao > INT32_MAX ||
      shape.terms > INT32_MAX || centers.size() > shape.atoms || shape.terms > options.max_terms ||
      !options.max_workspace_bytes || !options.max_ao_evaluations || !options.max_affine_terms ||
      !options.max_reduction_terms || !std::isfinite(options.symmetry_absolute_tolerance) ||
      !std::isfinite(options.symmetry_relative_tolerance) ||
      options.symmetry_absolute_tolerance < 0 || options.symmetry_relative_tolerance < 0)
    return cudaErrorInvalidValue;
  for (std::size_t index = 0; index < centers.size(); ++index)
    if (centers[index] >= shape.atoms || (index && centers[index - 1] >= centers[index]))
      return cudaErrorInvalidValue;
  const auto* ao = dft::hessian_aot::find("ao");
  if (!ao || ao->kind != dft::hessian_aot::Kind::ao || !ao->ao) return cudaErrorNotSupported;
  if (!multiply(shape.nao, shape.nao, plan.matrix) || !multiply(shape.atoms, 3, plan.vector))
    return cudaErrorInvalidValue;
  plan.basis_count = 3 * shape.atoms;
  if (!add(plan.basis_count, 2 * shape.primitives) || !add(plan.basis_count, 16 * shape.nao))
    return cudaErrorInvalidValue;
  std::size_t max_points{}, ao_lanes{}, projection_terms{}, reduction_terms{}, extra{};
  for (unsigned index = 0; index < 2; ++index) {
    const auto& grid = quadrature::grids[index];
    max_points = std::max(max_points, grid.points);
    std::size_t layers{}, evaluations{}, projections{}, reductions{}, matrix_sums{};
    if (!multiply(centers.size(), grid.radial_count, layers) ||
        !multiply(layers, grid.points, evaluations) ||
        !multiply(evaluations, shape.nao, evaluations) ||
        !multiply(evaluations, 160, projections) ||
        !multiply(grid.points + 16, shape.nao, reductions) ||
        !multiply(reductions, 2 * plan.vector, reductions) || !add(reductions, 4 * plan.vector) ||
        !multiply(plan.matrix, 4, matrix_sums))
      return cudaErrorInvalidValue;
    reductions = std::max(reductions, matrix_sums);
    if (!add(reductions, shape.terms) || !multiply(reductions, layers, reductions) ||
        !add(reductions, projections) || !add(plan.work.radial_layers_per_action, layers) ||
        !add(plan.work.ao_evaluations_per_action, evaluations) ||
        !add(projection_terms, projections) || !add(reduction_terms, reductions))
      return cudaErrorInvalidValue;
    for (const auto points : {grid.points, std::size_t{16}}) {
      const maps::Extents extents{shape.nao, points};
      std::size_t bytes{}, terms{};
      if (maps::workspace_matrix(extents, bytes) != cudaSuccess ||
          !maps::work_matrix(extents, terms) || !multiply(terms, layers, terms) ||
          !add(plan.work.matrix_affine_terms, terms))
        return cudaErrorInvalidValue;
      plan.map_bytes = std::max(plan.map_bytes, bytes);
      if (maps::workspace_weights(extents, bytes) != cudaSuccess ||
          !maps::work_weights(extents, terms) || !multiply(terms, layers, terms) ||
          !add(plan.work.contraction_affine_terms, terms))
        return cudaErrorInvalidValue;
      plan.map_bytes = std::max(plan.map_bytes, bytes);
    }
    auto& target = plan.grids[index];
    if (!plan.reserve(2 * grid.radial_count, sizeof(double), target.radial) ||
        !plan.reserve(3 * grid.points, sizeof(double), target.sphere) ||
        !plan.reserve(grid.points, sizeof(double), target.angular) ||
        !plan.reserve(16 * grid.points, sizeof(double), target.projector))
      return cudaErrorInvalidValue;
  }
  plan.work.projection_terms_per_action = projection_terms;
  plan.work.reduction_terms_bound = reduction_terms;
  if (plan.work.ao_evaluations_per_action > options.max_ao_evaluations ||
      plan.work.matrix_affine_terms > options.max_affine_terms ||
      plan.work.contraction_affine_terms > options.max_affine_terms ||
      reduction_terms > options.max_reduction_terms || !multiply(max_points, shape.nao, ao_lanes) ||
      !multiply(plan.matrix, 4, plan.candidates) || !multiply(plan.vector, 4, extra))
    return cudaErrorInvalidValue;
  plan.candidates = std::max(plan.candidates, extra);
  if (!plan.reserve(centers.size(), sizeof(unsigned), plan.centers) ||
      !plan.reserve(3 * max_points, sizeof(double), plan.points) ||
      !multiply(ao_lanes, 10, extra) || !plan.reserve(extra, sizeof(double), plan.jets) ||
      !multiply(shape.nao, 160, extra) || !plan.reserve(extra, sizeof(double), plan.projected) ||
      !plan.reserve(ao_lanes, sizeof(double), plan.motion) ||
      !plan.reserve(max_points, sizeof(double), plan.measure) ||
      !multiply(std::max(plan.matrix, ao_lanes), 2, extra) ||
      !plan.reserve(extra, sizeof(double), plan.map_output) ||
      !plan.reserve(plan.map_bytes, 1, plan.map_arena) ||
      !plan.reserve(plan.candidates, sizeof(double), plan.coarse) ||
      !plan.reserve(plan.candidates, sizeof(double), plan.fine) ||
      !plan.reserve(5, sizeof(double), plan.potential) ||
      !plan.reserve(1, sizeof(double), plan.seed) || !plan.reserve(1, sizeof(int), plan.status) ||
      plan.bytes > options.max_workspace_bytes)
    return cudaErrorInvalidValue;
  plan.work.workspace_bytes = plan.bytes;
  plan.work.centers = centers.size();
  return cudaSuccess;
}
struct Range {
  const void* pointer{};
  std::size_t bytes{}, alignment{alignof(double)};
};
bool disjoint(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first),
             right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left >= first_bytes : left - right >= second_bytes;
}
cudaError_t device_buffer(Range range, int device) {
  if (!range.pointer || reinterpret_cast<std::uintptr_t>(range.pointer) % range.alignment)
    return cudaErrorInvalidValue;
  cudaPointerAttributes attributes{};
  const auto error = cudaPointerGetAttributes(&attributes, range.pointer);
  if (error != cudaSuccess) return error;
  return attributes.type == cudaMemoryTypeDevice && attributes.device == device
             ? cudaSuccess
             : cudaErrorInvalidValue;
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
__global__ void finite(const double* values, std::size_t count, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(values[index])) atomicExch(status, 1);
}
__global__ void validate(CudaEcpNuclearShape shape, const double* basis, const core::EcpTerm* terms,
                         const unsigned* centers, std::size_t center_count, int* status) {
  const auto* primitives = basis + 3 * shape.atoms;
  const auto* records = primitives + 2 * shape.primitives;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < max(max(shape.primitives, shape.nao), shape.terms);
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    if (index < shape.primitives && !(primitives[2 * index] > 0)) atomicExch(status, 1);
    if (index < shape.terms) {
      const auto term = terms[index];
      bool found = false;
      for (std::size_t center = 0; center < center_count; ++center)
        found |= centers[center] == term.atom_index;
      if (!found || term.channel < -1 || term.channel > 3 || term.power > 4 ||
          !(term.exponent > 0) || !isfinite(term.exponent) || !isfinite(term.coefficient))
        atomicExch(status, 1);
    }
    if (index >= shape.nao) continue;
    const auto* record = records + 16 * index;
    const auto atom = record[0], begin = record[1], count = record[2], components = record[3];
    if (atom < 0 || atom >= shape.atoms || floor(atom) != atom || begin < 0 ||
        begin > shape.primitives || floor(begin) != begin || count < 1 ||
        count > shape.primitives - begin || floor(count) != count || components < 1 ||
        components > 3 || floor(components) != components) {
      atomicExch(status, 1);
      continue;
    }
    for (unsigned component = 0; component < static_cast<unsigned>(components); ++component) {
      double degree{};
      for (unsigned axis = 0; axis < 3; ++axis) {
        const auto power = record[4 + 4 * component + axis];
        if (power < 0 || power > 3 || floor(power) != power) atomicExch(status, 1);
        degree += power;
      }
      if (degree > 3) atomicExch(status, 1);
    }
  }
}
__global__ void symmetric(const double* values, std::size_t nao, double absolute, double relative,
                          int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nao * nao;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto value = values[index], other = values[(index % nao) * nao + index / nao];
    if (!isfinite(value) || !isfinite(other) ||
        fabs(value - other) > absolute + relative * fmax(fabs(value), fabs(other)))
      atomicExch(status, 1);
  }
}
__global__ void nodes(const double* basis, const double* radial, const double* sphere,
                      std::size_t points, unsigned center, std::size_t layer, double* output) {
  // Linked AO leaves do not consume an abort flag. Always initialize their
  // submitted points, including after a previous radial numerical failure.
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 3 * points;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output[index] = source::ecp_node_displacement(basis[3 * center + index % 3], radial[2 * layer],
                                                  sphere[index], 0);
}
__global__ void potentials(const core::EcpTerm* terms, std::size_t count, const double* radial,
                           std::size_t layer, unsigned center, double* output, int* status) {
  if (threadIdx.x) return;
  const EcpRadialPoint point{radial[2 * layer], radial[2 * layer + 1]};
  source::ecp_potentials(terms, static_cast<int>(count), point, center, output);
  for (unsigned channel = 0; channel < 5; ++channel)
    if (!isfinite(output[channel])) atomicExch(status, 1);
}
__global__ void project(const double* jets, const double* projector, std::size_t points,
                        std::size_t nao, double* output, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 160 * nao;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto orbital = index % nao, row = index / nao % 16, jet = index / (16 * nao);
    double sum{};
    for (std::size_t point = 0; point < points; ++point)
      sum += source::ecp_projection_term(1, projector[row * points + point],
                                         jets[(jet * points + point) * nao + orbital]);
    output[index] = sum;
    if (!isfinite(sum)) atomicExch(status, 1);
  }
}
__global__ void channel_inputs(CudaEcpNuclearShape shape, const double* basis, const double* jets,
                               const double* direction, const double* potential,
                               const double* angular, std::size_t rows, unsigned center,
                               bool nonlocal, double* motion, double* measure, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto* records = basis + 3 * shape.atoms + 2 * shape.primitives;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < rows * shape.nao;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto orbital = index % shape.nao, row = index / shape.nao;
    const auto atom = static_cast<std::size_t>(records[16 * orbital]);
    double left[3]{}, right[3], result[3];
    for (unsigned axis = 0; axis < 3; ++axis)
      right[axis] = source::ecp_node_displacement(direction[3 * center + axis], 0, 0,
                                                  direction[3 * atom + axis]);
    maps::ao_motion(jets + index, rows * shape.nao, left, right, result);
    motion[index] = result[1];
    if (!orbital) {
      const auto channel = row < 1 ? 1 : row < 4 ? 2 : row < 9 ? 3 : 4;
      measure[row] = nonlocal ? potential[channel]
                              : source::ecp_projection_term(angular[row], 1, potential[0]);
    }
  }
}
__global__ void accumulate_matrix(const double* values, std::size_t matrix, unsigned channel,
                                  double* candidate, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < matrix;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    candidate[channel * matrix + index] += values[matrix + index];
    candidate[(2 + channel) * matrix + index] += values[index];
  }
}
__global__ void contract_jets(CudaEcpNuclearShape shape, const double* basis, const double* jets,
                              const double* direction, const double* weights, std::size_t rows,
                              unsigned center, unsigned channel, double* candidate, int* status) {
  if (atomicAdd(status, 0)) return;
  const auto* records = basis + 3 * shape.atoms + 2 * shape.primitives;
  const auto coordinates = 3 * shape.atoms, lanes = rows * shape.nao;
  for (std::size_t coordinate = blockIdx.x * blockDim.x + threadIdx.x; coordinate < coordinates;
       coordinate += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double gradient{}, hvp{};
    for (std::size_t index = 0; index < lanes; ++index) {
      const auto atom = static_cast<std::size_t>(records[16 * (index % shape.nao)]);
      double left[3]{}, right[3], motion[3], result[4];
      left[coordinate % 3] =
          source::ecp_node_displacement(center == coordinate / 3, 0, 0, atom == coordinate / 3);
      for (unsigned axis = 0; axis < 3; ++axis)
        right[axis] = source::ecp_node_displacement(direction[3 * center + axis], 0, 0,
                                                    direction[3 * atom + axis]);
      maps::ao_motion(jets + index, lanes, left, right, motion);
      if (!ad::contract(jets[index], motion[0], motion[2], weights[lanes + index], weights[index],
                        1, 0, result))
        atomicExch(status, 1);
      gradient += result[2];
      hvp += result[3];
    }
    candidate[channel * coordinates + coordinate] += gradient;
    candidate[(2 + channel) * coordinates + coordinate] += hvp;
  }
}
__global__ void convergence(const double* coarse, const double* fine, std::size_t extent,
                            bool derivatives, int* status) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 4 * extent;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!source::ecp_grid_pair_accepted(coarse[index], fine[index],
                                        derivatives || index >= 2 * extent))
      atomicExch(status, 1);
}
}  // namespace

struct CudaEcpNuclearResponse::Impl {
  CudaEcpNuclearShape shape{};
  CudaEcpNuclearOptions options{};
  CudaEcpNuclearReference reference{};
  Layout plan;
  CudaEcpNuclearWork receipt{};
  std::vector<unsigned> centers;
  void* arena{};
  template <class Value = double>
  Value* at(std::size_t offset) const {
    return reinterpret_cast<Value*>(static_cast<char*>(arena) + offset);
  }
  cudaError_t authenticate(cudaStream_t stream) const {
    if (stream != reference.stream) return cudaErrorInvalidResourceHandle;
    int device{-1};
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    if (device != reference.device_id) return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture{};
    if ((error = cudaStreamIsCapturing(stream, &capture)) != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    return reference.authenticate(reference.owner);
  }
  cudaError_t buffers(std::span<const Range> inputs, std::span<const Range> outputs) const {
    for (const auto& input : inputs) {
      if (!input.bytes) continue;
      if (!disjoint(arena, plan.bytes, input.pointer, input.bytes)) return cudaErrorInvalidValue;
      if (const auto error = device_buffer(input, reference.device_id); error != cudaSuccess)
        return error;
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& output = outputs[index];
      if (!disjoint(arena, plan.bytes, output.pointer, output.bytes)) return cudaErrorInvalidValue;
      if (const auto error = device_buffer(output, reference.device_id); error != cudaSuccess)
        return error;
      for (const auto& input : inputs)
        if (input.bytes && !disjoint(output.pointer, output.bytes, input.pointer, input.bytes))
          return cudaErrorInvalidValue;
      for (std::size_t previous = 0; previous < index; ++previous)
        if (!disjoint(output.pointer, output.bytes, outputs[previous].pointer,
                      outputs[previous].bytes))
          return cudaErrorInvalidValue;
    }
    return cudaSuccess;
  }
  template <bool Contract>
  cudaError_t execute(const double* direction, CudaEcpNuclearWeights weights,
                      std::span<const Range> outputs, cudaStream_t stream) {
    if (receipt.attempted_actions == UINT64_MAX) return cudaErrorInvalidValue;
    ++receipt.attempted_actions;
    auto error = authenticate(stream);
    if (error != cudaSuccess) return error;
    const Range inputs[]{
        {reference.basis, plan.basis_count * sizeof(double)},
        {reference.terms, shape.terms * sizeof(core::EcpTerm), alignof(core::EcpTerm)},
        {direction, plan.vector * sizeof(double)},
        {weights.density, Contract ? plan.matrix * sizeof(double) : 0},
        {weights.direction_density, Contract ? plan.matrix * sizeof(double) : 0}};
    if ((error = buffers(inputs, outputs)) != cudaSuccess) return error;
    const auto work_terms =
        Contract ? plan.work.contraction_affine_terms : plan.work.matrix_affine_terms;
    if (receipt.radial_layers > UINT64_MAX - plan.work.radial_layers_per_action ||
        receipt.ao_evaluations > UINT64_MAX - plan.work.ao_evaluations_per_action ||
        receipt.projection_terms > UINT64_MAX - plan.work.projection_terms_per_action ||
        receipt.affine_terms > UINT64_MAX - work_terms)
      return cudaErrorInvalidValue;
    auto* status = at<int>(plan.status);
    error = cudaMemsetAsync(status, 0, sizeof(int), stream);
    if (error == cudaSuccess) {
      finite<<<maps::blocks(plan.vector), 128, 0, stream>>>(direction, plan.vector, status);
      error = cudaGetLastError();
    }
    if constexpr (Contract)
      for (const auto* panel : {weights.density, weights.direction_density})
        if (error == cudaSuccess) {
          symmetric<<<maps::blocks(plan.matrix), 128, 0, stream>>>(
              panel, shape.nao, options.symmetry_absolute_tolerance,
              options.symmetry_relative_tolerance, status);
          error = cudaGetLastError();
        }
    if (error == cudaSuccess) error = gate(status, stream);
    if (error != cudaSuccess) return drain(error, stream);
    error = cudaMemsetAsync(at(plan.coarse), 0, plan.candidates * sizeof(double), stream);
    if (error == cudaSuccess)
      error = cudaMemsetAsync(at(plan.fine), 0, plan.candidates * sizeof(double), stream);
    for (unsigned grid_index = 0; grid_index < 2 && error == cudaSuccess; ++grid_index) {
      const auto& grid = quadrature::grids[grid_index];
      const auto& device = plan.grids[grid_index];
      auto* candidate = at(grid_index ? plan.fine : plan.coarse);
      for (const auto center : centers) {
        for (std::size_t layer = 0; layer < grid.radial_count && error == cudaSuccess; ++layer) {
          nodes<<<maps::blocks(3 * grid.points), 128, 0, stream>>>(
              reference.basis, at(device.radial), at(device.sphere), grid.points, center, layer,
              at(plan.points));
          error = cudaGetLastError();
          if (error == cudaSuccess) {
            potentials<<<1, 1, 0, stream>>>(reference.terms, shape.terms, at(device.radial), layer,
                                            center, at(plan.potential), status);
            error = cudaGetLastError();
          }
          if (error == cudaSuccess) {
            error = dft::hessian_aot::launch_ao(reference.basis, shape.atoms, shape.primitives,
                                                shape.nao, at(plan.points), grid.points, 10,
                                                at(plan.jets), status, stream);
            ++receipt.radial_layers;
            receipt.ao_evaluations += grid.points * shape.nao;
          }
          if (error == cudaSuccess) {
            project<<<maps::blocks(160 * shape.nao), 128, 0, stream>>>(
                at(plan.jets), at(device.projector), grid.points, shape.nao, at(plan.projected),
                status);
            error = cudaGetLastError();
            receipt.projection_terms += 160 * shape.nao * grid.points;
          }
          for (unsigned channel = 0; channel < 2 && error == cudaSuccess; ++channel) {
            const auto rows = channel ? std::size_t{16} : grid.points;
            const auto* jets = at(channel ? plan.projected : plan.jets);
            channel_inputs<<<maps::blocks(rows * shape.nao), 128, 0, stream>>>(
                shape, reference.basis, jets, direction, at(plan.potential), at(device.angular),
                rows, center, channel, at(plan.motion), at(plan.measure), status);
            error = cudaGetLastError();
            if (error != cudaSuccess) break;
            const maps::Extents extents{shape.nao, rows};
            std::size_t terms{};
            if constexpr (Contract) {
              (void)maps::work_weights(extents, terms);
              error =
                  maps::launch_weights(extents, jets, at(plan.seed), at(plan.motion),
                                       weights.direction_density, weights.density, at(plan.measure),
                                       at(plan.map_output), at(plan.map_output) + rows * shape.nao,
                                       at(plan.map_arena), plan.map_bytes, status, stream);
              if (error == cudaSuccess) {
                contract_jets<<<maps::blocks(plan.vector), 128, 0, stream>>>(
                    shape, reference.basis, jets, direction, at(plan.map_output), rows, center,
                    channel, candidate, status);
                error = cudaGetLastError();
              }
            } else {
              (void)maps::work_matrix(extents, terms);
              error = maps::launch_matrix(extents, jets, at(plan.seed), at(plan.motion),
                                          at(plan.measure), at(plan.map_output),
                                          at(plan.map_output) + plan.matrix, at(plan.map_arena),
                                          plan.map_bytes, status, stream);
              if (error == cudaSuccess) {
                accumulate_matrix<<<maps::blocks(plan.matrix), 128, 0, stream>>>(
                    at(plan.map_output), plan.matrix, channel, candidate, status);
                error = cudaGetLastError();
              }
            }
            receipt.affine_terms += terms;
          }
        }
        if (error != cudaSuccess) break;
      }
    }
    const auto extent = Contract ? plan.vector : plan.matrix;
    if (error == cudaSuccess) {
      convergence<<<maps::blocks(4 * extent), 128, 0, stream>>>(at(plan.coarse), at(plan.fine),
                                                                extent, Contract, status);
      error = cudaGetLastError();
    }
    if constexpr (!Contract)
      for (std::size_t index = 0; index < 4 && error == cudaSuccess; ++index) {
        symmetric<<<maps::blocks(plan.matrix), 128, 0, stream>>>(
            at(plan.fine) + index * plan.matrix, shape.nao, options.symmetry_absolute_tolerance,
            options.symmetry_relative_tolerance, status);
        error = cudaGetLastError();
      }
    if (error == cudaSuccess) error = gate(status, stream);
    error = drain(error, stream);
    if (error != cudaSuccess) return error;
    if ((error = authenticate(stream)) != cudaSuccess) return error;
    for (std::size_t index = 0; index < outputs.size() && error == cudaSuccess; ++index)
      error =
          cudaMemcpyAsync(const_cast<void*>(outputs[index].pointer), at(plan.fine) + index * extent,
                          outputs[index].bytes, cudaMemcpyDeviceToDevice, stream);
    error = drain(error, stream);
    if (error == cudaSuccess) {
      if constexpr (Contract)
        ++receipt.contraction_actions;
      else
        ++receipt.value_actions;
    }
    return error;
  }
};

CudaEcpNuclearResponse::CudaEcpNuclearResponse() = default;
CudaEcpNuclearResponse::~CudaEcpNuclearResponse() = default;
cudaError_t cuda_ecp_nuclear_workspace(const CudaEcpNuclearShape& shape,
                                       std::span<const unsigned> centers,
                                       const CudaEcpNuclearOptions& options,
                                       std::size_t& bytes) noexcept {
  bytes = 0;
  Layout plan;
  const auto error = layout(shape, centers, options, plan);
  if (error == cudaSuccess) bytes = plan.bytes;
  return error;
}
cudaError_t CudaEcpNuclearResponse::prepare(const CudaEcpNuclearShape& shape,
                                            std::span<const unsigned> centers,
                                            const CudaEcpNuclearOptions& options,
                                            const CudaEcpNuclearReference& reference,
                                            void* workspace, std::size_t workspace_bytes) noexcept {
  impl_.reset();
  try {
    auto state = std::make_unique<Impl>();
    auto error = layout(shape, centers, options, state->plan);
    if (error != cudaSuccess) return error;
    if (!reference.basis || !reference.terms || reference.device_id < 0 || !reference.stream ||
        !reference.authenticate || !reference.owner || !workspace ||
        workspace_bytes < state->plan.bytes)
      return cudaErrorInvalidValue;
    state->shape = shape;
    state->options = options;
    state->reference = reference;
    state->centers.assign(centers.begin(), centers.end());
    state->arena = workspace;
    const auto& plan = state->plan;
    if ((error = state->authenticate(reference.stream)) != cudaSuccess) return error;
    if ((error = device_buffer({workspace, plan.bytes}, reference.device_id)) != cudaSuccess)
      return error;
    const Range inputs[]{
        {reference.basis, plan.basis_count * sizeof(double)},
        {reference.terms, shape.terms * sizeof(core::EcpTerm), alignof(core::EcpTerm)}};
    if ((error = state->buffers(inputs, {})) != cudaSuccess) return error;
    // Every map input/cache is initialized even on rejected numerical paths;
    // compiler finite-input gates need not consume the preceding abort flag.
    error = cudaMemsetAsync(workspace, 0, plan.bytes, reference.stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(state->at<unsigned>(plan.centers), centers.data(),
                              centers.size_bytes(), cudaMemcpyHostToDevice, reference.stream);
    const double seed = 1;
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(state->at(plan.seed), &seed, sizeof(seed), cudaMemcpyHostToDevice,
                              reference.stream);
    for (unsigned index = 0; index < 2 && error == cudaSuccess; ++index) {
      const auto& grid = quadrature::grids[index];
      const auto& device = plan.grids[index];
      const std::array<std::pair<const double*, std::size_t>, 4> arrays{
          {{grid.radial, 2 * grid.radial_count},
           {grid.sphere, 3 * grid.points},
           {grid.angular, grid.points},
           {grid.projector, 16 * grid.points}}};
      const std::size_t offsets[]{device.radial, device.sphere, device.angular, device.projector};
      for (unsigned slot = 0; slot < 4 && error == cudaSuccess; ++slot)
        error = cudaMemcpyAsync(state->at(offsets[slot]), arrays[slot].first,
                                arrays[slot].second * sizeof(double), cudaMemcpyHostToDevice,
                                reference.stream);
    }
    if (error == cudaSuccess) {
      finite<<<maps::blocks(plan.basis_count), 128, 0, reference.stream>>>(
          reference.basis, plan.basis_count, state->at<int>(plan.status));
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
      validate<<<maps::blocks(std::max({shape.primitives, shape.nao, shape.terms})), 128, 0,
                 reference.stream>>>(shape, reference.basis, reference.terms,
                                     state->at<unsigned>(plan.centers), centers.size(),
                                     state->at<int>(plan.status));
      error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = gate(state->at<int>(plan.status), reference.stream);
    error = drain(error, reference.stream);
    if (error != cudaSuccess) return error;
    if ((error = state->authenticate(reference.stream)) != cudaSuccess) return error;
    state->receipt = plan.work;
    impl_ = std::move(state);
    return cudaSuccess;
  } catch (const std::bad_alloc&) {
    return cudaErrorMemoryAllocation;
  } catch (...) {
    return cudaErrorUnknown;
  }
}
cudaError_t CudaEcpNuclearResponse::values(const double* direction,
                                           const CudaEcpNuclearValues& output,
                                           cudaStream_t stream) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  const auto bytes = impl_->plan.matrix * sizeof(double);
  const Range ranges[]{{output.local, bytes},
                       {output.nonlocal, bytes},
                       {output.direction_local, bytes},
                       {output.direction_nonlocal, bytes}};
  return impl_->execute<false>(direction, {}, ranges, stream);
}
cudaError_t CudaEcpNuclearResponse::contract(const double* direction,
                                             const CudaEcpNuclearWeights& weights,
                                             const CudaEcpNuclearVectors& output,
                                             cudaStream_t stream) noexcept {
  if (!impl_) return cudaErrorInvalidResourceHandle;
  const auto bytes = impl_->plan.vector * sizeof(double);
  const Range ranges[]{{output.gradient_local, bytes},
                       {output.gradient_nonlocal, bytes},
                       {output.hvp_local, bytes},
                       {output.hvp_nonlocal, bytes}};
  return impl_->execute<true>(direction, weights, ranges, stream);
}
const CudaEcpNuclearWork& CudaEcpNuclearResponse::work() const noexcept {
  static const CudaEcpNuclearWork empty;
  return impl_ ? impl_->receipt : empty;
}
}  // namespace generativeqc::integrals
